## %%
"""multi_agent_csv_ptbr.py – Núcleo do sistema multi‑agente

Este módulo define o workflow LangGraph/LLM que responde perguntas sobre um CSV.
Ele foi refatorado para ser *importável* em ambientes Streamlit ou notebooks
sem exigir entrada interativa de terminal.  A chave OPENAI_API_KEY é validada
somente quando `run_workflow` é chamado, permitindo que a aplicação defina a
chave antes da execução (via variável de ambiente, `st.secrets` ou parâmetro
explícito).
"""

from __future__ import annotations

import json
import os
import re
import traceback
import types
from typing_extensions import TypedDict

import pandas as pd
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

# -----------------------------------------------------------------------------
# Função utilitária para garantir a presença da OPENAI_API_KEY
# -----------------------------------------------------------------------------

def ensure_openai_key() -> None:
    """Verifica se OPENAI_API_KEY está definida; tenta resgatar de st.secrets.

    Lógica:
    1. Se já estiver em `os.environ`, ok.
    2. Se estiver rodando em Streamlit e a chave estiver em `st.secrets`, copia‑a.
    3. Caso contrário, se for um terminal interativo (sys.stdin.isatty()), pede
       via `getpass` (útil em notebooks e CLI).
    4. Se ainda faltar, lança *EnvironmentError* para que a aplicação trate.
    """

    if os.getenv("OPENAI_API_KEY"):
        return  # já configurada

    # 2) Tenta st.secrets se estiver em ambiente Streamlit
    try:
        import streamlit as st  # import leve – só falhará se Streamlit não estiver instalado

        key = st.secrets.get("OPENAI_API_KEY", "")  # type: ignore[attr-defined]
        if key:
            os.environ["OPENAI_API_KEY"] = key
            return
    except ModuleNotFoundError:
        pass  # Streamlit não disponível – segue adiante

    # 3) Fallback interativo
    try:
        import sys
        if sys.stdin and sys.stdin.isatty():
            from getpass import getpass

            os.environ["OPENAI_API_KEY"] = getpass("🔑 Digite sua OPENAI_API_KEY: ")
            return
    except Exception:
        pass  # Qualquer erro cai para o passo 4

    # 4) Falhou – lança erro
    raise EnvironmentError(
        "OPENAI_API_KEY não definida. Defina a variável de ambiente, adicione‑a em st.secrets "
        "ou forneça via parâmetro openai_api_key no run_workflow()."
    )

# -----------------------------------------------------------------------------
# Agent 1 – Data Loader
# -----------------------------------------------------------------------------
class DLState(TypedDict):
    csv_path: str
    user_question: str
    schema: dict
    df: pd.DataFrame


def agent_data_loader(state: DLState):
    path = state["csv_path"]
    print("[Loader] lendo CSV →", path)
    df = pd.read_csv(path)

    columns: dict[str, dict] = {}
    for col in df.columns:
        col_data = df[col]
        columns[col] = {
            "dtype": str(col_data.dtype),
            "sample_values": [str(v) for v in col_data.dropna().unique()[:3]],
        }
    schema = {"columns": columns, "row_count": len(df)}
    print("[Loader] esquema pronto – linhas:", schema["row_count"])
    return {"schema": schema, "df": df}

# -----------------------------------------------------------------------------
# Instância compartilhada de modelos OpenAI
# -----------------------------------------------------------------------------
print("[Setup] Preparando modelos OpenAI… (adiado até a primeira chamada)")
llm: ChatOpenAI | None = None
llm_code: ChatOpenAI | None = None


def _init_llms() -> None:
    global llm, llm_code
    if llm is None or llm_code is None:
        ensure_openai_key()
        llm = ChatOpenAI(model="gpt-4.1-nano-2025-04-14", temperature=0)
        llm_code = ChatOpenAI(model="gpt-4.1-mini-2025-04-14", temperature=0)

# -----------------------------------------------------------------------------
# Agent 2 – Code Analyst (LLM)
# -----------------------------------------------------------------------------
class CAState(DLState):
    analysis_spec: dict  # {executable_code, explanation}
    follow_up: str  # optional follow‑up prompt from Agent 3


def build_analyst_prompt(schema: dict, question: str, follow_up: str | None = None) -> str:
    base = f"""
    # Papel
    Você é um engenheiro de dados Python encarregado de gerar o código que responde à pergunta do usuário.

    # Ambiente
    - DataFrame já disponível: df  
    - Bibliotecas permitidas: pandas (pd) e numpy (np)  
    - Proibido abrir/ler arquivos externos (ex.: pd.read_csv, open())

    # Saída obrigatória (JSON puro, sem markdown)
    {{
    "executable_code": "<código Python que resolve a pergunta; máx. 30 linhas>",
    "explanation": "<1-3 frases descrevendo a lógica>"
    }}

    # Diretrizes de código
    - Comece com `import pandas as pd`  
    - Utilize todas as colunas que sejam relevantes para responder de forma completa a pergunta, sem poupar detalhes; mencione-as no código  
    - Prefira operações idiomáticas (query, groupby, agg)
    - Sempre quebre a tarefa em etapas lógicas
    - O resultado final deve ser atribuído a `result_obj` em formato intuitivo.
    - O resultado final sempre deve contextualizar a resposta ao usuário.
    - NÃO inclua somente a resposta, responda com todas as informações relevantes para o contexto da pergunta (Nomes, Valores, Datas).

    {f'MENSAGEM DE SEGUIMENTO: {follow_up}' if follow_up else ''}

    Esquema de df: {json.dumps(schema, ensure_ascii=False)}

    Pergunta do usuário: {question}
    """
    return base


def agent_code_analyst(state: CAState):
    _init_llms()
    assert llm_code is not None  # for type checkers

    schema = state["schema"]
    question = state["user_question"]
    follow_up = state.get("follow_up")
    prompt = build_analyst_prompt(schema, question, follow_up)

    print("[Analyst] prompt →", prompt[:140].replace("\n", " "), "…")
    json_reply = llm_code.invoke(prompt).content
    json_reply = re.sub(r"```[a-zA-Z]*", "", json_reply).strip()
    print("[Analyst] raw reply →", json_reply[:140], "…")
    try:
        spec = json.loads(json_reply)
    except json.JSONDecodeError as e:
        raise ValueError("Analyst LLM não retornou JSON válido") from e
    print("[Analyst] spec →", spec)
    return {"analysis_spec": spec, "follow_up": ""}  # zera follow_up após uso

# -----------------------------------------------------------------------------
# Agent 3 – Natural‑Language Responder (LLM)
# -----------------------------------------------------------------------------
class NLState(CAState):
    result_obj: object
    answer: str
    needs_follow_up: bool


def agent_nl_responder(state: NLState):
    _init_llms()
    assert llm is not None

    result_obj = state["result_obj"]
    question = state["user_question"]
    explanation = state["analysis_spec"].get("explanation", "")

    # Detecta resultado vazio/erro
    invalid = result_obj is None or (isinstance(result_obj, str) and result_obj.startswith("ERROR"))
    if invalid:
        follow_up = (
            "O código anterior produziu um resultado vazio ou com erro. Refinar código ou adicionar agregação para retornar "
            "uma resposta significativa (atribua-a a result_obj)."
        )
        print("[Responder] resultado inválido – solicitando novo código →", follow_up)
        return {
            "answer": "Não consegui calcular isso – tentando novamente…",
            "needs_follow_up": True,
            "follow_up": follow_up,
        }

    # Resposta normal
    print("[Responder] result_obj →", repr(result_obj)[:120])
    prompt = f"""
    # Papel
    Você é um assistente de dados cordial, preciso e direto.

    # Tarefa
    Responda à pergunta do usuário usando exclusivamente o conteúdo de `result_obj`.

    # Regras de estilo (máx. 100 palavras)
    - Mencione naturalmente as colunas relevantes entre “aspas”.  
    - Destaque números-chave (totais, médias, percentuais) com até dois decimais.  
    - Não inclua trechos de código, visualizações nem ofertas adicionais.
    - Responda com uma linguagem limpa, detalhando a resposta com todas as informações disponíveis (datas, nomes e valores) e relevantes.
    - NÃO explique o código, apenas responda à pergunta explicando a lógica usada para obtê-la.

    Pergunta do usuário: {question}
    Trecho de dados (repr): {result_obj!r}
    """
    print("[Responder] prompt →", prompt[:140].replace("\n", " "), "…")
    answer = llm.invoke(prompt).content.strip()
    return {"answer": answer, "needs_follow_up": False}

# -----------------------------------------------------------------------------
# Execução do código gerado pelo analista
# -----------------------------------------------------------------------------

def execute_code_node(state: NLState):
    """Executa o código Python gerado pelo Analyst com salvaguardas."""
    df = state["df"]
    code = state["analysis_spec"]["executable_code"]
    print("[Execute] executando código →\n", code)

    # Cria um módulo pandas seguro que bloqueia read_csv
    safe_pd = types.ModuleType("safe_pandas")
    for attr in dir(pd):
        setattr(safe_pd, attr, getattr(pd, attr))

    def _blocked(*_, **__):
        raise RuntimeError("pd.read_csv está desabilitado dentro do código de análise. Use o DataFrame `df` fornecido.")

    safe_pd.read_csv = _blocked

    local_ns = {"pd": safe_pd, "df": df}
    baseline = set(local_ns)
    try:
        exec(code, {}, local_ns)
        result_obj = local_ns.get("result_obj")
        if result_obj is None:
            # Se result_obj não tiver sido definido explicitamente, tenta pegar última variável criada
            new_vars = [k for k in local_ns if k not in baseline]
            result_obj = local_ns[new_vars[-1]] if new_vars else None
    except Exception as e:
        tb = traceback.format_exc(limit=2)
        result_obj = f"ERROR: {e}\n{tb}"
    return {"result_obj": result_obj}

# -----------------------------------------------------------------------------
# Construção do LangGraph com laço de feedback
# -----------------------------------------------------------------------------
State = NLState
builder = StateGraph(State)

builder.add_node("load_csv", agent_data_loader)
builder.add_node("analyze", agent_code_analyst)
builder.add_node("execute", execute_code_node)
builder.add_node("respond", agent_nl_responder)

builder.add_edge(START, "load_csv")
builder.add_edge("load_csv", "analyze")
builder.add_edge("analyze", "execute")
builder.add_edge("execute", "respond")

# Condição para reanálise

def decide_next(state: NLState):
    return "retry" if state.get("needs_follow_up") else "done"

builder.add_conditional_edges(
    "respond",
    decide_next,
    {
        "retry": "analyze",
        "done": END,
    },
)

compiled = builder.compile()

# -----------------------------------------------------------------------------
# Função utilitária para aplicação externa (Streamlit, CLI, etc.)
# -----------------------------------------------------------------------------

def run_workflow(
    csv_path: str,
    question: str,
    *,
    openai_api_key: str | None = None,
    verbose: bool = True,
) -> str:
    """Executa o workflow e devolve a resposta em português.

    Parâmetros
    ----------
    csv_path : str
        Caminho para o arquivo CSV.
    question : str
        Pergunta em linguagem natural sobre os dados.
    openai_api_key : str | None, opcional
        Se fornecido, substitui/define a variável de ambiente `OPENAI_API_KEY`.
    verbose : bool, padrão True
        Se True, imprime a resposta final no stdout.
    """

    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key

    ensure_openai_key()
    _init_llms()

    out = compiled.invoke({"csv_path": csv_path, "user_question": question})
    answer = out["answer"]
    if verbose:
        print("[Resposta Final]", answer)
    return answer