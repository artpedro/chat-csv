from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Importa o workflow multi‑agente
# ---------------------------------------------------------------------------
from streamlit_agents import run_workflow

# ---------------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ChatCSV",
    page_icon="📋",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Funções utilitárias
# ---------------------------------------------------------------------------

def sanitize_markdown(text: str) -> str:
    """Escapa caracteres que podem quebrar o Markdown padrão do Streamlit.
    Atualmente apenas o símbolo "$" é escapado.
    """
    return text.replace("$", r"\$")

# ==========================================================================
# 1️⃣  Cabeçalho
# ==========================================================================

st.title("📊 ChatCSV")
st.caption("Análise inteligente de CSV com IA multi‑agente (LangGraph + OpenAI)")

# ==========================================================================
# 2️⃣  Configurações avançadas
# ==========================================================================

with st.expander("⚙️ Configurações necessárias"):
    openai_key = st.text_input(
        "🔑 OPENAI_API_KEY",
        type="password",
        help="A chave de API da OpenAI é necessária para executar o modelo de linguagem.",
    )
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key.strip()

# ==========================================================================
# 3️⃣  Upload do CSV + pré‑visualização
# ==========================================================================

st.subheader("1. Upload do CSV")

uploaded_file = st.file_uploader(
    "📂 Arraste ou selecione um arquivo CSV (até 200 MB)",
    type=["csv"],
)

if uploaded_file is not None:
    try:
        df_preview = pd.read_csv(io.BytesIO(uploaded_file.getvalue()), nrows=200)
        with st.expander("👀 Pré‑visualizar primeiras 200 linhas"):
            st.dataframe(df_preview, use_container_width=True, height=300)
    except Exception as e:
        st.warning(f"⚠️ Não foi possível pré‑visualizar o CSV: {e}")
    finally:
        uploaded_file.seek(0)

# ==========================================================================
# 4️⃣  Pergunta + botão de execução
# ==========================================================================

st.subheader("2. Pergunta sobre os dados")

col_question, col_button = st.columns([4, 1])
with col_question:
    question = st.text_input(
        "Escreva sua pergunta:",
        placeholder="ex.: Qual a compra mais cara realizada?",
    )
with col_button:
    run_btn = st.button(
        "🚀 Analisar",
        disabled=not (uploaded_file and question),
        use_container_width=True,
    )

# ==========================================================================
# 5️⃣  Resultados
# ==========================================================================

st.markdown("---")
st.subheader("📈 Resultado da Análise")

answer_placeholder = st.empty()

if not run_btn:
    if uploaded_file is None:
        st.info("Envie um arquivo CSV para começar.")
    elif not question:
        st.info("Digite sua pergunta acima para habilitar a análise.")

# ---------------------------------------------------------------------------
# Execução do workflow
# ---------------------------------------------------------------------------
if run_btn:
    uploaded_file.seek(0)
    with st.spinner("Processando… aguarde alguns instantes"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = Path(tmp.name)

        try:
            resposta = run_workflow(
                csv_path=str(tmp_path),
                question=question,
                openai_api_key=openai_key or None,
                verbose=False,
            )
        except Exception as e:
            answer_placeholder.error(f"❌ Erro ao executar o workflow: {e}")
        else:
            sanitized = sanitize_markdown(str(resposta))
            answer_placeholder.success(sanitized)

        # Limpa arquivo temporário
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass

# ==========================================================================
# 6️⃣  Rodapé
# ==========================================================================

st.markdown("---")
st.caption("Feito com LangGraph, OpenAI e Streamlit • ")
st.write("[Repositório no GitHub](https://github.com/artpedro/chat-csv)")
