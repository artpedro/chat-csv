# 📊 ChatCSV

Análise inteligente de arquivos CSV usando um sistema **multi‑agente** (LangGraph + OpenAI) com interface **Streamlit**.

> *Faça upload de um CSV ➜ digite uma pergunta em português ➜ receba insights em segundos.*

https://github.com/user-attachments/assets/f1582184-5934-4fa7-aa7a-1e647a5cbfe4

---

## ✨ Recursos Principais

|                                | Descrição                                                                                              |
| ------------------------------ | ------------------------------------------------------------------------------------------------------ |
| 🚀 **Análise Automática**      | Orquestração de agentes (LangGraph) para interpretar dados e responder perguntas em linguagem natural. |
| 📂 **Upload Drag‑and‑Drop**    | Aceita arquivos CSV de até 200 MB.                                                                     |
| 👀 **Pré‑visualização Segura** | Mostra as primeiras linhas do CSV.                                                                     |
| 🔑 **Chave OpenAI**            | Pode ser informada na UI e não será compartilhada pela rede                                            |

---

## 🧩 Arquitetura do Projeto

![EsquemaVisualAgentes - I2A2](https://github.com/user-attachments/assets/de79a72f-dbb0-48a3-8123-9b7654ba60a9)

```
assistente-csv-ia/
├─ app.py                # Código principal (interface + lógica)
├─ streamlit_agents.py   # Workflow multi‑agente (importado pelo app)
├─ requirements.txt      # Dependências Python
├─ .gitignore
└─ README.md             # Este arquivo

```

---

## ⚙️ Pré‑requisitos

* Python **3.9+**
* Conta e chave da **OpenAI API**

---

## 📦 Instalação

```bash
# 1. Clone o repositório
$ git clone https://github.com/artpedro/chat-csv.git
$ cd chat-csv

# 2. Crie e ative um ambiente virtual (opcional, mas recomendado)
$ python -m venv .venv
$ source .venv/bin/activate  # Linux/Mac
$ .venv\Scripts\activate     # Windows

# 3. Instale as dependências
$ pip install -r requirements.txt
```

---

## 🚀 Como Executar

```bash
# No diretório do projeto
$ streamlit run app.py
```

1. Abra o navegador na URL indicada (geralmente `http://localhost:8501`).
2. Informe sua `OPENAI_API_KEY` no expansor de Configurações Avançadas.
3. Faça **upload** de um arquivo CSV.
4. Digite sua **pergunta** em português.
5. Clique em **Analisar** e aguarde a resposta.

---

## 🔧 Variáveis de Ambiente

| Variável         | Descrição                                                                   |
| ---------------- | --------------------------------------------------------------------------- |
| `OPENAI_API_KEY` | Chave da API da OpenAI. Pode ser definida aqui ou diretamente na interface. |

Crie um arquivo `.streamlit/secrets.toml` se preferir:

```toml
OPENAI_API_KEY = "sk‑..."
```
---

## 🤝 Contribuindo

1. Fork o projeto
2. Crie sua branch (`git checkout -b feature/minha-feature`)
3. Commit suas alterações (`git commit -m 'feat: Minha nova feature'`)
4. Push para a branch (`git push origin feature/minha-feature`)
5. Abra um **Pull Request**
