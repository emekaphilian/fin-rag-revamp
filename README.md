# 📊 Financial Document Intelligence Assistant — Revamped

A **production-grade RAG system** for querying financial and compliance documents.  
Built for **Streamlit Cloud deployment** with improved chunking, cross-encoder reranking, multi-LLM support, and persistent chat history.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## ✨ What's New (Revamp vs. Original)

| Feature | Original | Revamped |
|---|---|---|
| Chunking | Fixed 800/100 | Paragraph-aware + sliding window financial-tuned 900/150 |
| Reranking | ❌ | ✅ Cross-encoder (`ms-marco-MiniLM-L-6-v2`) |
| Chat history | ❌ | ✅ Multi-turn with context window management |
| Streamlit Cloud | ⚠️ torch breaks deployment | ✅ ONNX backend, no torch |
| API keys | `.env` only | ✅ `st.secrets` + UI fallback |
| System deps | missing | ✅ `packages.txt` included |
| HuggingFace models | outdated | ✅ Llama 3.3, Qwen 2.5, Phi-3.5 |
| Source cards | basic | ✅ Inline confidence badges + scores |
| Insights panel | separate page | ✅ Live right-side panel |

---

## 🏗️ Architecture

```
User uploads PDF/DOCX/TXT
        │
        ▼
DocumentParser  ──► page-level text extraction
        │
        ▼
FinancialChunker  ──► paragraph-aware split → 900-token chunks / 150 overlap
        │
        ▼
SentenceTransformer (all-MiniLM-L6-v2)  ──► 384-dim embeddings (ONNX, CPU)
        │
        ▼
FAISS IndexFlatIP  ──► cosine similarity search, top-K candidates
        │
        ▼
CrossEncoder (ms-marco-MiniLM-L-6-v2)  ──► reranks to top-N
        │
        ▼
LLMRouter  ──► OpenAI / Cohere / HuggingFace  ──► streaming response
        │
        ▼
Streamlit chat UI  ──► source cards + confidence + insights panel
```

---

## 🚀 Deploy to Streamlit Cloud

### 1. Push to GitHub

```bash
git clone https://github.com/emekaphilian/financial-document-intelligence-assistant-rag-system.git
# Replace the repo contents with this revamped version
git add .
git commit -m "revamp: improved chunking, reranking, chat history, cloud deployment"
git push
```

### 2. Connect on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **New app** → select your repo
3. Set **Main file path**: `app.py`
4. Click **Advanced settings** → **Secrets** and add:

```toml
openai_api_key = "sk-..."
cohere_api_key = "..."
huggingface_api_key = "hf_..."
```

5. Click **Deploy**

---

## 💻 Run Locally

```bash
git clone https://github.com/emekaphilian/financial-document-intelligence-assistant-rag-system.git
cd financial-document-intelligence-assistant-rag-system

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Add secrets
cp .streamlit/secrets.toml.template .streamlit/secrets.toml
# Edit secrets.toml with your API keys

streamlit run app.py
```

---

## 📁 Project Structure

```
├── app.py                          # Streamlit entry point (Cloud-ready)
├── core/
│   ├── __init__.py
│   ├── rag_engine.py               # RAG pipeline: parse, chunk, embed, rerank, generate
│   └── models.py                   # LLM model catalogue (OpenAI / Cohere / HF)
├── .streamlit/
│   ├── config.toml                 # Dark theme + server settings
│   └── secrets.toml.template       # API key template (do not commit real keys)
├── requirements.txt                # Streamlit Cloud-compatible (no torch)
├── packages.txt                    # System deps (libgomp1 for FAISS)
├── runtime.txt                     # Python 3.11
└── .gitignore
```

---

## 🤖 Supported LLM Providers

| Provider | Models |
|---|---|
| **OpenAI** | GPT-4o, GPT-4o Mini, GPT-4 Turbo, GPT-3.5 Turbo |
| **Cohere** | Command R+, Command R, Command |
| **HuggingFace** | Llama 3.3 70B, Mistral 7B, Qwen 2.5 72B, Phi-3.5 Mini |

---

## 🛡️ Policy Modes

| Mode | Behaviour |
|---|---|
| **Free** | General assistant — answers from context but can extrapolate |
| **Assistive** | Professional financial framing; signals low-confidence answers |
| **Strict** | Only answers from document evidence; refuses to extrapolate; cites every claim |

---

## 🔧 Known Limitations

- No OCR for scanned PDFs (use text-layer PDFs)
- FAISS index is in-memory per session; large document sets may need persistent storage
- Confidence scores are heuristic (similarity-based), not calibrated probabilities

---

## 🗺️ Roadmap

- [ ] Persistent vector store (SQLite + FAISS serialisation)
- [ ] Token usage & cost tracker
- [ ] Hybrid BM25 + semantic search
- [ ] RBAC / multi-user support
- [ ] Export chat as PDF report
- [ ] Evaluation framework (RAGAS)

---

## 👤 Author

**Emeka Philian Ogbonna** — Applied LLM Engineer  
GitHub: [github.com/emekaphilian](https://github.com/emekaphilian)

## 📄 License

MIT
