"""
app.py  ←  Streamlit Cloud entry point
---------------------------------------
Financial Document Intelligence Assistant — Revamped
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from typing import List, Optional
import time

from core.rag_engine import FinancialRAGEngine
from core.models import (
    PROVIDER_MODELS,
    PROVIDER_DISPLAY_NAMES,
    DEFAULT_MODELS,
    model_display_names,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinDoc AI — Financial RAG Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.user-bubble {
    background: #1a1f2e;
    border: 1px solid #2d3748;
    border-radius: 12px 12px 2px 12px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #e2e8f0;
}

.assistant-bubble {
    background: #0f172a;
    border: 1px solid #1e3a5f;
    border-left: 3px solid #3b82f6;
    border-radius: 2px 12px 12px 12px;
    padding: 14px 18px;
    margin: 8px 0;
    color: #e2e8f0;
    line-height: 1.7;
}

.source-card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.85em;
    color: #94a3b8;
}

.badge-high   { background:#064e3b; color:#6ee7b7; border-radius:4px; padding:2px 8px; }
.badge-medium { background:#451a03; color:#fcd34d; border-radius:4px; padding:2px 8px; }
.badge-low    { background:#450a0a; color:#fca5a5; border-radius:4px; padding:2px 8px; }

.metric-card {
    background: #1e293b;
    border-radius: 8px;
    padding: 10px 14px;
    flex: 1;
    min-width: 100px;
    text-align: center;
}

.metric-val { font-size: 1.4em; font-weight: 700; color: #38bdf8; }
.metric-lbl { font-size: 0.75em; color: #64748b; }

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "engine": None,
        "chat_history": [],
        "display_history": [],
        "docs_indexed": [],
        "llm_configured": False,
        "provider": "cohere",
        "model": DEFAULT_MODELS["cohere"],
        "policy_mode": "Free",
        "policy_text": None,
        "top_k": 6,
        "rerank_top_n": 3,
        "use_reranker": True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding models…")
def get_engine(use_reranker: bool):
    return FinancialRAGEngine(use_reranker=use_reranker)

def get_or_create_engine():
    if st.session_state.engine is None:
        st.session_state.engine = get_engine(st.session_state.use_reranker)
    return st.session_state.engine

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:

    st.markdown("## 📊 FinDoc AI")
    st.divider()

    st.markdown("### 🤖 LLM Configuration")

    provider_options = list(PROVIDER_DISPLAY_NAMES.keys())
    provider_labels = [PROVIDER_DISPLAY_NAMES[p] for p in provider_options]

    provider_idx = provider_options.index(st.session_state.provider)

    selected_label = st.selectbox(
        "Provider",
        provider_labels,
        index=provider_idx,
        key="provider_select",
    )

    new_provider = provider_options[provider_labels.index(selected_label)]
    if new_provider != st.session_state.provider:
        st.session_state.provider = new_provider
        st.session_state.model = DEFAULT_MODELS[new_provider]
        st.session_state.llm_configured = False

    model_map = model_display_names(st.session_state.provider)
    model_ids = list(model_map.keys())
    model_labels = list(model_map.values())

    cur_model_idx = model_ids.index(st.session_state.model) if st.session_state.model in model_ids else 0

    selected_model_label = st.selectbox("Model", model_labels, index=cur_model_idx)
    st.session_state.model = model_ids[model_labels.index(selected_model_label)]

    api_key = st.text_input("API Key", type="password")

    if st.button("Connect LLM"):
        engine = get_or_create_engine()
        engine.configure_llm(
            provider=st.session_state.provider,
            api_key=api_key,
            model=st.session_state.model,
        )
        st.session_state.llm_configured = True
        st.success("LLM connected")

    st.divider()

    st.markdown("### 📄 Documents")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    st.divider()

    st.markdown("### 🛡️ Policy Mode")

    st.session_state.policy_mode = st.radio(
        "Mode",
        ["Free", "Assistive", "Strict"],
    )

    st.divider()

    # ✅ FIXED INDENTATION HERE
    st.markdown("### ⚙️ Retrieval Settings")

    with st.expander("Retrieval Settings", expanded=True):
        st.session_state.top_k = st.slider(
            "Top-K",
            3,
            12,
            st.session_state.top_k,
        )

        st.session_state.rerank_top_n = st.slider(
            "Top-N after rerank",
            1,
            6,
            st.session_state.rerank_top_n,
        )

    st.divider()

    if st.button("Reset Session"):
        st.session_state.clear()
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("📊 Financial Document Intelligence Assistant")

st.markdown("Upload documents and ask questions.")

question = st.chat_input("Ask something…")

if question:
    engine = get_or_create_engine()

    st.session_state.display_history.append({
        "role": "user",
        "content": question
    })

    stream, results, latency = engine.query_stream(
        question=question,
        chat_history=st.session_state.chat_history,
        top_k=st.session_state.top_k,
        rerank_top_n=st.session_state.rerank_top_n,
    )

    answer = ""
    placeholder = st.empty()

    for token in stream:
        answer += token
        placeholder.markdown(answer)

    st.session_state.display_history.append({
        "role": "assistant",
        "content": answer,
        "results": results,
        "latency_ms": latency,
    })

    st.session_state.chat_history.append({"role": "user", "content": question})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
