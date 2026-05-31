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

st.set_page_config(
    page_title="FinDoc AI — Financial RAG Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.user-bubble {
    background: #1a1f2e; border: 1px solid #2d3748;
    border-radius: 12px 12px 2px 12px; padding: 12px 16px;
    margin: 8px 0; color: #e2e8f0;
}
.assistant-bubble {
    background: #0f172a; border: 1px solid #1e3a5f;
    border-left: 3px solid #3b82f6; border-radius: 2px 12px 12px 12px;
    padding: 14px 18px; margin: 8px 0; color: #e2e8f0; line-height: 1.7;
}
.source-card {
    background: #111827; border: 1px solid #1e293b;
    border-radius: 8px; padding: 10px 14px; margin: 6px 0;
    font-size: 0.85em; color: #94a3b8;
}
.source-card strong { color: #60a5fa; }
.badge-high   { background:#064e3b; color:#6ee7b7; border-radius:4px; padding:2px 8px; font-size:0.78em; }
.badge-medium { background:#451a03; color:#fcd34d; border-radius:4px; padding:2px 8px; font-size:0.78em; }
.badge-low    { background:#450a0a; color:#fca5a5; border-radius:4px; padding:2px 8px; font-size:0.78em; }
.metric-row { display: flex; gap: 8px; flex-wrap: wrap; }
.metric-card {
    background: #1e293b; border-radius: 8px; padding: 10px 14px;
    flex: 1; min-width: 100px; text-align: center;
}
.metric-val { font-size: 1.4em; font-weight: 700; color: #38bdf8; }
.metric-lbl { font-size: 0.75em; color: #64748b; margin-top: 2px; }

/* Settings toggle button */
.settings-toggle {
    display: flex;
    align-items: center;
    justify-content: space-between;
    width: 100%;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 8px 14px;
    color: #94a3b8;
    font-size: 0.9em;
    cursor: pointer;
    margin-bottom: 4px;
}
.settings-toggle:hover { background: #263548; border-color: #3b82f6; color: #e2e8f0; }

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stChatInputContainer { border-top: 1px solid #1e293b; }
</style>
""", unsafe_allow_html=True)


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
        "settings_open": False,   # ← NEW: retrieval settings collapsed by default
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


@st.cache_resource(show_spinner="Loading embedding models…")
def get_engine(use_reranker: bool) -> FinancialRAGEngine:
    return FinancialRAGEngine(use_reranker=use_reranker)


def get_or_create_engine() -> FinancialRAGEngine:
    if st.session_state.engine is None:
        st.session_state.engine = get_engine(st.session_state.use_reranker)
    return st.session_state.engine


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 FinDoc AI")
    st.markdown("*Financial Document Intelligence*")
    st.divider()

    # LLM Configuration
    st.markdown("### 🤖 LLM Configuration")
    provider_options = list(PROVIDER_DISPLAY_NAMES.keys())
    provider_labels  = [PROVIDER_DISPLAY_NAMES[p] for p in provider_options]
    provider_idx     = provider_options.index(st.session_state.provider)

    selected_label = st.selectbox("Provider", provider_labels, index=provider_idx, key="provider_select")
    new_provider   = provider_options[provider_labels.index(selected_label)]
    if new_provider != st.session_state.provider:
        st.session_state.provider       = new_provider
        st.session_state.model          = DEFAULT_MODELS[new_provider]
        st.session_state.llm_configured = False

    model_map            = model_display_names(st.session_state.provider)
    model_ids            = list(model_map.keys())
    model_labels         = list(model_map.values())
    cur_model_idx        = model_ids.index(st.session_state.model) if st.session_state.model in model_ids else 0
    selected_model_label = st.selectbox("Model", model_labels, index=cur_model_idx)
    st.session_state.model = model_ids[model_labels.index(selected_model_label)]

    secret_key           = f"{st.session_state.provider}_api_key"
    api_key_from_secrets = st.secrets.get(secret_key, "") if hasattr(st, "secrets") else ""

    if api_key_from_secrets:
        api_key = api_key_from_secrets
        st.success("✅ API key loaded from secrets")
    else:
        api_key = st.text_input(
            f"{PROVIDER_DISPLAY_NAMES[st.session_state.provider]} API Key",
            type="password",
            placeholder="Paste your API key here",
        )

    if st.button("🔗 Connect LLM", use_container_width=True):
        if not api_key:
            st.error("Please enter an API key.")
        else:
            try:
                engine = get_or_create_engine()
                engine.configure_llm(
                    provider=st.session_state.provider,
                    api_key=api_key,
                    model=st.session_state.model,
                )
                st.session_state.llm_configured = True
                st.success("LLM connected!")
            except Exception as e:
                st.error(f"Connection failed: {e}")

    st.divider()

    # Document Upload
    st.markdown("### 📄 Documents")
    uploaded_files = st.file_uploader(
        "Upload financial documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="PDF, DOCX, or TXT. Annual reports, 10-Ks, compliance docs, etc.",
    )

    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.docs_indexed]
        if new_files:
            if st.button(f"⚙️ Process {len(new_files)} new file(s)", use_container_width=True):
                engine       = get_or_create_engine()
                progress     = st.progress(0)
                total_chunks = 0
                for i, uf in enumerate(new_files):
                    with st.spinner(f"Indexing {uf.name}…"):
                        try:
                            n = engine.ingest_file(uf.read(), uf.name)
                            total_chunks += n
                            st.session_state.docs_indexed.append(uf.name)
                        except Exception as e:
                            st.error(f"Failed to process {uf.name}: {e}")
                    progress.progress((i + 1) / len(new_files))
                st.success(f"Indexed {total_chunks} chunks from {len(new_files)} file(s)!")
                progress.empty()
        else:
            st.info("All uploaded files are already indexed.")

    if st.session_state.docs_indexed:
        st.markdown("**Indexed files:**")
        for fn in st.session_state.docs_indexed:
            st.markdown(f"  • `{fn}`")

    st.divider()

    # Policy Mode
    st.markdown("### 🛡️ Compliance Policy")
    policy_descriptions = {
        "Free":      "No constraints — general assistant behaviour",
        "Assistive": "Guides tone; professional financial framing",
        "Strict":    "Only answers from document evidence; cites sources",
    }
    st.session_state.policy_mode = st.radio(
        "Mode",
        ["Free", "Assistive", "Strict"],
        index=["Free", "Assistive", "Strict"].index(st.session_state.policy_mode),
        help="\n".join(f"**{k}**: {v}" for k, v in policy_descriptions.items()),
    )
    st.caption(policy_descriptions[st.session_state.policy_mode])

    policy_doc = st.file_uploader(
        "Upload policy document (optional)", type=["pdf", "txt"], key="policy_upload"
    )
    if policy_doc:
        if policy_doc.name.endswith(".pdf"):
            from core.rag_engine import DocumentParser
            pages = DocumentParser().parse(policy_doc.read(), policy_doc.name)
            st.session_state.policy_text = "\n\n".join(t for t, _ in pages)
        else:
            st.session_state.policy_text = policy_doc.read().decode("utf-8", errors="replace")
        engine = get_or_create_engine()
        engine.set_policy(st.session_state.policy_mode, st.session_state.policy_text)
        st.success("Policy loaded!")
    else:
        engine = get_or_create_engine()
        engine.set_policy(st.session_state.policy_mode, st.session_state.policy_text)

    st.divider()

    # ── Retrieval Settings — toggle button ────────────────────────────────
    chevron = "▲" if st.session_state.settings_open else "▼"
    toggle_label = f"⚙️ Retrieval Settings  {chevron}"

    if st.button(toggle_label, use_container_width=True, key="settings_toggle_btn"):
        st.session_state.settings_open = not st.session_state.settings_open
        st.rerun()

    if st.session_state.settings_open:
        with st.container():
            st.session_state.top_k = st.slider(
                "Initial retrieval (top-K)",
                min_value=3,
                max_value=12,
                value=st.session_state.top_k,
                help="How many chunks to fetch from the vector index before reranking.",
            )
            st.session_state.rerank_top_n = st.slider(
                "After reranking (top-N)",
                min_value=1,
                max_value=6,
                value=st.session_state.rerank_top_n,
                help="How many chunks to keep after the cross-encoder reranker scores them.",
            )
            st.caption(
                f"Fetches **{st.session_state.top_k}** candidates → "
                f"reranks to best **{st.session_state.rerank_top_n}**."
            )

    st.divider()
    if st.button("🔄 Reset Session", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.cache_resource.clear()
        st.rerun()


# ── Main area ─────────────────────────────────────────────────────────────────
col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown("# 📊 Financial Document Intelligence Assistant")
    st.markdown(
        "*Upload financial documents, then ask questions — "
        "grounded answers with full source attribution.*"
    )

with col_status:
    engine_status = get_or_create_engine()
    llm_ok  = st.session_state.llm_configured
    docs_ok = bool(st.session_state.docs_indexed)
    st.markdown(f"**LLM:** {'🟢 Ready' if llm_ok else '🔴 Not connected'}")
    st.markdown(
        f"**Index:** {'🟢 ' + str(engine_status.stats['total_chunks']) + ' chunks' if docs_ok else '🔴 No docs'}"
    )
    st.markdown(f"**Reranker:** {'🟢 On' if engine_status._store.uses_reranker else '🟡 Off'}")
    st.markdown(f"**Mode:** `{st.session_state.policy_mode}`")

st.divider()


# ── Chat + Insights ───────────────────────────────────────────────────────────
chat_col, insight_col = st.columns([2, 1])

with chat_col:
    st.markdown("### 💬 Conversation")

    if not st.session_state.display_history:
        st.markdown("""
        <div style="text-align:center; padding:40px; color:#475569;">
            <div style="font-size:3em;">📑</div>
            <p>Upload documents in the sidebar, connect your LLM, then start asking questions.</p>
            <p style="font-size:0.85em; color:#334155;">
                Examples: <em>"What is the total revenue for FY2023?"</em> ·
                <em>"Summarise the key risk factors"</em> ·
                <em>"What does the policy say about data retention?"</em>
            </p>
        </div>
        """, unsafe_allow_html=True)

    for turn in st.session_state.display_history:
        if turn["role"] == "user":
            st.markdown(
                f'<div class="user-bubble">🧑 {turn["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="assistant-bubble">🤖 {turn["content"]}</div>',
                unsafe_allow_html=True,
            )
            if turn.get("results"):
                with st.expander(
                    f"📎 {len(turn['results'])} source(s) · "
                    f"⚡ {turn.get('latency_ms', 0):.0f}ms retrieval",
                    expanded=False,
                ):
                    for i, r in enumerate(turn["results"], 1):
                        badge_cls = f"badge-{r.confidence_label.lower()}"
                        score_val = r.rerank_score if r.rerank_score is not None else r.score
                        st.markdown(f"""<div class="source-card">
                            <strong>Source {i}: {r.chunk.source}</strong>
                            &nbsp; Page {r.chunk.page}
                            &nbsp; <span class="{badge_cls}">{r.confidence_label}</span>
                            &nbsp; score: {score_val:.3f}<br>
                            <em>{r.chunk.text[:280].strip()}…</em>
                        </div>""", unsafe_allow_html=True)

    question = st.chat_input(
        "Ask a question about your financial documents…",
        disabled=not (llm_ok and docs_ok),
    )

    if question:
        if not llm_ok:
            st.warning("Please connect an LLM first (sidebar → Connect LLM).")
        elif not docs_ok:
            st.warning("Please upload and process documents first.")
        else:
            st.session_state.display_history.append({"role": "user", "content": question})
            st.session_state.chat_history.append({"role": "user", "content": question})

            answer_placeholder = st.empty()
            full_answer        = ""
            retrieval_results  = []
            retrieval_latency  = 0.0

            try:
                engine = get_or_create_engine()
                engine.set_policy(st.session_state.policy_mode, st.session_state.policy_text)

                stream, retrieval_results, retrieval_latency = engine.query_stream(
                    question=question,
                    chat_history=st.session_state.chat_history[:-1],
                    top_k=st.session_state.top_k,
                    rerank_top_n=st.session_state.rerank_top_n,
                )

                for token in stream:
                    full_answer += token
                    answer_placeholder.markdown(
                        f'<div class="assistant-bubble">🤖 {full_answer}▌</div>',
                        unsafe_allow_html=True,
                    )
                answer_placeholder.empty()

            except Exception as e:
                full_answer = f"⚠️ Error: {e}"
                answer_placeholder.error(full_answer)

            st.session_state.display_history.append({
                "role":       "assistant",
                "content":    full_answer,
                "results":    retrieval_results,
                "latency_ms": retrieval_latency,
            })
            st.session_state.chat_history.append({"role": "assistant", "content": full_answer})
            st.rerun()


with insight_col:
    st.markdown("### 🔍 Retrieval Insights")
    engine = get_or_create_engine()
    stats  = engine.stats

    st.markdown("**System**")
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="metric-val">{stats['total_chunks']}</div>
            <div class="metric-lbl">Chunks</div>
        </div>
        <div class="metric-card">
            <div class="metric-val">{stats['chunk_size']}</div>
            <div class="metric-lbl">Chunk size</div>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin:8px 0; font-size:0.82em; color:#64748b;">
        <b>Embed:</b> {stats['embed_model']}<br>
        <b>Reranker:</b> {stats['reranker'].split('/')[-1] if '/' in stats['reranker'] else stats['reranker']}<br>
        <b>Overlap:</b> {stats['overlap']} tokens<br>
        <b>Provider:</b> {stats['llm_provider'].capitalize()}<br>
        <b>Model:</b> {stats['llm_model'].split('/')[-1] if '/' in stats['llm_model'] else stats['llm_model']}
    </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("**Last Query Sources**")

    last_assistant = next(
        (t for t in reversed(st.session_state.display_history) if t["role"] == "assistant"),
        None,
    )

    if last_assistant and last_assistant.get("results"):
        results = last_assistant["results"]
        latency = last_assistant.get("latency_ms", 0)
        st.caption(f"⚡ Retrieval: {latency:.0f}ms")
        for i, r in enumerate(results, 1):
            score_val = r.rerank_score if r.rerank_score is not None else r.score
            badge_cls = f"badge-{r.confidence_label.lower()}"
            st.markdown(f"""<div class="source-card">
                <strong>#{i} {r.chunk.source}</strong>
                &nbsp;<span class="{badge_cls}">{r.confidence_color} {r.confidence_label}</span><br>
                Page {r.chunk.page} · Score: {score_val:.3f}<br>
                <em style="font-size:0.9em;">{r.chunk.text[:120].strip()}…</em>
            </div>""", unsafe_allow_html=True)
    else:
        st.caption("No query yet.")

    st.divider()
    st.markdown("**Session**")
    n_turns = len([t for t in st.session_state.display_history if t["role"] == "user"])
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="metric-val">{n_turns}</div>
            <div class="metric-lbl">Questions</div>
        </div>
        <div class="metric-card">
            <div class="metric-val">{len(st.session_state.docs_indexed)}</div>
            <div class="metric-lbl">Docs</div>
        </div>
    </div>""", unsafe_allow_html=True)

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.display_history = []
        st.session_state.chat_history    = []
        st.rerun()
