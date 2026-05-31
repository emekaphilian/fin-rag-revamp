"""
core/rag_engine.py
==================
Financial Document Intelligence Assistant — RAG Engine
Fixed for Cohere v2 (cohere >= 5.0): system prompt goes in messages list,
not as a top-level `system=` kwarg. Also fixes v2 streaming event types.

Interface contract (matches app.py exactly):
    engine = FinancialRAGEngine(use_reranker=True)
    engine.configure_llm(provider, api_key, model)
    engine.set_policy(mode, policy_text)
    n_chunks = engine.ingest_file(file_bytes, filename)
    stream, results, latency_ms = engine.query_stream(question, chat_history, top_k, rerank_top_n)
    stats = engine.stats          # dict
    engine._store.uses_reranker   # bool
"""

from __future__ import annotations

import io
import re
import time
from dataclasses import dataclass, field
from typing import Generator, List, Optional, Tuple

import numpy as np

# ── optional heavy deps (graceful degradation) ───────────────────────────────
try:
    import faiss
    _FAISS_OK = True
except ImportError:
    _FAISS_OK = False

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    _ST_OK = True
except ImportError:
    _ST_OK = False

try:
    from pypdf import PdfReader
    _PDF_OK = True
except ImportError:
    _PDF_OK = False

try:
    from docx import Document as DocxDocument
    _DOCX_OK = True
except ImportError:
    _DOCX_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    text: str
    source: str
    page: int
    chunk_idx: int


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    rerank_score: Optional[float] = None

    @property
    def confidence_label(self) -> str:
        s = self.rerank_score if self.rerank_score is not None else self.score
        if s >= 0.75:
            return "High"
        if s >= 0.45:
            return "Medium"
        return "Low"

    @property
    def confidence_color(self) -> str:
        return {"High": "🟢", "Medium": "🟡", "Low": "🔴"}[self.confidence_label]


# ─────────────────────────────────────────────────────────────────────────────
# Document parser
# ─────────────────────────────────────────────────────────────────────────────

class DocumentParser:
    """Returns list of (text, page_number) tuples."""

    def parse(self, file_bytes: bytes, filename: str) -> List[Tuple[str, int]]:
        ext = filename.rsplit(".", 1)[-1].lower()
        if ext == "pdf":
            return self._parse_pdf(file_bytes)
        if ext == "docx":
            return self._parse_docx(file_bytes)
        # txt / fallback
        return [(file_bytes.decode("utf-8", errors="replace"), 1)]

    def _parse_pdf(self, data: bytes) -> List[Tuple[str, int]]:
        if not _PDF_OK:
            raise RuntimeError("pypdf not installed. Run: pip install pypdf")
        reader = PdfReader(io.BytesIO(data))
        pages = []
        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((text, i))
        return pages

    def _parse_docx(self, data: bytes) -> List[Tuple[str, int]]:
        if not _DOCX_OK:
            raise RuntimeError("python-docx not installed. Run: pip install python-docx")
        doc = DocxDocument(io.BytesIO(data))
        full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return [(full_text, 1)]


# ─────────────────────────────────────────────────────────────────────────────
# Financial-aware chunker
# ─────────────────────────────────────────────────────────────────────────────

class FinancialChunker:
    """
    Paragraph-aware sliding-window chunker tuned for financial documents.
    Tries to keep section headers with their content.
    Default: 900 tokens / 150 overlap (approx. by word count × 0.75).
    """

    SECTION_RE = re.compile(
        r"^\s*(§\d|section\s+\d|article\s+\d|\d+\.\d*\s+[A-Z])",
        re.IGNORECASE | re.MULTILINE,
    )

    def __init__(self, chunk_size: int = 900, overlap: int = 150):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _word_count(self, text: str) -> int:
        return len(text.split())

    def chunk(self, text: str, source: str, page: int) -> List[Chunk]:
        # Split on paragraph boundaries first
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        chunks: List[Chunk] = []
        buffer = ""
        chunk_idx = 0

        for para in paragraphs:
            candidate = (buffer + "\n\n" + para).strip() if buffer else para
            if self._word_count(candidate) <= self.chunk_size:
                buffer = candidate
            else:
                if buffer:
                    chunks.append(Chunk(
                        text=buffer,
                        source=source,
                        page=page,
                        chunk_idx=chunk_idx,
                    ))
                    chunk_idx += 1
                    # Carry overlap words into next buffer
                    words = buffer.split()
                    overlap_text = " ".join(words[-self.overlap:]) if len(words) > self.overlap else buffer
                    buffer = (overlap_text + "\n\n" + para).strip()
                else:
                    # Single paragraph too long — hard split by words
                    words = para.split()
                    for start in range(0, len(words), self.chunk_size - self.overlap):
                        segment = " ".join(words[start: start + self.chunk_size])
                        chunks.append(Chunk(text=segment, source=source, page=page, chunk_idx=chunk_idx))
                        chunk_idx += 1
                    buffer = ""

        if buffer.strip():
            chunks.append(Chunk(text=buffer, source=source, page=page, chunk_idx=chunk_idx))

        return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Vector store (FAISS)
# ─────────────────────────────────────────────────────────────────────────────

class VectorStore:
    EMBED_MODEL = "all-MiniLM-L6-v2"
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, use_reranker: bool = True):
        if not _ST_OK:
            raise RuntimeError("sentence-transformers not installed.")
        if not _FAISS_OK:
            raise RuntimeError("faiss-cpu not installed.")

        self.uses_reranker = use_reranker
        self._embedder = SentenceTransformer(self.EMBED_MODEL)
        self._dim = self._embedder.get_sentence_embedding_dimension()
        self._index = faiss.IndexFlatIP(self._dim)
        self._chunks: List[Chunk] = []

        self._reranker: Optional[CrossEncoder] = None
        if use_reranker:
            try:
                self._reranker = CrossEncoder(self.RERANKER_MODEL)
            except Exception:
                self.uses_reranker = False

    def add_chunks(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return
        texts = [c.text for c in chunks]
        embeddings = self._embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        self._index.add(np.array(embeddings, dtype="float32"))
        self._chunks.extend(chunks)

    def search(self, query: str, top_k: int = 6, rerank_top_n: int = 3) -> List[RetrievalResult]:
        if not self._chunks:
            return []

        q_emb = self._embedder.encode([query], normalize_embeddings=True, show_progress_bar=False)
        k = min(top_k, len(self._chunks))
        scores, indices = self._index.search(np.array(q_emb, dtype="float32"), k)

        results = [
            RetrievalResult(chunk=self._chunks[idx], score=float(scores[0][i]))
            for i, idx in enumerate(indices[0])
            if idx >= 0
        ]

        if self._reranker and results:
            pairs = [(query, r.chunk.text) for r in results]
            rr_scores = self._reranker.predict(pairs)
            for r, s in zip(results, rr_scores):
                r.rerank_score = float(s)
            results.sort(key=lambda r: r.rerank_score, reverse=True)  # type: ignore
            results = results[:rerank_top_n]
        else:
            results = results[:rerank_top_n]

        return results

    @property
    def total_chunks(self) -> int:
        return len(self._chunks)


# ─────────────────────────────────────────────────────────────────────────────
# Policy engine
# ─────────────────────────────────────────────────────────────────────────────

POLICY_SYSTEM_PROMPTS = {
    "Free": (
        "You are a helpful financial document assistant. "
        "Answer questions using the provided document context. "
        "You may draw on general knowledge when the documents don't fully address the question."
    ),
    "Assistive": (
        "You are a professional financial analyst assistant. "
        "Answer questions based primarily on the provided document context. "
        "Use professional financial language. "
        "If your confidence is low, explicitly signal uncertainty to the user."
    ),
    "Strict": (
        "You are a strict compliance-grade financial document assistant. "
        "You MUST answer ONLY from the provided document context. "
        "Do NOT extrapolate, assume, or use general knowledge. "
        "Every factual claim must be directly supported by the retrieved evidence. "
        "If the documents do not contain sufficient information, say: "
        "'I cannot answer this from the available documents.' "
        "Always cite which document and section your answer comes from."
    ),
}


class PolicyEngine:
    def __init__(self):
        self._mode = "Free"
        self._custom_text: Optional[str] = None

    def set(self, mode: str, custom_text: Optional[str] = None) -> None:
        self._mode = mode
        self._custom_text = custom_text

    def system_prompt(self) -> str:
        base = POLICY_SYSTEM_PROMPTS.get(self._mode, POLICY_SYSTEM_PROMPTS["Free"])
        if self._custom_text:
            base += f"\n\nAdditional policy constraints:\n{self._custom_text[:2000]}"
        return base


# ─────────────────────────────────────────────────────────────────────────────
# LLM router — Cohere v2 FIX HERE
# ─────────────────────────────────────────────────────────────────────────────

class LLMRouter:
    def __init__(self):
        self.provider: Optional[str] = None
        self.model: Optional[str] = None
        self._client = None

    def configure(self, provider: str, api_key: str, model: str) -> None:
        self.provider = provider
        self.model = model

        if provider == "cohere":
            import cohere
            # cohere >= 5.0 uses ClientV2
            try:
                self._client = cohere.ClientV2(api_key)
            except AttributeError:
                # Fallback for older cohere versions
                self._client = cohere.Client(api_key)

        elif provider == "openai":
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key)

        elif provider == "huggingface":
            from huggingface_hub import InferenceClient
            self._client = InferenceClient(token=api_key)

        else:
            raise ValueError(f"Unknown provider: {provider}")

    def stream(
        self,
        system_prompt: str,
        chat_history: List[dict],
        user_message: str,
        context: str,
    ) -> Generator[str, None, None]:
        if self._client is None:
            raise RuntimeError("LLM not configured. Call configure() first.")

        # Build the user turn with context injected
        user_with_context = (
            f"Document context:\n{context}\n\n"
            f"Question: {user_message}"
        )

        if self.provider == "cohere":
            yield from self._stream_cohere(system_prompt, chat_history, user_with_context)
        elif self.provider == "openai":
            yield from self._stream_openai(system_prompt, chat_history, user_with_context)
        elif self.provider == "huggingface":
            yield from self._stream_huggingface(system_prompt, chat_history, user_with_context)

    # ── Cohere v2 — THE FIX ─────────────────────────────────────────────────
    def _stream_cohere(
        self,
        system_prompt: str,
        chat_history: List[dict],
        user_message: str,
    ) -> Generator[str, None, None]:
        """
        Cohere v2 fix:
          - system prompt → messages[0] with role="system"
          - user message  → messages[-1] with role="user"
          - NO top-level system= kwarg (removed in v2)
          - NO top-level message= kwarg (removed in v2)
          - Streaming event type: "content-delta" (not "text-generation")
          - Token access: event.delta.message.content.text
        """
        messages = [{"role": "system", "content": system_prompt}]

        # Add prior turns (filter to only user/assistant roles)
        for turn in chat_history:
            if turn.get("role") in ("user", "assistant"):
                messages.append({"role": turn["role"], "content": turn["content"]})

        # Current user turn
        messages.append({"role": "user", "content": user_message})

        response = self._client.chat_stream(
            model=self.model,
            messages=messages,
            temperature=0.3,
        )

        for event in response:
            # v2 event type is "content-delta"
            if hasattr(event, "type") and event.type == "content-delta":
                try:
                    token = event.delta.message.content.text
                    if token:
                        yield token
                except (AttributeError, TypeError):
                    pass
            # Fallback: some v2 builds use different attribute paths
            elif hasattr(event, "delta") and hasattr(event.delta, "text"):
                token = event.delta.text
                if token:
                    yield token

    # ── OpenAI ──────────────────────────────────────────────────────────────
    def _stream_openai(
        self,
        system_prompt: str,
        chat_history: List[dict],
        user_message: str,
    ) -> Generator[str, None, None]:
        messages = [{"role": "system", "content": system_prompt}]
        for turn in chat_history:
            if turn.get("role") in ("user", "assistant"):
                messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": user_message})

        stream = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    # ── HuggingFace ─────────────────────────────────────────────────────────
    def _stream_huggingface(
        self,
        system_prompt: str,
        chat_history: List[dict],
        user_message: str,
    ) -> Generator[str, None, None]:
        messages = [{"role": "system", "content": system_prompt}]
        for turn in chat_history:
            if turn.get("role") in ("user", "assistant"):
                messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": user_message})

        try:
            stream = self._client.chat_completion(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=1024,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
        except Exception:
            # Pseudo-streaming fallback for models that don't stream
            response = self._client.chat_completion(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=1024,
                stream=False,
            )
            text = response.choices[0].message.content or ""
            # Yield in small chunks to simulate streaming in Streamlit
            words = text.split()
            for i in range(0, len(words), 5):
                yield " ".join(words[i:i+5]) + " "


# ─────────────────────────────────────────────────────────────────────────────
# Main engine — public API consumed by app.py
# ─────────────────────────────────────────────────────────────────────────────

class FinancialRAGEngine:
    """
    Public interface:
        engine.configure_llm(provider, api_key, model)
        engine.set_policy(mode, policy_text)
        n = engine.ingest_file(file_bytes, filename)
        stream, results, latency_ms = engine.query_stream(...)
        engine.stats  → dict
        engine._store.uses_reranker → bool
    """

    def __init__(self, use_reranker: bool = True):
        self._parser  = DocumentParser()
        self._chunker = FinancialChunker(chunk_size=900, overlap=150)
        self._store   = VectorStore(use_reranker=use_reranker)
        self._policy  = PolicyEngine()
        self._llm     = LLMRouter()

    # ── Configuration ────────────────────────────────────────────────────────

    def configure_llm(self, provider: str, api_key: str, model: str) -> None:
        self._llm.configure(provider, api_key, model)

    def set_policy(self, mode: str, policy_text: Optional[str] = None) -> None:
        self._policy.set(mode, policy_text)

    # ── Ingestion ────────────────────────────────────────────────────────────

    def ingest_file(self, file_bytes: bytes, filename: str) -> int:
        """Parse, chunk, embed and index a file. Returns number of chunks added."""
        pages = self._parser.parse(file_bytes, filename)
        all_chunks: List[Chunk] = []
        for text, page_num in pages:
            chunks = self._chunker.chunk(text, source=filename, page=page_num)
            all_chunks.extend(chunks)
        self._store.add_chunks(all_chunks)
        return len(all_chunks)

    # ── Query ────────────────────────────────────────────────────────────────

    def query_stream(
        self,
        question: str,
        chat_history: List[dict],
        top_k: int = 6,
        rerank_top_n: int = 3,
    ) -> Tuple[Generator[str, None, None], List[RetrievalResult], float]:
        """
        Returns (token_stream, retrieval_results, retrieval_latency_ms).
        The token stream is a generator — iterate it to get streamed tokens.
        """
        # Retrieval
        t0 = time.perf_counter()
        results = self._store.search(question, top_k=top_k, rerank_top_n=rerank_top_n)
        latency_ms = (time.perf_counter() - t0) * 1000

        # Build context string
        context_parts = []
        for i, r in enumerate(results, 1):
            context_parts.append(
                f"[Source {i}: {r.chunk.source}, Page {r.chunk.page}]\n{r.chunk.text}"
            )
        context = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant documents found."

        # Stream
        system_prompt = self._policy.system_prompt()
        stream = self._llm.stream(
            system_prompt=system_prompt,
            chat_history=chat_history,
            user_message=question,
            context=context,
        )

        return stream, results, latency_ms

    # ── Stats ────────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        return {
            "total_chunks":  self._store.total_chunks,
            "chunk_size":    f"{self._chunker.chunk_size}/{self._chunker.overlap}",
            "overlap":       self._chunker.overlap,
            "embed_model":   VectorStore.EMBED_MODEL,
            "reranker":      VectorStore.RERANKER_MODEL if self._store.uses_reranker else "disabled",
            "llm_provider":  self._llm.provider or "not configured",
            "llm_model":     self._llm.model or "not configured",
        }

