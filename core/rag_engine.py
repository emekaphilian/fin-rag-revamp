"""
core/rag_engine.py
------------------
Financial RAG Engine — revamped with:
  - Semantic + recursive chunking strategy
  - FAISS vector search (no torch required)
  - Cross-encoder reranking via sentence-transformers
  - Multi-LLM abstraction (OpenAI / Cohere / HuggingFace)
  - Streaming support
  - Policy-aware mode (Free / Assistive / Strict)
"""

from __future__ import annotations

import io
import time
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import numpy as np

# ── Document parsing ──────────────────────────────────────────────────────────
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    import docx as python_docx
except ImportError:
    python_docx = None

# ── Embeddings + FAISS ────────────────────────────────────────────────────────
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss

# ── LLM clients ───────────────────────────────────────────────────────────────
try:
    from openai import OpenAI as OpenAIClient
except ImportError:
    OpenAIClient = None

try:
    import cohere
except ImportError:
    cohere = None

try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    text: str
    source: str          # filename
    page: int = 0
    chunk_id: str = ""

    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = hashlib.md5(
                f"{self.source}:{self.page}:{self.text[:50]}".encode()
            ).hexdigest()[:8]


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float          # cosine similarity (0-1)
    rerank_score: Optional[float] = None

    @property
    def confidence_label(self) -> str:
        s = self.rerank_score if self.rerank_score is not None else self.score
        if s >= 0.70:
            return "High"
        elif s >= 0.45:
            return "Medium"
        return "Low"

    @property
    def confidence_color(self) -> str:
        return {"High": "🟢", "Medium": "🟡", "Low": "🔴"}[self.confidence_label]


@dataclass
class RAGResponse:
    answer: str
    results: List[RetrievalResult]
    query_latency_ms: float
    total_chunks_searched: int
    llm_provider: str
    policy_mode: str
    chat_history_turns: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────────────────

class FinancialChunker:
    """
    Two-pass chunking strategy tuned for financial documents:
      1. Paragraph-aware split (respects section boundaries)
      2. Fixed-size fallback with sliding window overlap
    """

    SECTION_MARKERS = [
        "\n\n", "\n---\n", "\nSection ", "\nARTICLE ", "\n## ", "\n### ",
        "\nITEM ", "\nNote ", "\nSchedule ", "\nExhibit "
    ]

    def __init__(self, chunk_size: int = 900, overlap: int = 150):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str, source: str, page: int = 0) -> List[Chunk]:
        # First try paragraph-aware split
        paragraphs = self._paragraph_split(text)
        chunks = self._merge_paragraphs(paragraphs, source, page)
        return chunks

    def _paragraph_split(self, text: str) -> List[str]:
        """Split on financial section markers first, then double newlines."""
        import re
        # Normalise whitespace
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r" {3,}", "  ", text)

        # Split on known markers
        for marker in self.SECTION_MARKERS:
            if marker in text:
                parts = text.split(marker)
                return [p.strip() for p in parts if p.strip()]

        # Fallback: double-newline paragraphs
        parts = text.split("\n\n")
        return [p.strip() for p in parts if p.strip()]

    def _merge_paragraphs(
        self, paragraphs: List[str], source: str, page: int
    ) -> List[Chunk]:
        chunks: List[Chunk] = []
        buffer = ""
        for para in paragraphs:
            if len(buffer) + len(para) + 1 <= self.chunk_size:
                buffer = (buffer + " " + para).strip()
            else:
                if buffer:
                    chunks.append(Chunk(text=buffer, source=source, page=page))
                # If single paragraph exceeds chunk_size, do sliding window
                if len(para) > self.chunk_size:
                    chunks.extend(self._sliding_window(para, source, page))
                    buffer = ""
                else:
                    buffer = para
        if buffer:
            chunks.append(Chunk(text=buffer, source=source, page=page))
        return chunks

    def _sliding_window(self, text: str, source: str, page: int) -> List[Chunk]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(Chunk(text=chunk_text, source=source, page=page))
            start += self.chunk_size - self.overlap
        return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Document parser
# ─────────────────────────────────────────────────────────────────────────────

class DocumentParser:
    def parse(self, file_bytes: bytes, filename: str) -> List[Tuple[str, int]]:
        """Returns list of (page_text, page_number) tuples."""
        ext = Path(filename).suffix.lower()
        if ext == ".pdf":
            return self._parse_pdf(file_bytes)
        elif ext == ".docx":
            return self._parse_docx(file_bytes)
        elif ext == ".txt":
            text = file_bytes.decode("utf-8", errors="replace")
            return [(text, 0)]
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _parse_pdf(self, data: bytes) -> List[Tuple[str, int]]:
        if PdfReader is None:
            raise ImportError("pypdf not installed")
        reader = PdfReader(io.BytesIO(data))
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((text, i + 1))
        return pages

    def _parse_docx(self, data: bytes) -> List[Tuple[str, int]]:
        if python_docx is None:
            raise ImportError("python-docx not installed")
        doc = python_docx.Document(io.BytesIO(data))
        full_text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return [(full_text, 0)]


# ─────────────────────────────────────────────────────────────────────────────
# Vector store (FAISS)
# ─────────────────────────────────────────────────────────────────────────────

class FAISSVectorStore:
    EMBED_MODEL = "all-MiniLM-L6-v2"
    RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, use_reranker: bool = True):
        self._embedder = SentenceTransformer(self.EMBED_MODEL)
        self._reranker: Optional[CrossEncoder] = None
        self._use_reranker = use_reranker
        if use_reranker:
            try:
                self._reranker = CrossEncoder(self.RERANK_MODEL)
            except Exception:
                self._reranker = None

        self._index: Optional[faiss.Index] = None
        self._chunks: List[Chunk] = []
        self.embed_dim: int = 384  # MiniLM output

    # ── Indexing ──────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return
        texts = [c.text for c in chunks]
        vecs = self._embed(texts)

        if self._index is None:
            self._index = faiss.IndexFlatIP(self.embed_dim)  # Inner product = cosine on normalised

        self._index.add(vecs)
        self._chunks.extend(chunks)

    def clear(self) -> None:
        self._index = None
        self._chunks = []

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 6,
        rerank_top_n: int = 3,
    ) -> List[RetrievalResult]:
        if self._index is None or not self._chunks:
            return []

        qvec = self._embed([query])
        k = min(top_k, len(self._chunks))
        scores, indices = self._index.search(qvec, k)

        results: List[RetrievalResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append(RetrievalResult(
                chunk=self._chunks[idx],
                score=float(score),
            ))

        # ── Cross-encoder reranking ───────────────────────────────────────────
        if self._reranker and results:
            pairs = [[query, r.chunk.text] for r in results]
            rerank_scores = self._reranker.predict(pairs)
            for r, rs in zip(results, rerank_scores):
                r.rerank_score = float(rs)
            results.sort(key=lambda x: x.rerank_score or 0, reverse=True)
            results = results[:rerank_top_n]
        else:
            results = results[:rerank_top_n]

        return results

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _embed(self, texts: List[str]) -> np.ndarray:
        vecs = self._embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vecs.astype(np.float32)

    @property
    def total_chunks(self) -> int:
        return len(self._chunks)

    @property
    def is_ready(self) -> bool:
        return self._index is not None and len(self._chunks) > 0

    @property
    def uses_reranker(self) -> bool:
        return self._reranker is not None


# ─────────────────────────────────────────────────────────────────────────────
# LLM Abstraction
# ─────────────────────────────────────────────────────────────────────────────

POLICY_SYSTEM_PROMPTS = {
    "Free": (
        "You are a knowledgeable financial analyst assistant. "
        "Answer based on the provided context. Be clear and precise. "
        "If the answer is not in the context, say so honestly."
    ),
    "Assistive": (
        "You are a compliance-aware financial analyst assistant. "
        "Ground every answer strictly in the retrieved document excerpts. "
        "Frame responses professionally, suitable for financial reporting. "
        "Always note when confidence is limited. If the answer is not in the context, say so."
    ),
    "Strict": (
        "You are a strict compliance enforcement assistant. "
        "You MUST only answer from the exact retrieved document excerpts provided. "
        "Do NOT infer, extrapolate, or use general knowledge. "
        "If the context does not contain the answer, respond: "
        "'The provided documents do not contain sufficient information to answer this question.' "
        "Cite the source document for every claim."
    ),
}


def _build_rag_prompt(
    query: str,
    results: List[RetrievalResult],
    chat_history: List[dict],
    policy_mode: str,
    policy_text: Optional[str],
) -> List[dict]:
    """Build the messages list for the LLM call."""

    context_blocks = []
    for i, r in enumerate(results, 1):
        context_blocks.append(
            f"[Source {i}: {r.chunk.source}, p.{r.chunk.page}, "
            f"score={r.score:.2f}]\n{r.chunk.text}"
        )
    context = "\n\n---\n\n".join(context_blocks)

    policy_note = ""
    if policy_text:
        policy_note = f"\n\n## Policy Document (for compliance reference)\n{policy_text[:1500]}"

    system_msg = POLICY_SYSTEM_PROMPTS.get(policy_mode, POLICY_SYSTEM_PROMPTS["Free"])
    system_msg += policy_note

    messages = [{"role": "system", "content": system_msg}]
    # Add prior chat history (keep last 6 turns to manage tokens)
    for turn in chat_history[-6:]:
        messages.append(turn)

    user_content = (
        f"## Retrieved Financial Document Excerpts\n\n{context}"
        f"\n\n## Question\n{query}"
    )
    messages.append({"role": "user", "content": user_content})
    return messages


class LLMRouter:
    """
    Routes to the correct LLM provider and returns streaming or full response.
    Supported: openai | cohere | huggingface
    """

    def __init__(
        self,
        provider: str,
        api_key: str,
        model: str,
    ):
        self.provider = provider.lower()
        self.api_key = api_key
        self.model = model
        self._client = self._init_client()

    def _init_client(self):
        if self.provider == "openai":
            if OpenAIClient is None:
                raise ImportError("openai package not installed")
            return OpenAIClient(api_key=self.api_key)
        elif self.provider == "cohere":
            if cohere is None:
                raise ImportError("cohere package not installed")
            return cohere.ClientV2(api_key=self.api_key)
        elif self.provider == "huggingface":
            if InferenceClient is None:
                raise ImportError("huggingface_hub not installed")
            return InferenceClient(api_key=self.api_key)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def generate_stream(self, messages: List[dict]) -> Generator[str, None, None]:
        if self.provider == "openai":
            yield from self._openai_stream(messages)
        elif self.provider == "cohere":
            yield from self._cohere_stream(messages)
        elif self.provider == "huggingface":
            yield from self._hf_stream(messages)

    def _openai_stream(self, messages: List[dict]) -> Generator[str, None, None]:
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            temperature=0.1,
            max_tokens=1024,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def _cohere_stream(self, messages: List[dict]) -> Generator[str, None, None]:
        # Convert to Cohere message format
        cohere_messages = []
        system_content = ""
        for m in messages:
            if m["role"] == "system":
                system_content = m["content"]
            else:
                cohere_messages.append({"role": m["role"], "content": m["content"]})

        stream = self._client.chat_stream(
            model=self.model,
            messages=cohere_messages,
            system=system_content,
            temperature=0.1,
            max_tokens=1024,
        )
        for event in stream:
            if event and hasattr(event, "delta") and hasattr(event.delta, "message"):
                content = event.delta.message.content
                if content:
                    for part in content:
                        if hasattr(part, "text"):
                            yield part.text

    def _hf_stream(self, messages: List[dict]) -> Generator[str, None, None]:
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            temperature=0.1,
            max_tokens=1024,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


# ─────────────────────────────────────────────────────────────────────────────
# Main RAG Engine
# ─────────────────────────────────────────────────────────────────────────────

class FinancialRAGEngine:
    def __init__(self, use_reranker: bool = True):
        self._parser = DocumentParser()
        self._chunker = FinancialChunker(chunk_size=900, overlap=150)
        self._store = FAISSVectorStore(use_reranker=use_reranker)
        self._llm: Optional[LLMRouter] = None
        self._policy_text: Optional[str] = None
        self._policy_mode: str = "Free"

    # ── Setup ─────────────────────────────────────────────────────────────────

    def configure_llm(self, provider: str, api_key: str, model: str) -> None:
        self._llm = LLMRouter(provider=provider, api_key=api_key, model=model)

    def set_policy(self, mode: str, policy_text: Optional[str] = None) -> None:
        self._policy_mode = mode
        self._policy_text = policy_text

    def clear_index(self) -> None:
        self._store.clear()

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest_file(self, file_bytes: bytes, filename: str) -> int:
        """Parse + chunk + embed a file. Returns number of chunks added."""
        pages = self._parser.parse(file_bytes, filename)
        all_chunks: List[Chunk] = []
        for text, page_num in pages:
            chunks = self._chunker.split(text, source=filename, page=page_num)
            all_chunks.extend(chunks)
        self._store.add_chunks(all_chunks)
        return len(all_chunks)

    # ── Query ─────────────────────────────────────────────────────────────────

    def query_stream(
        self,
        question: str,
        chat_history: List[dict],
        top_k: int = 6,
        rerank_top_n: int = 3,
    ) -> Tuple[Generator[str, None, None], List[RetrievalResult], float]:
        """
        Returns (token_stream, retrieval_results, query_latency_ms).
        Caller consumes the stream for real-time display.
        """
        if not self._store.is_ready:
            raise RuntimeError("No documents indexed. Please upload and process documents first.")
        if self._llm is None:
            raise RuntimeError("LLM not configured. Please provide API key and select a model.")

        t0 = time.perf_counter()
        results = self._store.search(question, top_k=top_k, rerank_top_n=rerank_top_n)
        latency_ms = (time.perf_counter() - t0) * 1000

        messages = _build_rag_prompt(
            query=question,
            results=results,
            chat_history=chat_history,
            policy_mode=self._policy_mode,
            policy_text=self._policy_text,
        )

        stream = self._llm.generate_stream(messages)
        return stream, results, latency_ms

    # ── Metadata ──────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._store.is_ready and self._llm is not None

    @property
    def stats(self) -> dict:
        return {
            "total_chunks": self._store.total_chunks,
            "chunk_size": self._chunker.chunk_size,
            "overlap": self._chunker.overlap,
            "embed_model": FAISSVectorStore.EMBED_MODEL,
            "reranker": FAISSVectorStore.RERANK_MODEL if self._store.uses_reranker else "disabled",
            "policy_mode": self._policy_mode,
            "llm_provider": self._llm.provider if self._llm else "none",
            "llm_model": self._llm.model if self._llm else "none",
        }
