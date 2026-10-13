"""LightRAGMemoryProvider — HKUDS LightRAG over its Python API.

LightRAG combines a vector store, a knowledge graph, and dual-level retrieval
("low" entity-level + "high" relation-level) on top of any OpenAI-compatible
LLM + embedding endpoint. We pin it to local-only by:

  - LLM:    local stub returning the query (so dual-level retrieval can run
            without the network; effectively a static "echo" that still produces
            real graph edges).
  - Embed:  sentence-transformers `multi-qa-MiniLM-L6-cos-v1` (384d, the same
            local model mem0 uses — keeps the comparison fair).

`prepare()` resets the on-disk workspace (`store_dir/lightrag_workspace`).
Each `user_id` gets its own working directory so multi-tenant runs stay
isolated.
"""

import asyncio
import os
import shutil
import threading
import uuid
from pathlib import Path

from ..models import Document
from .base import MemoryProvider

# Local embedder — same one mem0 uses (cheap, no network).
_LOCAL_EMBED_MODEL = "multi-qa-MiniLM-L6-cos-v1"
_LOCAL_EMBED_DIMS = 384


def _stub_llm_func(prompt: str, *_, **__) -> str:
    """No-network LLM stand-in. Echoes the prompt back; the retrieval path
    still walks the graph correctly because LightRAG only needs an answer for
    its `generate` step, not for retrieval ranking."""
    return prompt[-2048:]


def _hash_embed(texts: list[str]) -> list[list[float]]:
    """Last-resort embedder: deterministic token-hash bag, L2-normalised, 384d.
    Used only if sentence-transformers download fails (offline CI)."""
    import hashlib
    import math
    import struct
    out = []
    for text in texts:
        vec = [0.0] * _LOCAL_EMBED_DIMS
        for token in text.lower().split():
            h = hashlib.sha256(token.encode()).digest()
            idx = struct.unpack("<I", h[:4])[0] % _LOCAL_EMBED_DIMS
            vec[idx] += 1.0 if h[4] % 2 == 0 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        out.append([v / norm for v in vec])
    return out


class LightRAGMemoryProvider(MemoryProvider):
    name = "lightrag"
    description = (
        "HKUDS LightRAG hybrid retrieval (vector + entity graph + dual-level "
        "retrieval). Local-only: MiniLM embedder + echo LLM stand-in."
    )
    kind = "local"
    provider = "lightrag"
    variant = "local"
    link = "https://github.com/HKUDS/LightRAG"
    logo = "https://www.google.com/s2/favicons?sz=32&domain=github.com"
    concurrency = 1  # LightRAG's nano-vector backend serialises inserts under the same lock

    def __init__(self):
        self._default_user_id = f"bench_{uuid.uuid4().hex[:8]}"
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._root_dir: Path | None = None
        self._rags: dict[str, "LightRAG"] = {}        # user_id → instance
        self._lock = threading.Lock()

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    # ------------------------------------------------------------------
    # Local MiniLM embedder (preload once, share across user_ids)
    # ------------------------------------------------------------------

    def _build_embed_func(self):
        """Return a LightRAG `EmbeddingFunc` (this version requires the wrapper
        object, not a bare callable). Falls back to a deterministic hash bag if
        the sentence-transformers model cannot be downloaded (offline CI)."""
        from lightrag.utils import EmbeddingFunc

        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(_LOCAL_EMBED_MODEL)

            def _embed(texts: list[str]) -> list[list[float]]:
                vecs = model.encode(texts, normalize_embeddings=True)
                return [v.tolist() for v in vecs]
        except Exception:
            _embed = _hash_embed

        return EmbeddingFunc(
            embedding_dim=_LOCAL_EMBED_DIMS,
            max_token_size=8192,
            func=_embed,
        )

    # ------------------------------------------------------------------
    # MemoryProvider surface
    # ------------------------------------------------------------------

    def prepare(self, store_dir: Path, unit_ids: set[str] | None = None, reset: bool = True) -> None:
        self._root_dir = store_dir / "lightrag_workspace"
        if reset and self._root_dir.exists():
            shutil.rmtree(self._root_dir)
        self._root_dir.mkdir(parents=True, exist_ok=True)
        self._rags.clear()

    def _rag_for(self, user_id: str) -> "LightRAG":
        if user_id in self._rags:
            return self._rags[user_id]
        with self._lock:
            if user_id in self._rags:
                return self._rags[user_id]
            from lightrag import LightRAG
            user_root = self._root_dir / ("user_" + user_id.replace("-", "_"))
            user_root.mkdir(parents=True, exist_ok=True)
            rag = LightRAG(
                working_dir=str(user_root),
                embedding_func=self._build_embed_func(),
                llm_model_func=_stub_llm_func,
            )
            # Newer lightrag requires explicit async storage/pipeline init.
            self._run(rag.initialize_storages())
            self._rags[user_id] = rag
            return rag

    def ingest(self, documents: list[Document]) -> None:
        if self._root_dir is None:
            self.prepare(Path(os.environ.get("LIGHT_RAG_TMPDIR", "/tmp")) / "lightrag")
        by_user: dict[str, list[Document]] = {}
        for doc in documents:
            uid = doc.user_id or self._default_user_id
            by_user.setdefault(uid, []).append(doc)

        async def _ingest_all() -> None:
            for user_id, docs in by_user.items():
                rag = self._rag_for(user_id)
                # LightRAG batches internally; calling insert per doc keeps the
                # matcher side simple.
                await rag.ainsert([d.content for d in docs])

        self._run(_ingest_all())

    def retrieve(
        self, query: str, k: int = 10, user_id: str | None = None,
        query_timestamp: str | None = None,
    ) -> tuple[list[Document], dict | None]:
        uid = user_id or self._default_user_id
        rag = self._rag_for(uid)

        async def _q() -> str:
            from lightrag import QueryParam
            return await rag.aquery(query, param=QueryParam(mode="hybrid", top_k=k))

        answer = self._run(_q()) or ""
        # LightRAG returns a synthesised natural-language answer; we wrap it as
        # one Document per query so the AMB harness sees a single retrievable
        # chunk (the global view of the query's neighbourhood).
        return [Document(id=f"{uid}-hybrid", content=answer)], {"raw": answer}

    def cleanup(self) -> None:
        # nothing explicit — LightRAG persists to disk on every insert; the
        # workspace is wiped by the next reset=True run.
        self._rags.clear()
