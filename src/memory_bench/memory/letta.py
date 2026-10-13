"""LettaMemoryProvider — Letta (formerly MemGPT) over its REST/socket client.

Mirrors the cognee provider shape: one persistent event loop in a background
thread so SDK calls from sync ingest/retrieve work without leaking Letta's
async internals across our sync ABC.

Setup at run time:

    # Option A — point at an externally-running Letta server (the usual setup
    #            for non-toy workloads; Letta needs Postgres + an LLM endpoint
    #            to actually run reliably).
    export LETTA_BASE_URL=http://127.0.0.1:8283

    # Option B — point at Letta Cloud
    export LETTA_API_KEY=...
    # the client auto-discovers https://app.letta.com when LETTA_API_KEY is set

For a fair retrieval-only benchmark we ask Letta's archival_memory_search
endpoint (1-hop recall over the agent's recall storage). Conversational
agentic answering lives on `direct_answer`, which the runner does not call
in `mode=retrieval`.

One agent per `user_id` (the benchmark's isolation unit). Agents live for
the lifetime of the run; cleanup() deletes them in one batch to avoid
Postgres row bloat.
"""

import asyncio
import os
import threading
import uuid
from pathlib import Path

from ..models import Document
from .base import MemoryProvider


def _agent_name(user_id: str) -> str:
    return f"amb_{user_id.replace('-', '_')[:32]}"


class LettaMemoryProvider(MemoryProvider):
    name = "letta"
    description = (
        "Letta's archival-memory search. One agent per user_id; documents are "
        "delivered as user messages so Letta's core-memory summarization pipeline "
        "populates recall storage. Retrieval is the agent's `archival_memory_search` "
        "endpoint (top-k passages)."
    )
    kind = "local"
    provider = "letta"
    variant = "local"
    link = "https://github.com/letta-ai/letta"
    logo = "https://www.google.com/s2/favicons?sz=32&domain=letta.com"
    concurrency = 1  # Letta's REST endpoints are not optimised for fan-out

    def __init__(self):
        self._default_user_id = f"bench_{uuid.uuid4().hex[:8]}"
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._client = None       # set in prepare() via the background loop
        self._agents: dict[str, str] = {}   # user_id -> agent_id
        self._lock = threading.Lock()
        self._run(self._setup())

    # ------------------------------------------------------------------
    # Async bridge
    # ------------------------------------------------------------------

    def _run(self, coro):
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result()

    async def _setup(self):
        from letta_client import Letta  # the official SDK (>=0.6)

        api_key = os.environ.get("LETTA_API_KEY")
        base_url = os.environ.get("LETTA_BASE_URL")
        if api_key:
            self._client = Letta(token=api_key)
        elif base_url:
            self._client = Letta(base_url=base_url)
        else:
            # No env vars — fall back to in-process Letta SDK defaults; will
            # fail loudly at first ingest if the operator forgot to set
            # either LETTA_API_KEY or LETTA_BASE_URL.
            self._client = Letta()

    # ------------------------------------------------------------------
    # MemoryProvider surface
    # ------------------------------------------------------------------

    def prepare(self, store_dir: Path, unit_ids: set[str] | None = None, reset: bool = True) -> None:
        # Reset = drop every agent we'll create in this run. We can't enumerate
        # them (Letta's list API is paginated and slow), but the unique agent
        # name seed includes uuid4().hex[:8] so collisions across runs are
        # effectively nil.
        if reset:
            self._agents.clear()

    def _agent_for(self, user_id: str) -> str:
        if user_id in self._agents:
            return self._agents[user_id]
        with self._lock:
            if user_id in self._agents:
                return self._agents[user_id]
            agent_id = self._run(self._create_agent(user_id))
            self._agents[user_id] = agent_id
            return agent_id

    async def _create_agent(self, user_id: str) -> str:
        from letta_client.types import LLMConfig, EmbeddingConfig
        agent = await self._client.agents.create_async(
            name=_agent_name(user_id),
            memory_blocks=[{
                "label": "persona",
                "value": f"Benchmark run user_id={user_id}",
            }],
            llm_config=LLMConfig(
                model=os.environ.get("LETTA_MODEL", "letta/letta-free"),
                model_endpoint_type="openai",
                model_endpoint=os.environ.get("LLM_API_BASE", "https://api.openai.com/v1"),
                context_window=8192,
            ),
            embedding_config=EmbeddingConfig(
                model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
                embedding_endpoint_type="openai",
                embedding_endpoint=os.environ.get("LLM_API_BASE", "https://api.openai.com/v1"),
                embedding_dim=1536,
            ),
        )
        return agent.id

    def ingest(self, documents: list[Document]) -> None:
        by_user: dict[str, list[Document]] = {}
        for doc in documents:
            uid = doc.user_id or self._default_user_id
            by_user.setdefault(uid, []).append(doc)
        self._run(self._ingest_async(by_user))

    async def _ingest_async(self, by_user: dict[str, list[Document]]) -> None:
        for user_id, docs in by_user.items():
            agent_id = self._agent_for(user_id)
            for doc in docs:
                await self._client.agents.messages.create_async(
                    agent_id=agent_id,
                    messages=[{"role": "user", "content": doc.content}],
                )

    def retrieve(
        self, query: str, k: int = 10, user_id: str | None = None,
        query_timestamp: str | None = None,
    ) -> tuple[list[Document], dict | None]:
        uid = user_id or self._default_user_id
        agent_id = self._agent_for(uid)
        return self._run(self._retrieve_async(agent_id, query, k))

    async def _retrieve_async(self, agent_id: str, query: str, k: int) -> tuple[list[Document], dict | None]:
        response = await self._client.agents.archival_memory_search_async(
            agent_id=agent_id, query=query, limit=k,
        )
        # Letta returns passages with .text + .metadata or a list of dicts
        # depending on SDK version; normalise.
        docs: list[Document] = []
        raw = []
        for r in (getattr(response, "results", None) or response or []):
            text = getattr(r, "text", None) or (isinstance(r, dict) and r.get("text")) or str(r)
            rid = (
                getattr(r, "id", None)
                or (isinstance(r, dict) and r.get("id"))
                or str(uuid.uuid4())
            )
            docs.append(Document(id=str(rid), content=text))
            raw.append({"id": str(rid), "text": text})
        return docs, {"results": raw}

    def cleanup(self) -> None:
        # Best-effort teardown; the in-process event loop self-terminates
        # when the daemon thread exits.
        if self._client is not None:
            for agent_id in list(self._agents.values()):
                try:
                    self._run(self._client.agents.delete_async(agent_id=agent_id))
                except Exception:
                    pass  # best-effort: leaking a few agents across runs is fine
            self._agents.clear()
