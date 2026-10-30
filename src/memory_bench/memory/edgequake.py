"""EdgeQuake — Rust GraphRAG (the LightRAG algorithm) over PostgreSQL.

Why this provider exists: the AMB matrix had `lightrag` (Python library, stubbed
embeddings) and `graphrag` (CLI, keyless indexing timed out) but no Rust
GraphRAG row. EdgeQuake is a single Rust binary with a REST API, and its
workspace header maps one-to-one onto the benchmark's one-user-per-question
isolation, so it slots in where the other two were impractical.

Setup (see docs/operations/edgequake.md in the hippocampus repo):
  - EdgeQuake API on EDGEQUAKE_URL (default http://127.0.0.1:8080/api/v1)
  - one workspace per user, created in prepare()
  - documents go in with POST /documents, queries with POST /query/context

Retrieval returns the retrieved chunks as documents so the harness's reader
answers them, exactly like every other provider in the matrix — the score
therefore measures the retrieval architecture, not the reader.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from ..models import Document
from .base import MemoryProvider

_DEFAULT_TENANT = "00000000-0000-0000-0000-000000000002"


class EdgeQuakeMemoryProvider(MemoryProvider):
    name = "edgequake"
    description = "EdgeQuake: Rust GraphRAG (LightRAG algorithm) on PostgreSQL + pgvector"
    kind = "local"
    provider = "edgequake"
    variant = "local"
    link = "https://github.com/raphaelmansuy/edgequake"
    logo = "https://www.google.com/s2/favicons?domain=github.com"
    concurrency = 2
    supports_filters = False

    def __init__(self, k: int = 10):
        self.k = k
        self.base = os.environ.get("EDGEQUAKE_URL", "http://127.0.0.1:8080/api/v1").rstrip("/")
        self.tenant = os.environ.get("EDGEQUAKE_TENANT_ID", _DEFAULT_TENANT)
        self.mode = os.environ.get("EDGEQUAKE_QUERY_MODE", "mix")
        self.timeout = float(os.environ.get("EDGEQUAKE_TIMEOUT", "300"))
        # Model ids as they appear in EdgeQuake's catalogue. The workspace
        # inherits the tenant default ("local-model"), which is not a
        # catalogue entry, so a fresh workspace has to be configured before
        # it can ingest.
        self.llm_model_id = os.environ.get("EDGEQUAKE_LLM_MODEL_ID", "default")
        self.embedding_model_id = os.environ.get(
            "EDGEQUAKE_EMBEDDING_MODEL_ID", "text-embedding-nomic-embed-text-v1.5"
        )
        self.embedding_dim = int(os.environ.get("EDGEQUAKE_EMBEDDING_DIM", "768"))
        self._llm_provider = os.environ.get("EDGEQUAKE_LLM_PROVIDER", "lmstudio")
        # Chunk size drives the LLM call count: extraction and description run
        # per chunk, so 1200-token chunks over a 15k-character session mean
        # four extraction rounds per document. 4000 halves it.
        self.chunk_tokens = int(os.environ.get("EDGEQUAKE_CHUNK_TOKENS", "4000"))
        self.chunk_overlap = int(os.environ.get("EDGEQUAKE_CHUNK_OVERLAP", "100"))
        self._embedding_provider = os.environ.get("EDGEQUAKE_EMBEDDING_PROVIDER", "lmstudio")
        self._workspaces: dict[str, str] = {}
        self._by_doc: dict[str, str] = {}
        # A fresh workspace per run: EdgeQuake's wipe is asynchronous and
        # rejects uploads with 409 while it runs, and reusing a workspace
        # would leak the previous run's documents into this run's retrieval.
        self._run_tag = os.environ.get("EDGEQUAKE_RUN_TAG", "") or uuid.uuid4().hex[:8]

    # ------------------------------------------------------------------ http

    def _request(self, method: str, path: str, body: dict | None = None,
                 workspace: str | None = None,
                 headers: dict | None = None) -> dict:
        url = f"{self.base}{path}"
        headers = {"Content-Type": "application/json", **(headers or {})}
        if workspace:
            headers["X-Workspace-ID"] = workspace
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            body_text = e.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(f"edgequake {method} {path} -> HTTP {e.code}: {body_text}") from e
        return json.loads(raw or b"{}")

    # -------------------------------------------------------------- lifecycle

    def prepare(self, store_dir: Path, unit_ids: set[str] | None = None, reset: bool = True) -> None:
        """Create one workspace per user.

        The benchmark's isolation unit is the user, and EdgeQuake scopes
        documents by workspace, so a workspace per user is the equivalent of
        a per-user memory bank.
        """
        self._workspaces.clear()
        self._by_doc.clear()
        if not unit_ids:
            return
        if reset:
            self._drop_stale_workspaces()
        for uid in sorted(unit_ids):
            ws = self._create_workspace(uid)
            self._configure_workspace(ws)
            self._workspaces[uid] = ws

    def _drop_stale_workspaces(self) -> None:
        """Remove workspaces from earlier runs (the tenant caps them)."""
        try:
            existing = self._request("GET", f"/tenants/{self.tenant}/workspaces")
        except Exception:  # noqa: BLE001 - best effort
            return
        for ws in existing.get("items") or existing.get("workspaces") or []:
            name = str(ws.get("name") or "")
            if self._run_tag in name:
                continue
            try:
                self._request("DELETE", f"/workspaces/{ws['id']}")
            except Exception:  # noqa: BLE001
                pass

    def _create_workspace(self, uid: str) -> str:
        name = f"{uid}-{self._run_tag}"
        existing = self._request("GET", f"/tenants/{self.tenant}/workspaces")
        for ws in existing.get("items") or existing.get("workspaces") or []:
            if ws.get("name") == name:
                return ws["id"]
        created = self._request(
            "POST", f"/tenants/{self.tenant}/workspaces", {"name": name}
        )
        return created["id"]

    def _configure_workspace(self, ws: str) -> None:
        """Point a workspace at the models the benchmark runs with.

        Two things are needed: the policy rows that say which model serves
        which usage, and the workspace state that names the models. A fresh
        workspace inherits the tenant defaults, which are not catalogue
        entries, so without both it refuses to ingest.
        """
        try:
            self._request("PUT", f"/workspaces/{ws}/model-policy", {"rows": [
                {"provider": self._llm_provider, "model": self.llm_model_id, "usage": usage}
                for usage in ("extract", "query", "summary", "keyword")
            ] + [
                {"provider": self._embedding_provider, "model": self.embedding_model_id,
                 "usage": "embedding"},
            ]}, workspace=ws)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"edgequake could not set the model policy for {ws}: {exc}") from exc
        try:
            self._request("PUT", f"/workspaces/{ws}", {
                "llm_provider": self._llm_provider,
                "llm_model": self.llm_model_id,
                "embedding_provider": self._embedding_provider,
                "embedding_model": self.embedding_model_id,
                "embedding_dimension": self.embedding_dim,
                "chunk_token_size": self.chunk_tokens,
                "chunk_overlap_token_size": self.chunk_overlap,
                "chunking_mode": "fixed",
            }, workspace=ws)
        except Exception as exc:  # noqa: BLE001 - a misconfigured workspace
            # should surface as a clear ingestion error, not a crash later.
            raise RuntimeError(f"edgequake could not configure workspace {ws}: {exc}") from exc

    def _wipe_workspace(self, ws: str) -> None:
        """Clear documents left in a reused workspace (reset runs).

        The wipe is asynchronous: it returns a track id and the API rejects
        uploads with 409 "Workspace wipe in progress" until it finishes, so
        the wipe has to be waited for before the first upload.
        """
        try:
            out = self._request("DELETE", "/documents", {}, workspace=ws,
                                 headers={"X-EdgeQuake-Confirm": "delete-all-documents"})
        except Exception:  # noqa: BLE001 - wiping is best effort
            return
        track = out.get("wipe_track_id") or out.get("task_id") or out.get("track_id")
        if track:
            deadline = time.time() + 300
            while time.time() < deadline:
                status, _ = self._task_status(track, ws)
                if status in ("completed", "succeeded", "done", "failed", "error"):
                    return
                time.sleep(2)

    def _workspace_for(self, user_id: str | None) -> str | None:
        if not user_id:
            return None
        if user_id not in self._workspaces:
            ws = self._create_workspace(user_id)
            self._configure_workspace(ws)
            self._workspaces[user_id] = ws
        return self._workspaces[user_id]

    # ----------------------------------------------------------------- ingest

    def ingest(self, documents: list[Document]) -> None:
        """Queue every document, then wait for the queue to drain.

        Submitting one document and waiting for it before submitting the
        next serialises the whole ingest behind one extraction round per
        document — hours for a benchmark unit. EdgeQuake queues internally
        (21 workers), so the provider submits everything first and then
        polls the tasks together.
        """
        pending: list[tuple[str, str]] = []  # (task_id, workspace)
        for doc in documents:
            ws = self._workspace_for(doc.user_id)
            title = doc.id or f"doc-{len(self._by_doc)}"
            out = self._request(
                "POST",
                "/documents",
                {"title": title, "content": doc.content, "metadata": {"source_id": title}},
                workspace=ws,
            )
            self._by_doc[title] = ws or ""
            task = out.get("task_id") or out.get("track_id")
            if task:
                pending.append((task, ws or ""))

        deadline = time.time() + 3600
        while pending and time.time() < deadline:
            still: list[tuple[str, str]] = []
            for task, ws in pending:
                status, st = self._task_status(task, ws)
                # EdgeQuake's task states are pending / processing / indexed /
                # failed / cancelled — "indexed" is the success terminal.
                if status in ("completed", "succeeded", "done", "indexed"):
                    continue
                if status in ("failed", "error", "cancelled"):
                    raise RuntimeError(
                        f"edgequake task {task} {status}: {str(st.get('error'))[:200]}"
                    )
                still.append((task, ws))
            pending = still
            if pending:
                time.sleep(3)

    def _task_status(self, task_id: str, workspace: str | None) -> tuple[str, dict]:
        st = self._request("GET", f"/tasks/{task_id}", workspace=workspace)
        return st.get("status") or "unknown", st

    def _await_task(self, task_id: str, timeout: float = 900,
                    workspace: str | None = None) -> dict:
        deadline = time.time() + timeout
        status = "unknown"
        st: dict = {}
        while time.time() < deadline:
            st = self._request("GET", f"/tasks/{task_id}", workspace=workspace)
            status = st.get("status") or status
            if status in ("completed", "succeeded", "done", "indexed",
                          "failed", "error", "cancelled"):
                break
            time.sleep(2)
        if status in ("failed", "error", "cancelled"):
            raise RuntimeError(f"edgequake task {task_id} {status}: {str(st.get('error'))[:200]}")
        return st

    # ----------------------------------------------------------------- search

    def retrieve(self, query: str, k: int = 10, user_id: str | None = None,
                 query_timestamp: str | None = None,
                 filters: dict | None = None) -> tuple[list[Document], dict | None]:
        ws = self._workspace_for(user_id)
        out = self._request(
            "POST", "/query/context", {"query": query, "mode": self.mode}, workspace=ws
        )
        bundle = out.get("bundle") or {}
        chunks = bundle.get("chunks") or []
        docs: list[Document] = []
        for i, ch in enumerate(chunks[:k]):
            content = ch.get("content") or ch.get("text") or ""
            docs.append(Document(
                id=str(ch.get("id") or ch.get("document_id") or f"chunk-{i}"),
                content=content,
                user_id=user_id,
            ))
        # The graph arm of the mix mode returns entities and relationships
        # rather than chunks; surface them as documents too so the reader
        # sees the same evidence the graph search found.
        entities = (bundle.get("subgraph") or {}).get("entities") or []
        for ent in entities:
            if len(docs) >= k:
                break
            desc = ent.get("description") or ent.get("name") or ""
            if not desc:
                continue
            docs.append(Document(
                id=str(ent.get("id") or ent.get("graph_node_id") or ""),
                content=f"[{ent.get('entity_type', 'entity')}] {ent.get('name', '')}: {desc}",
                user_id=user_id,
            ))
        return docs, out
