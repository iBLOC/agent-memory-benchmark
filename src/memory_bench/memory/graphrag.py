"""GraphRAGMemoryProvider — Microsoft GraphRAG via its CLI.

The installed graphrag (post-relaunch) changes its Python API and settings
schema between minor versions, so this provider goes through the CLI for all
three phases instead of importing internals:

    graphrag init   → prompts/ + settings.yaml scaffold (once per user root)
    graphrag index  → builds the knowledge graph + Lance vector store
    graphrag query  → local/global search over the index artifacts

Each `user_id` gets its own workspace under `<store_dir>/graphrag_workspace/`,
so settings-relative `input/` / `output/` base dirs resolve per user.

LLM key requirement: this graphrag version has no mock provider — indexing
needs a real OpenAI-compatible key (`OPENAI_API_KEY` or `LLM_API_KEY`).
"""

import os
import shutil
import subprocess
import threading
import uuid
from pathlib import Path

from ..models import Document
from .base import MemoryProvider

# Characters that must never appear in any path passed to subprocess argv.
# store_dir comes from the AMB runner's tmpdir (operator-controlled); the
# `_safe_root` validator below is defence in depth. Backslash and drive
# colon are legal Windows path characters and must stay allowed.
_FORBIDDEN_PATH_CHARS = set(";&|`$<>(){}[]!#\"'*?~\n\r")


def _safe_root(p: Path) -> Path:
    s = str(p)
    if any(c in _FORBIDDEN_PATH_CHARS for c in s):
        raise ValueError(f"unsafe root_dir: {s!r}")
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
    return p


def _write_input_files(docs: list[Document], input_dir: Path) -> None:
    """GraphRAG's pipeline expects one .txt per document in input/."""
    input_dir.mkdir(parents=True, exist_ok=True)
    for doc in docs:
        safe_id = doc.id.replace("/", "_")[:64] or uuid.uuid4().hex[:8]
        (input_dir / f"{safe_id}.txt").write_text(doc.content, encoding="utf-8")


class GraphRAGMemoryProvider(MemoryProvider):
    name = "graphrag"
    description = (
        "Microsoft GraphRAG: LLM-driven community detection over an LLM-extracted "
        "entity graph. ingest() runs `graphrag index`; retrieve() merges "
        "`graphrag query --method local` and `--method global` answers."
    )
    kind = "local"
    provider = "graphrag"
    variant = "index-on-disk"
    link = "https://github.com/microsoft/graphrag"
    logo = "https://www.google.com/s2/favicons?sz=32&domain=github.com"
    concurrency = 1  # indexing is a single CLI run; reads serialise on _lock

    def __init__(self):
        self._default_user_id = f"bench_{uuid.uuid4().hex[:8]}"
        self._root_dir: Path | None = None
        self._indexed_users: set[str] = set()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # MemoryProvider surface
    # ------------------------------------------------------------------

    def prepare(self, store_dir: Path, unit_ids: set[str] | None = None, reset: bool = True) -> None:
        if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")):
            raise RuntimeError(
                "graphrag requires an OpenAI-compatible key (OPENAI_API_KEY or "
                "LLM_API_KEY) — this graphrag version ships no mock provider."
            )
        self._root_dir = store_dir / "graphrag_workspace"
        if reset and self._root_dir.exists():
            shutil.rmtree(self._root_dir)
        self._root_dir.mkdir(parents=True, exist_ok=True)
        _safe_root(self._root_dir)
        self._indexed_users.clear()

    def _graphrag(self, *argv: str, cwd: Path | None = None, timeout: int = 3600,
                  stdin: bytes | None = None) -> str:
        """Run the graphrag CLI with an argv list (no shell). Raises with the
        CLI's own stderr/stdout tail on failure."""
        env = os.environ.copy()
        cmd = ["graphrag", *argv]
        result = subprocess.run(
            cmd,
            shell=False,
            input=stdin,
            capture_output=True,
            cwd=str(cwd) if cwd else None,
            env=env,
            timeout=timeout,
        )
        if result.returncode != 0:
            tail_out = (result.stdout or b"")[-800:].decode("utf-8", "replace")
            tail_err = (result.stderr or b"")[-1200:].decode("utf-8", "replace")
            raise RuntimeError(
                f"graphrag {' '.join(argv[:2])} failed (rc={result.returncode}): "
                f"{tail_err or tail_out}"
            )
        return (result.stdout or b"").decode("utf-8", "replace")

    def _index_root_for(self, user_id: str) -> Path:
        # Per-user workspace: prompts/ + settings.yaml + input/ + output/ live
        # inside so graphrag's relative base dirs resolve against the user's
        # own corpus.
        root = self._root_dir / ("user_" + user_id.replace("-", "_"))
        _safe_root(root)
        if not (root / "settings.yaml").exists():
            # `graphrag init` scaffolds prompts/ + settings.yaml; pipe newlines
            # to accept the interactive model prompts (we overwrite settings
            # right after).
            self._graphrag("init", "--root", str(root), stdin=b"\n\n", timeout=300)
            self._write_settings_yaml(root)
        return root

    def _write_settings_yaml(self, root: Path) -> None:
        """Overwrite the init scaffold with benchmark-tuned settings in the
        post-relaunch schema (completion_models / embedding_models registries)."""
        llm_model = os.environ.get("GRAPH_RAG_LLM_MODEL", "gpt-4o-mini")
        emb_model = os.environ.get("GRAPH_RAG_EMB_MODEL", "text-embedding-3-small")
        content = f"""\
completion_models:
  default_completion_model:
    model_provider: openai
    model: {llm_model}
    auth_method: api_key
    api_key: ${{GRAPHRAG_API_KEY}}
    retry:
      type: exponential_backoff

embedding_models:
  default_embedding_model:
    model_provider: openai
    model: {emb_model}
    auth_method: api_key
    api_key: ${{GRAPHRAG_API_KEY}}
    retry:
      type: exponential_backoff

input:
  type: text

chunking:
  type: tokens
  size: 1200
  overlap: 100
  encoding_model: o200k_base

input_storage:
  type: file
  base_dir: "input"

output_storage:
  type: file
  base_dir: "output"

reporting:
  type: file
  base_dir: "logs"

cache:
  type: json
  storage:
    type: file
    base_dir: "cache"

vector_store:
  type: lancedb
  db_uri: output/lancedb

embed_text:
  embedding_model_id: default_embedding_model

extract_graph:
  completion_model_id: default_completion_model
  prompt: "prompts/extract_graph.txt"
  entity_types: [organization, person, geo, event]
  max_gleanings: 1

summarize_descriptions:
  completion_model_id: default_completion_model
  prompt: "prompts/summarize_descriptions.txt"
  max_length: 500

cluster_graph:
  max_cluster_size: 10

extract_claims:
  enabled: false

community_reports:
  completion_model_id: default_completion_model
  graph_prompt: "prompts/community_report_graph.txt"
  text_prompt: "prompts/community_report_text.txt"
  max_length: 2000
  max_input_length: 8000

snapshots:
  graphml: false
  embeddings: false

local_search:
  completion_model_id: default_completion_model
  embedding_model_id: default_embedding_model
  prompt: "prompts/local_search_system_prompt.txt"

global_search:
  completion_model_id: default_completion_model
  map_prompt: "prompts/global_search_map_system_prompt.txt"
  reduce_prompt: "prompts/global_search_reduce_system_prompt.txt"
  knowledge_prompt: "prompts/global_search_knowledge_system_prompt.txt"
"""
        (root / "settings.yaml").write_text(content, encoding="utf-8")
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY", "")
        (root / ".env").write_text(f"GRAPHRAG_API_KEY={api_key}\n", encoding="utf-8")

    def ingest(self, documents: list[Document]) -> None:
        if self._root_dir is None:
            self.prepare(Path(os.environ.get("GRAPH_RAG_TMPDIR", "/tmp")) / "graphrag")
        by_user: dict[str, list[Document]] = {}
        for doc in documents:
            uid = doc.user_id or self._default_user_id
            by_user.setdefault(uid, []).append(doc)
        with self._lock:
            for user_id, docs in by_user.items():
                if user_id in self._indexed_users:
                    continue
                user_root = self._index_root_for(user_id)
                _write_input_files(docs, user_root / "input")
                self._graphrag("index", "--root", str(user_root), cwd=user_root)
                self._indexed_users.add(user_id)

    def retrieve(
        self, query: str, k: int = 10, user_id: str | None = None,
        query_timestamp: str | None = None,
    ) -> tuple[list[Document], dict | None]:
        uid = user_id or self._default_user_id
        if uid not in self._indexed_users:
            return ([], {"warning": f"no index for user_id={uid}"})
        user_root = self._index_root_for(uid)

        # GraphRAG returns synthesised natural-language answers, not raw
        # passages. Each search method's answer is surfaced as one Document;
        # ID-scored datasets (PrecisionMemBench) should expect ~0 here —
        # this provider's real evaluation happens on judge-scored datasets.
        with self._lock:
            local = self._graphrag(
                "query", query, "--root", str(user_root),
                "--method", "local", "--community-level", "2",
                cwd=user_root, timeout=600,
            )
            global_ = self._graphrag(
                "query", query, "--root", str(user_root),
                "--method", "global", "--community-level", "2",
                cwd=user_root, timeout=600,
            )
        docs = [
            Document(id=f"{uid}-local",  content=local.strip()),
            Document(id=f"{uid}-global", content=global_.strip()),
        ]
        return docs, {"local": local, "global": global_}
