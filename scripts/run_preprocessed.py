#!/usr/bin/env python3
"""Offline AMB runner for preprocessed benchmark exports.

The iBLOC/agent-memory-benchmark dataset loaders (locomo, beam, lifebench,
personamem, sdebench) all fetch their *raw* upstream JSON on first use
(LOCOMO_DATA_PATH / BEAM_DATA_PATH / LIFEBENCH_DATA_PATH or HuggingFace
download). On this machine those upstreams are unreachable (raw.githubusercontent
returns 403/404 for snap-research/locomo, HF requires auth for the gated
personamem/beam repos), so the loaders abort before ingesting anything.

But the repo already ships a *preprocessed* export under
`data/<dataset>/<split>/`:

    queries.json.gz    -> [{id, query, gold_ids, gold_answers, user_id, meta}]
    documents.json.gz  -> [{id, content, user_id, timestamp}]

which carries exactly the gold_ids we need for hit@k scoring. This runner
reads those two files directly and drives the hippocampus provider over
HTTP, bypassing the dataset loaders entirely.

Usage:
    python scripts/run_preprocessed.py <dataset> <split> [options]

Options:
    --url URL            hippocampus base URL (default http://127.0.0.1:8081)
    --limit N            max queries to evaluate
    --seed N             sample N queries with this RNG seed (default: all)
    --topk N             recall cutoff (default 10)
    --out PATH           write the recall report JSON here
    --name NAME          run name under outputs/<dataset>/<name>/

Environment (must match the running server):
    LLM_API_BASE, LLM_API_KEY, EMBEDDING_MODEL, LANCE_EMBEDDING_DIM,
    HIPPOCAMPUS_DOC_CHAR_LIMIT, HIPPOCAMPUS_FAST_RETRIEVE

Example:
    cd agent-memory-benchmark
    PYTHONPATH=src \
    LLM_API_BASE=http://127.0.0.1:8086/v1 LLM_API_KEY=dummy \
    EMBEDDING_MODEL=all-MiniLM-L6-v2 LANCE_EMBEDDING_DIM=384 \
    HIPPOCAMPUS_DOC_CHAR_LIMIT=6000 HIPPOCAMPUS_FAST_RETRIEVE=1 \
    uv run --no-sync python scripts/run_preprocessed.py \
      personamem 32k --limit 200 --topk 10
"""

import argparse
import gzip
import json
import os
import random
import sys
import time
from pathlib import Path
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data"
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


# ── HTTP ────────────────────────────────────────────────────────────────────

def request(url_base, method, path, body=None, timeout=300):
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = Request(f"{url_base}{path}", data=data, headers=headers, method=method)
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except Exception as exc:  # noqa: BLE001 - surface as RuntimeError
        raise RuntimeError(f"{method} {path} failed: {exc}") from exc
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{method} {path} returned non-JSON body ({len(raw)} bytes)"
        ) from exc


def embed_texts(url_base, texts):
    """Call the local MiniLM embedding server (OpenAI-compatible)."""
    api_base = os.environ.get("LLM_API_BASE", "")
    api_key = os.environ.get("LLM_API_KEY", "dummy")
    if not api_base:
        raise RuntimeError("LLM_API_BASE must point at the embedding endpoint")
    payload = {"input": texts, "model": os.environ.get("EMBEDDING_MODEL", "x")}
    req = Request(
        f"{api_base.rstrip('/')}/embeddings",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read())
    return [d["embedding"] for d in body["data"]]


# ── data loading ────────────────────────────────────────────────────────────

def load_split(dataset, split):
    split_dir = DATA_DIR / dataset / split
    if not split_dir.is_dir():
        sys.exit(f"no preprocessed split at {split_dir}")
    with gzip.open(split_dir / "queries.json.gz", "rt", encoding="utf-8") as fh:
        queries = json.load(fh)
    with gzip.open(split_dir / "documents.json.gz", "rt", encoding="utf-8") as fh:
        documents = json.load(fh)
    return queries, documents


# ── ingest / retrieve ───────────────────────────────────────────────────────

def ingest(url_base, documents, char_limit):
    pending: list[dict] = []

    def flush():
        """Embed and POST the buffered docs, then clear the buffer."""
        if not pending:
            return
        chunk = pending[:]
        pending.clear()
        vectors = embed_texts(url_base, chunk)
        for doc, vec in zip(chunk, vectors):
            doc["embedding"] = vec
            request(url_base, "POST", "/api/memory", doc)

    for doc in documents:
        content = doc.get("content") or ""
        if char_limit:
            content = content[:char_limit]
        pending.append(
            {
                "uri": doc["id"],
                "content": content,
                "dimension": "resources",
                "tenantId": doc.get("user_id") or "default",
            }
        )
        if len(pending) >= 64:
            flush()
    flush()


def retrieve(url_base, query, tenant, topk):
    t0 = time.perf_counter()
    hits = request(
        url_base,
        "POST",
        "/api/search",
        {"query": query, "limit": topk, "minScore": 0.0, "tenantId": tenant},
    ) or []
    return [h.get("memory", {}).get("uri") for h in hits], (time.perf_counter() - t0) * 1000


# ── scoring ─────────────────────────────────────────────────────────────────

def score(queries, docs_by_uri, url_base, topk, limit, seed):
    """Score retrieved hits against gold_ids.

    Two gold_id conventions appear in the preprocessed exports:
      * PMB / LME / LoCoMo / PersonaMem — gold_ids is a list of *doc URIs*
        (the retrieved `uri` matches it directly).
      * BEAM / lifebench / some others — gold_ids is a list of *user ids*
        (i.e. the first "synthetic" URI of the conversation); the
        benchmark scores hit@k as "did we surface *any* doc from that
        user". We detect that by seeing whether any gold_id string
        is a prefix of any doc URI in our store.

    In both cases the scoring is per-query and the aggregate metric
    is hit@k / MRR; for the user-id case the rank is the position of
    the first *tenant-matching* URI, not the first gold-id substring.
    """
    hit1 = hitk = 0
    rr = 0.0
    latencies = []
    user_prefix_matches = 0
    for i, q in enumerate(queries):
        uris, ms = retrieve(
            url_base, q["query"], q.get("user_id") or "default", topk
        )
        latencies.append(ms)
        gold = q.get("gold_ids") or []
        gold_set = set(gold)

        # Decide if these gold_ids look like doc URIs or user ids.
        is_user_id = bool(gold) and not any(
            g in docs_by_uri for g in gold
        )

        if is_user_id:
            # Mark any URIs whose doc_uri starts with one of the gold ids.
            rank = None
            for j, u in enumerate(uris):
                u_id = u.split("__", 1)[0] if "__" in u else u
                if any(u_id.startswith(g) or u.startswith(g) for g in gold):
                    rank = j + 1
                    break
            if rank is None and any(
                u.startswith(g) for u in uris for g in gold
            ):
                rank = 1
            user_prefix_matches += int(bool(rank))
        else:
            rank = next((j + 1 for j, u in enumerate(uris) if u in gold_set), None)
            # Fallback: gold_ids may be empty (mcq) — check if the gold
            # answer text appears anywhere in the top-k retrieved content.
            # Works for BEAM (which has gold_answers like "There is no
            # information..." for abstention) and similar open-qa splits.
            if rank is None:
                gold_text = " ".join(q.get("gold_answers") or []).lower()
                if gold_text and len(gold_text) > 20:
                    needle = gold_text[:80]  # first 80 chars is plenty
                    for j, u in enumerate(uris):
                        if needle in (docs_by_uri.get(u, {}).get("content", "")).lower():
                            rank = j + 1
                            break

        if rank == 1:
            hit1 += 1
        if rank:
            hitk += 1
            rr += 1.0 / rank
        if (i + 1) % 25 == 0:
            print(
                f"  {i + 1}/{len(queries)} queries | "
                f"hit@{topk}={hitk / (i + 1):.3f} | "
                f"avg {sum(latencies) / len(latencies):.0f} ms",
                flush=True,
            )
    n = len(queries)
    return {
        "total_queries": n,
        "hit@1": round(hit1 / n, 4) if n else 0.0,
        f"hit@{topk}": round(hitk / n, 4) if n else 0.0,
        "mrr": round(rr / n, 4) if n else 0.0,
        "avg_retrieve_ms": round(sum(latencies) / n, 1) if n else 0.0,
        "p95_retrieve_ms": (
            round(sorted(latencies)[int(len(latencies) * 0.95)], 1) if n else 0.0
        ),
        "user_id_queries_matched": user_prefix_matches,
    }


# ── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset")
    ap.add_argument("split")
    ap.add_argument("--url", default=os.environ.get("HIPPOCAMPUS_URL", "http://127.0.0.1:8081"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--out", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--skip-ingest", action="store_true")
    args = ap.parse_args()

    url_base = args.url.rstrip("/")
    host = urlparse(url_base).hostname or ""
    if host not in LOCAL_HOSTS:
        sys.exit(f"refusing non-local hippocampus URL {url_base}")

    queries, documents = load_split(args.dataset, args.split)
    print(f"{args.dataset}/{args.split}: {len(queries)} queries, {len(documents)} docs")

    # Only ingest docs belonging to tenants that actually have a scored query;
    # that keeps a --limit run from paying for the whole split.
    if args.seed is not None:
        random.seed(args.seed)
        queries = random.sample(queries, min(args.limit or len(queries), len(queries)))
    elif args.limit:
        queries = queries[: args.limit]

    wanted = {q.get("user_id") for q in queries}
    scoped = [d for d in documents if (d.get("user_id") or "default") in wanted]
    print(f"scoped to {len(wanted)} tenants -> {len(scoped)} docs, {len(queries)} queries")

    request(url_base, "GET", "/health")

    if not args.skip_ingest:
        t0 = time.perf_counter()
        char_limit = int(os.environ.get("HIPPOCAMPUS_DOC_CHAR_LIMIT", "6000"))
        ingest(url_base, scoped, char_limit)
        print(f"ingested {len(scoped)} docs in {time.perf_counter() - t0:.1f}s")

    # Build docs_by_uri unconditionally — the gold-text fallback in
    # `score` needs the source content even on --skip-ingest reruns.
    docs_by_uri = {d["id"]: d for d in scoped}
    report = score(queries, docs_by_uri, url_base, args.topk, args.limit, args.seed)
    report.update({"dataset": args.dataset, "split": args.split, "topk": args.topk})

    print(json.dumps(report, indent=2))
    out = Path(args.out) if args.out else (
        REPO / "outputs" / args.dataset / (args.name or "preprocessed") / "retrieval"
        / f"{args.split}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report -> {out}")


if __name__ == "__main__":
    main()