#!/usr/bin/env python3
"""Keyless baselines: BM25Okapi + vanilla (no-retrieval).

Sibling of scripts/run_preprocessed.py. Same input (the preprocessed
`data/<dataset>/<split>/queries.json.gz` + `documents.json.gz`),
same scoring contract, but the retrieval stage is replaced:

  - bm25     : tokenize every doc + query, BM25Okapi, top-10.
  - vanilla  : no memory at all -> 0/0/0 on every dataset (returned
               as hit@1=hit@10=0, mrr=0, and a non-zero mrr only if
               the gold happens to be a memory id that the benchmark
               marks as "always pass" without retrieval; in practice
               mrr=0).

Both baselines run purely locally (no hippocampus server required),
so they're the fastest sanity check that any retrieval system is
actually doing work.

Usage:
    python scripts/run_baselines.py <dataset> <split> [options]

Options:
    --limit N        max queries (default: all)
    --topk N         recall cutoff (default 10)
    --seed N         sample N queries (default: all)
    --out PATH       output JSON path
    --name NAME      run name (default: bm25-<dataset>-<split>)

Example:
    cd agent-memory-benchmark
    python scripts/run_baselines.py personamem 32k --limit 300 --topk 10
"""

import argparse
import gzip
import json
import random
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data"

# Light tokenization: lowercase, keep alnum + cjk chars, split on
# whitespace. Mirrors what Lance's Tantivy-style inverted index does
# well enough for a like-for-like BM25 baseline.
_TOK = re.compile(r"[\w一-鿿]+", re.UNICODE)


def tokenize(text):
    return _TOK.findall((text or "").lower())


def load_split(dataset, split):
    split_dir = DATA_DIR / dataset / split
    with gzip.open(split_dir / "queries.json.gz", "rt", encoding="utf-8") as fh:
        queries = json.load(fh)
    with gzip.open(split_dir / "documents.json.gz", "rt", encoding="utf-8") as fh:
        documents = json.load(fh)
    return queries, documents


def bm25_topk(queries, scoped_docs, topk, limit, seed, char_limit):
    if seed is not None:
        random.seed(seed)
        queries = random.sample(queries, min(limit or len(queries), len(queries)))
    elif limit:
        queries = queries[:limit]

    # Build BM25 index scoped to the relevant tenants.
    wanted = {q.get("user_id") for q in queries}
    scoped = [d for d in scoped_docs if (d.get("user_id") or "default") in wanted]
    if not scoped:
        return {"baseline": "bm25", "total_queries": len(queries),
                "hit@1": 0.0, "hit@10": 0.0, "mrr": 0.0,
                "avg_retrieve_ms": 0.0, "note": "no docs in scope"}

    from rank_bm25 import BM25Okapi
    tokenized_corpus = [tokenize(d.get("content", "")[:char_limit]) for d in scoped]
    bm25 = BM25Okapi(tokenized_corpus)

    hit1 = hitk = 0
    rr = 0.0
    latencies = []
    for q in queries:
        q_text = q.get("query", "")
        t0 = time.perf_counter()
        scores = bm25.get_scores(tokenize(q_text))
        ranked = sorted(range(len(scored := scores)), key=lambda i: -scored[i])[:topk]
        latencies.append((time.perf_counter() - t0) * 1000)
        uris = [scoped[i]["id"] for i in ranked]
        gold = set(q.get("gold_ids") or [])
        rank = next((j + 1 for j, u in enumerate(uris) if u in gold), None)
        if rank == 1:
            hit1 += 1
        if rank:
            hitk += 1
            rr += 1.0 / rank

    n = len(queries)
    return {
        "baseline": "bm25",
        "total_queries": n,
        "ingested_docs": len(scoped),
        "hit@1": round(hit1 / n, 4) if n else 0.0,
        f"hit@{topk}": round(hitk / n, 4) if n else 0.0,
        "mrr": round(rr / n, 4) if n else 0.0,
        "avg_retrieve_ms": round(sum(latencies) / n, 1) if n else 0.0,
    }


def vanilla_topk(queries, topk, limit, seed):
    """No-memory baseline: every score is 0 by construction."""
    if seed is not None:
        random.seed(seed)
        queries = random.sample(queries, min(limit or len(queries), len(queries)))
    elif limit:
        queries = queries[:limit]
    n = len(queries)
    return {
        "baseline": "vanilla",
        "total_queries": n,
        "hit@1": 0.0,
        f"hit@{topk}": 0.0,
        "mrr": 0.0,
        "avg_retrieve_ms": 0.0,
        "note": "no-memory baseline: every retrieved set is empty",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset")
    ap.add_argument("split")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--char-limit", type=int, default=6000)
    ap.add_argument("--baseline", choices=["bm25", "vanilla", "both"], default="both")
    args = ap.parse_args()

    queries, documents = load_split(args.dataset, args.split)
    print(f"{args.dataset}/{args.split}: {len(queries)} queries, {len(documents)} docs")

    results = {}
    if args.baseline in ("bm25", "both"):
        t0 = time.perf_counter()
        r = bm25_topk(queries, documents, args.topk, args.limit, args.seed, args.char_limit)
        r["wallclock_s"] = round(time.perf_counter() - t0, 1)
        results["bm25"] = r
        print(f"  bm25     hit@{args.topk}={r[f'hit@{args.topk}']:.4f}  MRR={r['mrr']:.4f}  ({r['wallclock_s']}s)")

    if args.baseline in ("vanilla", "both"):
        results["vanilla"] = vanilla_topk(queries, args.topk, args.limit, args.seed)
        print(f"  vanilla  hit@{args.topk}={results['vanilla'][f'hit@{args.topk}']:.4f}  (no-memory)")

    out = Path(args.out) if args.out else (
        REPO / "outputs" / args.dataset / (args.name or f"bm25-{args.dataset}-{args.split}") / "retrieval"
        / f"{args.split}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
