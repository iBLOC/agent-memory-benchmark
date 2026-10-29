#!/usr/bin/env python3
"""Context precision@k for a retrieval run: how much of the retrieved set
is gold-bearing.

EdgeQuake gates its benchmark on `ctx_rel` (context relevancy) — the share
of the reader's context that actually carries the answer — and fails that
gate at 0.414. This is the same idea measured on the retrieved document
set: of the k documents handed to the reader, how many are gold documents.

Usage: python scripts/precision_at_k.py outputs/<run>/retrieval/s.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    path = Path(sys.argv[1])
    data = json.loads(path.read_text())
    results = data.get("results", [])
    hits = 0
    gold_total = 0
    doc_total = 0
    queries_with_gold = 0
    for r in results:
        gold = set(r.get("gold_ids") or [])
        docs = (r.get("raw_response") or {}).get("documents") or []
        if not gold:
            continue
        queries_with_gold += 1
        ids = {str(d.get("id")) for d in docs}
        hits += len(ids & gold)
        gold_total += len(gold)
        doc_total += len(docs)
    if not queries_with_gold:
        print("no gold-bearing queries in this run")
        return 1
    print(f"queries: {queries_with_gold}")
    print(f"avg docs per query: {doc_total / queries_with_gold:.1f}")
    print(f"gold docs in the retrieved set: {hits}")
    print(f"precision@k (gold-bearing / retrieved): {hits / max(doc_total, 1):.4f}")
    print(f"recall of gold in the retrieved set: {hits / max(gold_total, 1):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
