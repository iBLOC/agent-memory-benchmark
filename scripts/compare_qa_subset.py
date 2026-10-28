#!/usr/bin/env python3
"""Compare two QA runs on the intersection of their query sets.

The AMB harness scores `open` datasets by judge LLM, so two providers can only
be compared on the same questions. This script intersects the `query_id` sets of
two result JSONs and reports each provider's accuracy on the shared subset
plus the per-question disagreements.

Usage:
    python scripts/compare_qa_subset.py \
        outputs/longmemeval/qa500-ds/rag/s.json \
        outputs/longmemeval/lme-mem0-100q/rag/s.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def load(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text())
    return {r["query_id"]: r for r in data.get("results", [])}


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    left_path, right_path = Path(sys.argv[1]), Path(sys.argv[2])
    left, right = load(left_path), load(right_path)
    shared = sorted(set(left) & set(right))
    if not shared:
        print("no shared query_id between the two runs")
        return 1

    def tally(rows: dict[str, dict]) -> tuple[int, float]:
        ok = sum(1 for q in shared if rows[q].get("correct"))
        return ok, ok / len(shared)

    left_ok, left_acc = tally(left)
    right_ok, right_acc = tally(right)
    print(f"shared queries: {len(shared)}")
    print(f"{left_path.parent.name:28s} {left_ok:3d}/{len(shared)}  {left_acc:.3f}")
    print(f"{right_path.parent.name:28s} {right_ok:3d}/{len(shared)}  {right_acc:.3f}")

    print("\ndisagreements:")
    for q in shared:
        l, r = left[q].get("correct"), right[q].get("correct")
        if l != r:
            winner = left_path.parent.name if l else right_path.parent.name
            print(f"  {q}  {left_path.parent.name}={l} {right_path.parent.name}={r}  -> {winner}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
