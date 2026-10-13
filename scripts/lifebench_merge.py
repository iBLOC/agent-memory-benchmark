#!/usr/bin/env python
"""Merge the 10 LifeBench per-user unit results into a single official-split
result file at outputs/lifebench/hippocampus/rag/en.json.

Run after lifebench-peruser.sh finishes all units:
    PYTHONPATH=. python scripts/lifebench_merge.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNITS = sorted(ROOT.glob("outputs/lifebench/hippo-lifebench-mimo-u*/rag/en.json"))
OUT = ROOT / "outputs/lifebench/hippocampus/rag/en.json"


def main() -> None:
    if not UNITS:
        print("no unit results found under outputs/lifebench/hippo-lifebench-mimo-u*/rag/en.json")
        sys.exit(1)

    all_results = []
    total_correct = 0
    ingested = 0
    for path in UNITS:
        d = json.loads(path.read_text(encoding="utf-8"))
        results = d.get("queries") or d.get("results") or []
        all_results.extend(results)
        total_correct += int(d.get("correct") or 0)
        ingested += int(d.get("ingested_docs") or 0)
        unit = path.parts[-4]
        print(f"  {unit:32} {d.get('correct')}/{len(results)}")

    total = len(all_results)
    acc = total_correct / total if total else 0.0
    merged = {
        "dataset": "lifebench",
        "split": "en",
        "memory_provider": "hippocampus",
        "run_name": "hippo-lifebench-mimo-merged",
        "mode": "rag",
        "answer_llm": all_results and "openai:mimo-v2.6-flash" or None,
        "judge_llm": "openai:mimo-v2.6-flash",
        "total_queries": total,
        "correct": total_correct,
        "accuracy": acc,
        "ingested_docs": ingested,
        "results": all_results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"merged {len(UNITS)} units -> {OUT}")
    print(f"LifeBench/en TOTAL: {total_correct}/{total} = {acc:.4f}")


if __name__ == "__main__":
    main()
