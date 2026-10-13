"""Extract per-query top-k retrieved uris from a Python-harness LME run JSON
and score hit@k — the Python-side twin of hippocampus-bench's lme-recall.

Run:  python compare_python_recall.py <run_json> [topk]
e.g.  python compare_python_recall.py outputs/longmemeval/hippo-lme-mimo-full/rag/s.json 10
"""
import json
import sys
from pathlib import Path

run_arg = sys.argv[1] if len(sys.argv) > 1 else ""
topk = int(sys.argv[2]) if len(sys.argv) > 2 else 10

run_path = Path(run_arg).resolve()
if ".." in run_path.parts:
    sys.exit("path must not contain ..")
if not run_path.is_file():
    sys.exit(f"not a file: {run_path}")

# output lands next to the input, same stem + metric suffix
out_path = run_path.with_name(f"{run_path.stem}.python-recall-k{topk}.json")

d = json.loads(run_path.read_text(encoding="utf-8"))
results = d.get("queries") or d.get("results") or []

hit = 0
rr = 0.0
n = 0
per = []
for r in results:
    docs = (r.get("raw_response") or {}).get("documents") or []
    uris = [doc.get("uri") or doc.get("id") for doc in docs][:topk]
    gold = r.get("gold_ids") or []
    rank = next((i + 1 for i, u in enumerate(uris) if u in gold), None)
    n += 1
    if rank:
        hit += 1
        rr += 1 / rank
    per.append({"query_id": r.get("query_id"), "hit": bool(rank), "first_gold_rank": rank})

acc = hit / n if n else 0.0
mrr = rr / n if n else 0.0
print(f"python pipeline: hit@{topk} {hit}/{n} = {acc:.4f} | MRR {mrr:.4f}")
out_path.write_text(
    json.dumps(
        {"benchmark": "lme-recall", "runner": "python-harness", "topk": topk,
         "total_queries": n, "hit": hit, "hit_at_k": acc, "mrr": mrr, "per_query": per},
        ensure_ascii=False, indent=2,
    ),
    encoding="utf-8",
)
print(f"report -> {out_path}")
