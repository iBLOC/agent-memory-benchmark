#!/usr/bin/env bash
# LifeBench FULL run: 10 users x ~338 QA = 3380 queries, MiMo answer+judge.
# ~40-48h wall clock at concurrency 4; saves incrementally after each user
# (10 checkpoints), so an interrupted run can be assessed/resumed.
# Data: HF v2.0 en (cached at .datasets/lifebench/our_en.json).
set -u
cd "C:/Projects/iBLOC/agent-memory-benchmark" || exit 1
export PYTHONPATH=src
export HIPPOCAMPUS_URL=http://127.0.0.1:8082
export HIPPOCAMPUS_DOC_CHAR_LIMIT=15000
export HIPPOCAMPUS_TIMEOUT=600

echo "=== lifebench FULL start $(date) ==="
timeout 172800 uv run --no-sync amb run --dataset lifebench --split en \
    --memory hippocampus --mode rag \
    --name hippo-lifebench-mimo-full 2>&1 | tail -20
echo "=== lifebench FULL done $(date) rc=$? ==="
echo "LIFEBENCH FULL DONE"
