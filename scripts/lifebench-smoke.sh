#!/usr/bin/env bash
# LifeBench smoke on the v2.0 HF data (cached at .datasets/lifebench/our_en.json):
# 1 user x 365 sessions ingest + 5 QA, MiMo answer+judge.
set -u
cd "C:/Projects/iBLOC/agent-memory-benchmark" || exit 1
export PYTHONPATH=src
export HIPPOCAMPUS_URL=http://127.0.0.1:8082
export HIPPOCAMPUS_DOC_CHAR_LIMIT=15000
export HIPPOCAMPUS_TIMEOUT=600

timeout 3600 uv run --no-sync amb run --dataset lifebench --split en \
    --memory hippocampus --mode rag --query-limit 5 \
    --name hippo-lifebench-mimo-smoke 2>&1 | tail -16
echo "LIFEBENCH SMOKE DONE"
