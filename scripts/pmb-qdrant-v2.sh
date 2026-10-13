#!/usr/bin/env bash
# qdrant PMB pass 2 — after adding the `filters` kwarg to async_retrieve.
# Ingest is fast this time (models already downloaded); previous pass died
# on the first query.
set -u
cd "C:/Projects/iBLOC/agent-memory-benchmark" || exit 1
export PYTHONPATH=src GEMINI_API_KEY=dummy OMB_ANSWER_LLM=gemini OMB_JUDGE_LLM=gemini
export HIPPOCAMPUS_URL=http://127.0.0.1:8082

echo "=== qdrant v2 start $(date +%T) ==="
timeout 7200 uv run --no-sync amb run --dataset precisionmembench --split single-turn \
    --memory qdrant --mode retrieval --name "pmb-keyless-qdrant-v2" 2>&1 | tail -25
echo "=== qdrant v2 done $(date +%T) rc=$? ==="
echo "QDRANT PMB V2 DONE"
