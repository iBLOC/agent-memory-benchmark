#!/usr/bin/env bash
# PMB for the qdrant hybrid-search provider (local Qwen3-0.6B dense + BM42
# sparse — keyless, but first run downloads ~1.5 GB of models).
set -u
cd "C:/Projects/iBLOC/agent-memory-benchmark" || exit 1
export PYTHONPATH=src GEMINI_API_KEY=dummy OMB_ANSWER_LLM=gemini OMB_JUDGE_LLM=gemini
export HIPPOCAMPUS_URL=http://127.0.0.1:8082

echo "=== qdrant start $(date +%T) ==="
timeout 7200 uv run --no-sync amb run --dataset precisionmembench --split single-turn \
    --memory qdrant --mode retrieval --name "pmb-keyless-qdrant" 2>&1 | tail -25
echo "=== qdrant done $(date +%T) rc=$? ==="
echo "QDRANT PMB DONE"
