#!/usr/bin/env bash
# Third pass: lightrag (+initialize_storages) and graphrag (no cache section).
set -u
cd "C:/Projects/iBLOC/agent-memory-benchmark" || exit 1
export PYTHONPATH=src GEMINI_API_KEY=dummy OMB_ANSWER_LLM=gemini OMB_JUDGE_LLM=gemini
export HIPPOCAMPUS_URL=http://127.0.0.1:8082

for provider in lightrag graphrag; do
  echo "=== $provider start $(date +%T) ==="
  timeout 3600 uv run --no-sync amb run --dataset precisionmembench --split single-turn \
      --memory "$provider" --mode retrieval --name "pmb-keyless-${provider}-v3" 2>&1 | tail -25
  echo "=== $provider done $(date +%T) rc=$? ==="
done
echo "FIX RERUNS V3 DONE"
