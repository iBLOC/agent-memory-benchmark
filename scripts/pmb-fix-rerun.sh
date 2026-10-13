#!/usr/bin/env bash
# Re-run the two PMB providers that crashed on first pass (lightrag
# EmbeddingFunc wrapper, graphrag Windows path validation) after fixes.
set -u
cd "C:/Projects/iBLOC/agent-memory-benchmark" || exit 1
export PYTHONPATH=src GEMINI_API_KEY=dummy OMB_ANSWER_LLM=gemini OMB_JUDGE_LLM=gemini
export HIPPOCAMPUS_URL=http://127.0.0.1:8082

for provider in lightrag graphrag; do
  echo "=== $provider start $(date +%T) ==="
  timeout 3600 uv run --no-sync amb run --dataset precisionmembench --split single-turn \
      --memory "$provider" --mode retrieval --name "pmb-keyless-${provider}-v2" 2>&1 | tail -20
  echo "=== $provider done $(date +%T) rc=$? ==="
done
echo "FIX RERUNS DONE"
