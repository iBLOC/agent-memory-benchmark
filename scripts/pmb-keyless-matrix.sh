#!/usr/bin/env bash
# Zero-key PMB matrix: the only AMB dataset whose scoring is pure ID-set
# matching (no judge LLM). Runs the providers that work without any API
# key or external service. For agentmemorybenchmark.ai publication the
# judge-keyed datasets (longmemeval / locomo / ...) are a separate pass.
set -u
cd "C:/Projects/iBLOC/agent-memory-benchmark" || exit 1
export PYTHONPATH=src GEMINI_API_KEY=dummy OMB_ANSWER_LLM=gemini OMB_JUDGE_LLM=gemini
export HIPPOCAMPUS_URL=http://127.0.0.1:8082

for provider in hippocampus bm25 vanilla lightrag graphrag; do
  echo "=== $provider start $(date +%T) ==="
  timeout 3600 uv run --no-sync amb run --dataset precisionmembench --split single-turn \
      --memory "$provider" --mode retrieval --name "pmb-keyless-${provider}" 2>&1 | tail -20
  echo "=== $provider done $(date +%T) rc=$? ==="
done
echo "ALL PMB KEYLESS RUNS DONE"
