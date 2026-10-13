#!/usr/bin/env bash
# Official QA runs for the three remaining judge-scored datasets, in sequence:
#   locomo/locomo10 → lifebench/en → personamem/32k
# MiMo (mimo-v2.6-flash) as answer + judge, per .env.
#
# Config notes:
#   DOC_CHAR_LIMIT=15000 — LoCoMo/PersonaMem sessions are very long; full docs
#   would push prompts past MiMo's context window. 15k/doc keeps prompts
#   ≲150k chars. (LME's 85.2% run used 0/unlimited; its docs are smaller.)
#   HIPPOCAMPUS_TIMEOUT=120 — one /api/search on the big 88q store exceeded
#   60s during the LME full run; double the budget.
set -u
cd "C:/Projects/iBLOC/agent-memory-benchmark" || exit 1
export PYTHONPATH=src
export HIPPOCAMPUS_URL=http://127.0.0.1:8082
export HIPPOCAMPUS_DOC_CHAR_LIMIT=15000
export HIPPOCAMPUS_TIMEOUT=120

run_one() {
  local ds="$1" split="$2"
  echo "=== $ds/$split start $(date +%T) ==="
  timeout 14400 uv run --no-sync amb run --dataset "$ds" --split "$split" \
      --memory hippocampus --mode rag \
      --name "hippo-${ds}-mimo" 2>&1 | tail -16
  echo "=== $ds/$split done $(date +%T) rc=$? ==="
}

run_one locomo locomo10
run_one lifebench en
run_one personamem 32k
echo "ALL THREE DATASETS DONE"
