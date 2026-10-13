#!/usr/bin/env bash
# Dual-judge sensitivity experiment: re-judge the CACHED LME + LoCoMo answers
# (MiMo-generated) with deepseek-flash as judge, preserving the MiMo verdicts.
#
# Required environment (export before running — no credentials in this file;
# see docs/FAIRNESS_ANALYSIS.md §5 for the credential-handling protocol):
#   OPENAI_API_KEY    <- your DeepSeek key
#   OPENAI_BASE_URL   <- https://api.deepseek.com
#
# Cost: ~321 judge calls on short prompts at deepseek-flash pricing
# (CNY 1-2/M input, 4-8/M output) — well under CNY 1 total.
set -u
: "${OPENAI_API_KEY:?set OPENAI_API_KEY (DeepSeek key) and OPENAI_BASE_URL first}"
: "${OPENAI_BASE_URL:?set OPENAI_BASE_URL first}"
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=src
export HIPPOCAMPUS_URL=http://127.0.0.1:8082   # only initialize()'s health ping
export OMB_JUDGE_LLM=openai
export OMB_JUDGE_MODEL=deepseek-flash

rejudge() {
  local ds="$1" split="$2" src="$3" dst="$4"
  echo "=== $ds/$split: snapshot MiMo verdicts -> $dst ==="
  rm -rf "outputs/$ds/$dst"
  cp -r "outputs/$ds/$src" "outputs/$ds/$dst"
  echo "=== $ds/$split: re-judge with deepseek-flash ==="
  timeout 7200 uv run --no-sync amb run --dataset "$ds" --split "$split" \
      --memory hippocampus --mode rag --name "$dst" \
      --skip-ingestion --skip-answer 2>&1 | tail -12
}

# LME first (88 cached answers), then LoCoMo (233).
rejudge longmemeval s      hippo-lme-mimo-full      hippo-lme-dsjudge
rejudge locomo    locomo10 hippo-locomo-mimo-v2     hippo-locomo-dsjudge
echo "DUAL-JUDGE DONE"
