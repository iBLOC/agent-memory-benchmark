#!/usr/bin/env bash
# Re-judge the three QA runs with the OFFICIAL default judge (real Gemini via
# system GOOGLE_API_KEY) using cached answers: --skip-ingestion --skip-answer
# means zero server calls and zero answer-LLM calls — verdicts only.
# Shell-exported OMB_JUDGE_LLM=gemini overrides the .env MiMo judge
# (python-dotenv does not override pre-set env vars).
set -u
cd "C:/Projects/iBLOC/agent-memory-benchmark" || exit 1
export PYTHONPATH=src
export OMB_JUDGE_LLM=gemini
unset OMB_JUDGE_MODEL OMB_ANSWER_LLM OMB_ANSWER_MODEL 2>/dev/null || true

rejudge() {
  local ds="$1" split="$2" name="$3"
  echo "=== rejudge $ds/$split ($name) start $(date +%T) ==="
  timeout 7200 uv run --no-sync amb run --dataset "$ds" --split "$split" \
      --memory hippocampus --mode rag --name "$name" \
      --skip-ingestion --skip-answer 2>&1 | tail -14
  echo "=== rejudge $ds/$split done $(date +%T) rc=$? ==="
}

rejudge longmemeval s hippo-lme-mimo-full
rejudge locomo locomo10 hippo-locomo-mimo-v2
rejudge personamem 32k hippo-personamem-mimo
echo "ALL REJUDGES DONE"
