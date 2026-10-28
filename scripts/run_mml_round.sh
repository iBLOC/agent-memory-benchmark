#!/usr/bin/env bash
# Run a LongMemEval-S QA round with MiniMax as both answer LLM and judge.
#
# Why this exists: the DeepSeek round cost ~15.5 CNY for the 500-question
# hippocampus row, and the balance runs out mid-run. A MiniMax coding-plan
# key (MINIMAX_API_KEY in .env) is covered by the subscription, so the same
# round has no marginal cost.
#
# Two MiniMax quirks are handled here:
#   1. It accepts response_format=json_schema but IGNORES it and answers in
#      prose, so the harness must be pinned to json_object
#      (OMB_JSON_MODE=json_object, added in commit 263b8c0).
#   2. It is a reasoning model: the chain of thought comes back in
#      reasoning_content and the answer stays in content, which the harness
#      already handles.
#
# Note: a different judge changes the scores. Rows scored by MiniMax are a
# separate round and must not be mixed with the DeepSeek-scored rows.
#
# Usage:
#   scripts/run_mml_round.sh hippocampus   # the memory provider to score
#   scripts/run_mml_round.sh bm25
set -u
cd "$(dirname "$0")/.."

PROVIDER="${1:-hippocampus}"
NAME="${2:-qa500-${PROVIDER}-mml}"

set -a && source .env && set +a
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=""
export OPENAI_API_KEY="$MINIMAX_API_KEY"
export OPENAI_BASE_URL="$MINIMAX_BASE_URL"
export OMB_JSON_MODE=json_object
export OMB_ANSWER_LLM=openai OMB_ANSWER_MODEL="${OMB_ANSWER_MODEL:-MiniMax-M3.1-Flash}"
export OMB_JUDGE_LLM=openai OMB_JUDGE_MODEL="${OMB_JUDGE_MODEL:-MiniMax-M3.1-Flash}"
export EMBEDDING_MODEL=all-MiniLM-L6-v2 LANCE_EMBEDDING_DIM=384
export LONGMEMEVAL_DATA_PATH="${LONGMEMEVAL_DATA_PATH:-$PWD/../amb_data/longmemeval_s_cleaned.json}"
export GEMINI_API_KEY=dummy
export HIPPOCAMPUS_TIMEOUT=600

exec .venv/bin/amb run --split s --dataset longmemeval \
    --memory "$PROVIDER" --mode rag --name "$NAME"
