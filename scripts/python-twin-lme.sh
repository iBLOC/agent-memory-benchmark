#!/usr/bin/env bash
# Python twin of hippocampus-bench's lme-recall: uses the AMB harness's own
# `hippocampus` provider (which carries the rich adapter-side rerank/filter),
# runs the first N LME-S questions, and saves the rag-mode run JSON. Pairs
# with hippocampus-bench's Rust runner on the same :8083 store for an
# apples-to-apples language comparison.
#
# Prereq: :8083 already has a server (any phase) with the LME-S docs ingested
# (hippocampus-bench lme-recall completes that step). We re-use --skip-ingestion.
set -u
cd "C:/Projects/iBLOC/agent-memory-benchmark" || exit 1
export PYTHONPATH=src
export HIPPOCAMPUS_URL=http://127.0.0.1:8083
export HIPPOCAMPUS_DOC_CHAR_LIMIT=0     # mirror Rust: no truncate, raw docs in
export HIPPOCAMPUS_TIMEOUT=600
export GEMINI_API_KEY=dummy
export OMB_ANSWER_LLM=gemini OMB_JUDGE_LLM=gemini
N="${1:-88}"   # default 88 to match the existing python-run result

echo "=== python twin lme-recall: ${N} queries on :8083 ==="
timeout 3600 uv run --no-sync amb run \
    --dataset longmemeval --split s \
    --memory hippocampus --mode rag \
    --query-limit "$N" \
    --skip-ingestion \
    --name "hippo-lme-python-twin" 2>&1 | tail -12

echo
echo "=== compare to Rust run ==="
python scripts/compare_python_recall.py \
    "outputs/longmemeval/hippo-lme-python-twin/rag/s.json" 10 2>&1 | tail -3
