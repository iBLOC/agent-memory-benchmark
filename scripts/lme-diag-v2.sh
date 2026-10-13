#!/usr/bin/env bash
# LME-S rag re-diagnostic after the raw_docs content fix (same 5 queries,
# same config as hippo-lme-mimo-diag for apples-to-apples comparison).
set -u
cd "C:/Projects/iBLOC/agent-memory-benchmark" || exit 1
export PYTHONPATH=src HIPPOCAMPUS_URL=http://127.0.0.1:8082 HIPPOCAMPUS_DOC_CHAR_LIMIT=6000

timeout 2400 uv run --no-sync amb run --dataset longmemeval --split s \
    --memory hippocampus --mode rag --query-limit 5 \
    --name hippo-lme-mimo-diag-v2 2>&1 | tail -18
echo "DIAG V2 DONE"
