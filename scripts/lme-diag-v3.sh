#!/usr/bin/env bash
# LME-S rag diag v3: HIPPOCAMPUS_DOC_CHAR_LIMIT=0 (no truncation) to measure
# the ceiling now that raw_docs carries content. If MiMo chokes on very long
# prompts we fall back to a higher-but-finite cap.
set -u
cd "C:/Projects/iBLOC/agent-memory-benchmark" || exit 1
export PYTHONPATH=src HIPPOCAMPUS_URL=http://127.0.0.1:8082 HIPPOCAMPUS_DOC_CHAR_LIMIT=0

timeout 2400 uv run --no-sync amb run --dataset longmemeval --split s \
    --memory hippocampus --mode rag --query-limit 5 \
    --name hippo-lme-mimo-diag-v3 2>&1 | tail -18
echo "DIAG V3 DONE"
