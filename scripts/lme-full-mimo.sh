#!/usr/bin/env bash
# Full LME-S 31-question official QA run — content fix + no truncation,
# MiMo (mimo-v2.6-flash) as answer + judge. Expect ~40-70 min.
set -u
cd "C:/Projects/iBLOC/agent-memory-benchmark" || exit 1
export PYTHONPATH=src HIPPOCAMPUS_URL=http://127.0.0.1:8082 HIPPOCAMPUS_DOC_CHAR_LIMIT=0

timeout 7200 uv run --no-sync amb run --dataset longmemeval --split s \
    --memory hippocampus --mode rag \
    --name hippo-lme-mimo-full 2>&1 | tail -18
echo "LME FULL DONE"
