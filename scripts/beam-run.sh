#!/usr/bin/env bash
# BEAM/100k official QA run against the pre-ingested :8083 store.
# 170 docs (100k = context-token scale), 400 queries, MiMo answer+judge.
set -u
cd "C:/Projects/iBLOC/agent-memory-benchmark" || exit 1
export PYTHONPATH=src
export HIPPOCAMPUS_URL=http://127.0.0.1:8083
export HIPPOCAMPUS_DOC_CHAR_LIMIT=15000
export HIPPOCAMPUS_TIMEOUT=600

echo "=== beam/100k start $(date +%T) ==="
timeout 43200 uv run --no-sync amb run --dataset beam --split 100k \
    --memory hippocampus --mode rag --name hippo-beam-mimo \
    --skip-ingestion 2>&1 | tail -18
echo "=== beam/100k done $(date +%T) rc=$? ==="
echo "BEAM DONE"
