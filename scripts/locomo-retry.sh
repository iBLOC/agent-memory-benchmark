#!/usr/bin/env bash
# LoCoMo retry: /api/search on locomo's big per-tenant stores exceeds 120s
# server-side; give the client a 600s read budget. Everything else identical
# to the qa-three-datasets.sh pass.
set -u
cd "C:/Projects/iBLOC/agent-memory-benchmark" || exit 1
export PYTHONPATH=src
export HIPPOCAMPUS_URL=http://127.0.0.1:8082
export HIPPOCAMPUS_DOC_CHAR_LIMIT=15000
export HIPPOCAMPUS_TIMEOUT=600

echo "=== locomo retry start $(date +%T) ==="
timeout 14400 uv run --no-sync amb run --dataset locomo --split locomo10 \
    --memory hippocampus --mode rag \
    --name "hippo-locomo-mimo-v2" 2>&1 | tail -16
echo "=== locomo retry done $(date +%T) rc=$? ==="
echo "LOCOMO RETRY DONE"
