#!/usr/bin/env bash
# Re-judge BEAM with MiMo (cached answers, verdicts only). Same judge protocol
# as longmemeval/locomo/personamem rows. NOTE: do NOT use gemini judge — no
# real Google key exists; GEMINI_API_KEY=dummy in .env silently zeroes every
# rubric score.
set -u
cd "C:/Projects/iBLOC/agent-memory-benchmark" || exit 1
export PYTHONPATH=src
export HIPPOCAMPUS_URL=http://127.0.0.1:8082  # only initialize()'s health ping hits this; :8083 may be cycled by the LifeBench loop
export OMB_JUDGE_LLM=openai
export OMB_JUDGE_MODEL=mimo-v2.6-flash

echo "=== beam rejudge (mimo) start $(date +%T) ==="
timeout 7200 uv run --no-sync amb run --dataset beam --split 100k \
    --memory hippocampus --mode rag --name hippo-beam-mimo \
    --skip-ingestion --skip-answer 2>&1 | tail -14
echo "=== beam rejudge done $(date +%T) rc=$? ==="
echo "BEAM REJUDGE DONE"
