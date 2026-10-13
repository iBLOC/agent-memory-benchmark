#!/usr/bin/env bash
# LifeBench per-user loop with the documented Lance ops runbook between units:
#   stop server -> compact memory.lance -> restart server -> run one user.
# Rationale: the :8082 server accumulates every run's fragments in ONE
# memory.lance; per-query search degrades (>600s) and connect-time FTS goes
# stale after bulk ingest. Compacting + restarting between units restores
# both. Each unit = ~338 QA, results saved per run name; merged at the end.
set -u
cd "C:/Projects/iBLOC/agent-memory-benchmark" || exit 1

HIPPO_ROOT="C:/Projects/iBLOC/hippocampus"
mkdir -p "C:/Projects/iBLOC/amb_runs/lance_lifebench"
BIN="$HIPPO_ROOT/target/release/hippocampus-server.exe"
# Fresh dedicated store: the shared lance_1791549010 store accumulated
    # every run's fragments + delete churn today and OOM-aborted the server
    # mid-ingest (memory allocation of 5242880 bytes failed).
    LANCE_URI="file:///C:/Projects/iBLOC/amb_runs/lance_lifebench"
PORT=8082
SERVER_LOG="C:/Projects/iBLOC/amb_runs/lance_lifebench/server.log"
USERS=("Yu Xiaowei" "Yu Xiaowen" "Feng Haoran" "Ye Mingxuan" "Sun Yuwei" "Song Yajing" "Yin Hao" "Lu Mingqiang" "Lei Mingxuan" "Ma Xiulan")

export PYTHONPATH=src
export HIPPOCAMPUS_URL="http://127.0.0.1:${PORT}"
export HIPPOCAMPUS_DOC_CHAR_LIMIT=15000
export HIPPOCAMPUS_INGEST_CHAR_LIMIT=24000
export HIPPOCAMPUS_TIMEOUT=600
# Pin the MiMo judge explicitly: the background-task env carries an INVALID
# GOOGLE_API_KEY, and the default (gemini) judge then fails every query.
export OMB_JUDGE_LLM=openai
export OMB_JUDGE_MODEL=mimo-v2.6-flash
# Server-side LLM must fail FAST: with no timeout configured, each ingest's
# spawned episode-linker task hangs on an unreachable api.openai.com while
# holding the full document prompt — 365 of those concurrently OOM'd the
# server (memory allocation of 5242880 bytes failed). A dead local endpoint
# refuses instantly, so FallbackExtractor lands on the heuristic path.
export LLM_API_BASE=http://127.0.0.1:9
export LLM_API_KEY=dummy

stop_server() {
  taskkill //F //IM hippocampus-server.exe >/dev/null 2>&1 || true
  sleep 3
}

start_server() {
  ( cd "$HIPPO_ROOT" && \
    HIPPOCAMPUS_DB=lance LANCE_URI="$LANCE_URI" HIPPOCAMPUS_CONSOLIDATOR=off \
    PORT="$PORT" "$BIN" >> "$SERVER_LOG" 2>&1 & )
  for i in $(seq 1 60); do
    sleep 2
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      echo "[loop] server healthy (after ~$((i*2))s)"
      return 0
    fi
  done
  echo "[loop] SERVER FAILED TO START"; return 1
}

compact() {
  echo "[loop] compacting $(date +%T)..."
  # lancedb-python compaction — the cargo compact_lance release example
  # deadlocks on Windows linking; pylance does the same job in seconds.
  ( cd "C:/Projects/iBLOC/agent-memory-benchmark" && uv run --no-sync python -c "
import lancedb
db = lancedb.connect('C:/Projects/iBLOC/amb_runs/lance_1791549010')
for t in db.table_names():
    try:
        db.open_table(t).compact_files()
        print('  compacted', t, flush=True)
    except Exception as e:
        print('  skip', t, type(e).__name__, flush=True)
" ) || echo "[loop] compact failed (continuing)"
}

for i in "${!USERS[@]}"; do
  user="${USERS[$i]}"
  n=$((i+1))
  echo "=== unit $n/10: ${user} start $(date +%T) ==="
  stop_server
  compact
  start_server || exit 1
  timeout 28800 uv run --no-sync amb run --dataset lifebench --split en \
      --unit "$user" \
      --memory hippocampus --mode rag \
      --name "hippo-lifebench-mimo-u${n}" 2>&1 | tail -6
  echo "=== unit $n/10 done $(date +%T) rc=$? ==="
done

echo "ALL UNITS DONE $(date)"
