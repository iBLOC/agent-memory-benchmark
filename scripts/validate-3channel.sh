#!/usr/bin/env bash
# Post-LifeBench-loop validation, judge-free edition (MiMo credits exhausted):
# rebuild release with the channel fixes, restart :8082, then run the FULL
# PrecisionMemBench retrieval pass (77 queries, zero LLM) and compare against
# the 77/77 baseline. Channel-health signals come from the server log:
#   - "graph expansion added N"  → associative expansion firing
#   - "hash-bag fallback"        → vector channel alive keyless
#   - "falling back to heuristic extractor" → graph nodes being created keyless
# If PMB regresses below 77, expansion is hurting ID-precision and needs a gate.
set -u
HIPPO_ROOT="C:/Projects/iBLOC/hippocampus"
LANCE_URI="file:///C:/Projects/iBLOC/amb_runs/lance_pmb_validate"
PORT=8082

echo "[validate] cargo build --release -j 2  $(date +%T)"
cd "$HIPPO_ROOT" || exit 1
cargo build --release -p hippocampus-server -j 2 || exit 1

echo "[validate] fresh store + restart :${PORT}"
taskkill //F //IM hippocampus-server.exe >/dev/null 2>&1 || true
sleep 3
mkdir -p "C:/Projects/iBLOC/amb_runs/lance_pmb_validate"
HIPPOCAMPUS_DB=lance LANCE_URI="file:///C:/Projects/iBLOC/amb_runs/lance_pmb_validate" \
    HIPPOCAMPUS_CONSOLIDATOR=off PORT="$PORT" \
    ./target/release/hippocampus-server.exe > "C:/Projects/iBLOC/amb_runs/lance_pmb_validate/server.log" 2>&1 &
for i in $(seq 1 60); do
    sleep 2
    curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break
done
curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1
echo " <- healthy"

echo "[validate] PMB single-turn retrieval, FULL 77 queries (keyless)"
cd "C:/Projects/iBLOC/agent-memory-benchmark" || exit 1
export PYTHONPATH=src
export HIPPOCAMPUS_URL="http://127.0.0.1:${PORT}"
export GEMINI_API_KEY=dummy OMB_ANSWER_LLM=gemini OMB_JUDGE_LLM=gemini

timeout 1800 uv run --no-sync amb run --dataset precisionmembench --split single-turn \
    --memory hippocampus --mode retrieval \
    --name pmb-3channel-validate 2>&1 | tail -16

echo "[validate] channel-health from server log:"
for sig in "graph expansion added" "hash-bag fallback" "falling back to heuristic extractor"; do
    n=$(grep -c "$sig" "C:/Projects/iBLOC/amb_runs/lance_pmb_validate/server.log" 2>/dev/null || echo 0)
    echo "  $sig: $n"
done
echo "VALIDATE DONE"
