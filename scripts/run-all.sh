#!/usr/bin/env bash
# Run the AMB cross-provider retrieval matrix against a fresh hippocampus-server.
#
# Defaults match the public-leaderboard recipe:
#   7 datasets × { hippocampus, mem0, cognee, hindsight, qdrant,
#                   letta, graphrag, lightrag }
#   + bm25 + vanilla baselines = 8+2 providers per dataset
#   Mode = retrieval (no LLM-as-judge; pure hit@k)
#
# Selective runs are easy: pass dataset / memory lists to override:
#   ./scripts/run-all.sh --datasets "longmemeval,locomo" --memories "hippocampus,mem0"
#
# Each dataset run is one amb invocation against a fresh server (so FTS
# state never leaks across datasets). provider runs reuse one prepared store.
#
# Wall clock for the full matrix is ~3-5 days on a 16-CPU release server;
# use the --limit flag (forwarded to amb) to cap query count during smoke.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$PWD"
OUT="$ROOT/outputs"
LOG_DIR="$ROOT/.runs"

# Defaults — full public matrix. Trim with --datasets / --memories.
DATASETS=( "beam:100k" "lifebench:en" "locomo:locomo10" "longmemeval:s" "personamem:32k" "precisionmembench:single-turn" "sdebench:boltons" )
MEMORIES=( "hippocampus" "mem0" "cognee" "hindsight" "qdrant" "letta" "graphrag" "lightrag" "bm25" "vanilla" )
EXTRA_FLAGS=()
HIPPOCAMPUS_PORT_DEFAULT=8081
PARALLEL=1
DRY_RUN=0

usage() {
    cat <<'USAGE'
Usage: scripts/run-all.sh [options]

Options:
  --datasets "ds1[:unit1],ds2[:unit2]"   Restrict datasets (default: 7)
  --memories "m1,m2,..."                 Restrict providers (default: 8+2)
  --parallel N                           Concurrent (dataset,provider) pairs (default 1)
  --port P                               Hippocampus PORT (default 8081)
  --limit N                              Forward --query-limit to amb (smoke)
  --dry-run                              Print the matrix; do not execute
  -h, --help                             Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --datasets)    shift; IFS=',' read -ra DATASETS <<<"$1";;
        --memories)    shift; IFS=',' read -ra MEMORIES <<<"$1";;
        --parallel)    shift; PARALLEL="$1";;
        --port)        shift; HIPPOCAMPUS_PORT_DEFAULT="$1";;
        --limit)       shift; EXTRA_FLAGS+=("--query-limit" "$1");;
        --dry-run)     DRY_RUN=1;;
        -h|--help)     usage; exit 0;;
        *)             echo "[run-all] unknown arg: $1" >&2; exit 2;;
    esac
    shift
done

mkdir -p "$OUT" "$LOG_DIR"

echo "[run-all] $(date +%FT%T) — $(echo "${DATASETS[@]}" | wc -w) datasets × $(echo "${MEMORIES[@]}" | wc -w) providers"
echo "[run-all] parallel=$PARALLEL logs=$LOG_DIR outputs=$OUT"
echo

# Print the matrix once before running
total=0
for ds in "${DATASETS[@]}"; do
    for mem in "${MEMORIES[@]}"; do
        printf '%s\n' "  - ${ds} :: ${mem}"
        total=$(( total + 1 ))
    done
done | head -10
echo "  ... (${total} runs total)"
echo

if [[ "$DRY_RUN" == "1" ]]; then
    echo "[run-all] --dry-run set; exiting."
    exit 0
fi

# Pre-flight: hippocampus server reachable when needed
needs_hippo=0
for mem in "${MEMORIES[@]}"; do [[ "$mem" == "hippocampus" ]] && needs_hippo=1; done
if [[ "$needs_hippo" == "1" ]]; then
    if ! curl -fsS "http://127.0.0.1:${HIPPOCAMPUS_PORT_DEFAULT}/health" >/dev/null 2>&1; then
        echo "[run-all] ERROR: hippocampus server not reachable on :${HIPPOCAMPUS_PORT_DEFAULT}"
        echo "  boot one first: ../hippocampus/scripts/reproduce-amb.sh <dataset>"
        exit 1
    fi
    export HIPPOCAMPUS_URL="http://127.0.0.1:${HIPPOCAMPUS_PORT_DEFAULT}"
fi

export ROOT OUT LOG_DIR HIPPOCAMPUS_PORT_DEFAULT EXTRA_FLAGS

in_flight=0
for ds_full in "${DATASETS[@]}"; do
    ds="${ds_full%%:*}"
    unit="${ds_full#*:}"
    [[ "$unit" == "$ds_full" ]] && unit=""

    for provider in "${MEMORIES[@]}"; do
        # sdebench is coding mode
        if [[ "$ds" == "sdebench" ]]; then
            amb_args=( --mode coding --skip-answer )
        else
            amb_args=( --mode retrieval )
        fi
        unit_arg=()
        [[ -n "$unit" ]] && unit_arg+=( --unit "$unit" )

        tag="$(date +%Y%m%d-%H%M%S)-${ds}-${provider}"
        log="$LOG_DIR/${tag}.log"

        (
            set +e
            if PYTHONPATH=src uv run amb run \
                --dataset "$ds" "${unit_arg[@]}" \
                --memory "$provider" \
                "${amb_args[@]}" \
                --name "hippocampus-${tag}" \
                --output-dir "$OUT" \
                "${EXTRA_FLAGS[@]}" \
                >"$log" 2>&1; then
                echo "OK  ${tag}"
            else
                echo "ERR ${tag} (log=$log)"
            fi
        ) &
        in_flight=$(( in_flight + 1 ))

        if [[ "$in_flight" -ge "$PARALLEL" ]]; then
            wait -n
            in_flight=$(( in_flight - 1 ))
        fi
    done
done

wait
echo
echo "[run-all] all done. outputs=$OUT logs=$LOG_DIR"
