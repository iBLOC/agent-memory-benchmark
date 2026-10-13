#!/usr/bin/env bash
# Chain: wait for the release rebuild to land (exec_5638f900 builds it), then
# fresh store + relaunch the LifeBench 10-unit loop. Run AFTER the rebuild
# task completes — this script just does store-clean + loop-start.
set -u
cd "C:/Projects/iBLOC/agent-memory-benchmark" || exit 1
taskkill //F //IM hippocampus-server.exe >/dev/null 2>&1 || true
sleep 2
rm -rf "C:/Projects/iBLOC/amb_runs/lance_lifebench"
mkdir -p "C:/Projects/iBLOC/amb_runs/lance_lifebench"
bash scripts/lifebench-peruser.sh
