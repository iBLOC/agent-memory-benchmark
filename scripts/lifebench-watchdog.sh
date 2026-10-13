#!/usr/bin/env bash
# Waits for the compact_lance release build to finish (background cargo holds
# the target lock), then launches the LifeBench per-user loop immediately.
set -u
EXE="C:/Projects/iBLOC/hippocampus/target/release/examples/compact_lance.exe"
echo "[watch] waiting for compact_lance.exe $(date +%T)"
for i in $(seq 1 240); do   # up to 120 min
  if [[ -f "$EXE" ]]; then
    # give cargo a few seconds to finish writing + release the lock
    sleep 15
    echo "[watch] build complete $(date +%T) — launching per-user loop"
    exec bash "C:/Projects/iBLOC/agent-memory-benchmark/scripts/lifebench-peruser.sh"
  fi
  # if cargo died without producing the exe, bail out
  if ! tasklist //FI "IMAGENAME eq cargo.exe" 2>/dev/null | grep -q cargo.exe; then
    sleep 30
    if [[ ! -f "$EXE" ]]; then
      echo "[watch] ERROR: cargo exited without producing compact_lance.exe"
      exit 1
    fi
  fi
  sleep 30
done
echo "[watch] timed out after 120 min"
exit 1
