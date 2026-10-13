#!/usr/bin/env bash
# Publish AMB results to Vercel Blob (`agentmemorybenchmark.ai` CDN).
#
# Flow per the AMB CLI help (cli.py docstring at line 136):
#   1. amb publish-results --push --all    gzips outputs/ + updates results-manifest.json
#   2. git add outputs/ results-manifest.json ...
#   3. git commit + git push
# Vercel is wired to the main branch of iBLOC/agent-memory-benchmark and
# auto-deploys the site on push.
#
# Subsequent re-runs are no-ops because amb's publish step is content-hash
# dedup'd (cli.py:215-237), so it's safe to re-run as fresh providers land.
#
# Required env:
#   BLOB_READ_WRITE_TOKEN   Vercel Blob token with rw scope
# Optional env:
#   AMB_GIT_REMOTE          (default: origin)
#   AMB_GIT_BRANCH          (default: main)

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ -z "${BLOB_READ_WRITE_TOKEN:-}" ]]; then
    echo "[publish-all] WARNING: BLOB_READ_WRITE_TOKEN unset — `amb publish-results` will write the manifest locally only."
    echo "  set BLOB_READ_WRITE_TOKEN in env to push to the live site."
fi

REMOTE="${AMB_GIT_REMOTE:-origin}"
BRANCH="${AMB_GIT_BRANCH:-main}"

echo "[publish-all] amb publish-results --push --all"
PYTHONPATH=src uv run amb publish-results --push --all

echo
echo "[publish-all] git add outputs/ results-manifest.json catalog.json blob-manifest.json external_results.json"
git add outputs/ results-manifest.json catalog.json blob-manifest.json external_results.json 2>/dev/null || true
git add -u

if git diff --cached --quiet; then
    echo "[publish-all] no staged changes — nothing to commit (probably a checksum-dedup hit; amb already published an identical manifest)."
    exit 0
fi

echo "[publish-all] git commit -m \"chore(amb): refresh retrieval results across 7 datasets\""
git commit -m "chore(amb): refresh retrieval results across 7 datasets"

echo "[publish-all] git push ${REMOTE} ${BRANCH}  (Vercel auto-deploys)"
git push "${REMOTE}" "${BRANCH}"

echo
echo "[publish-all] done. site will refresh within a few minutes."
echo "  → https://agentmemorybenchmark.ai"
echo "  → API: https://agentmemorybenchmark.ai/api/results?provider=<name>"
