#!/usr/bin/env bash
# Fetch the Alibaba PAI GPU cluster trace sample used for workload replay.
# 100K real jobs over ~290 hours, bundled in the official alibaba/clusterdata repo.
# data/ is gitignored, so this script is the record of where the trace came from.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO_ROOT/data/raw/pai_job_duration_estimate_100K.csv"
[ -f "$DEST" ] && { echo "trace already present: $DEST"; exit 0; }
TMP="$(mktemp -d)"
git clone --depth 1 https://github.com/alibaba/clusterdata.git "$TMP/clusterdata"
mkdir -p "$REPO_ROOT/data/raw"
cp "$TMP/clusterdata/cluster-trace-gpu-v2020/simulator/traces/pai/pai_job_duration_estimate_100K.csv" "$DEST"
rm -rf "$TMP"
echo "wrote $DEST"
