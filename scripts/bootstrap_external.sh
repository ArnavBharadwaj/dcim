#!/usr/bin/env bash
# Fetch the pinned SustainDC checkout that the Phase 0 findings were measured against.
# external/ is gitignored, so this script is the record of which commit we read.
set -euo pipefail

SUSTAINDC_SHA="a92b4755aca560e34a98d14028dda629eb968482"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO_ROOT/external/dc-rl"

if [ -d "$DEST/.git" ]; then
  echo "SustainDC already present: $(git -C "$DEST" rev-parse HEAD)"
  exit 0
fi

mkdir -p "$REPO_ROOT/external"
git clone https://github.com/HewlettPackard/dc-rl.git "$DEST"
git -C "$DEST" checkout "$SUSTAINDC_SHA"
echo "SustainDC pinned at $(git -C "$DEST" rev-parse HEAD)"
