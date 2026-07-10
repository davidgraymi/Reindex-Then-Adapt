#!/bin/bash
# Sync the preprocessed .pt encodings (and the small data files training
# needs) from this machine to the Mac mini, skipping the multi-GB JSONL.
# Run from anywhere; paths are derived from this script's location.
#
# Usage: ./scripts/sync_pt_to_mini.sh [user@host]   (default: serveradmin@mac-mini.local)

set -euo pipefail

REMOTE="${1:-mini-admin}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$REMOTE:~/git/Reindex-Then-Adapt"

# 1. The 7 preprocessed encoding tensors (inspired/redial: train+valid+test,
#    wikipedia: train only) — this is what preprocess.py would have produced.
rsync -av --progress \
    --include '*/' \
    --include 'output-encoding.pt' \
    --exclude '*' \
    "$REPO_DIR/reindex_step/data/" \
    "$DEST/reindex_step/data/"

# 2. Small shared files train.py/test.py also load (safe to re-send).
#    NOTE: run rsync from inside the repo with relative paths — macOS ships
#    openrsync, which ignores the "/./" marker of --relative and would
#    otherwise recreate the full absolute source path on the remote.
cd "$REPO_DIR"
rsync -av --progress --relative \
    data/llm_embedding.pt \
    data/item_entity_name_to_item_indices.json \
    data/inspired/id2name.json \
    data/redial/id2name.json \
    data/redditv1.5/id2name.json \
    data/wikipedia/id2name.json \
    "$DEST/"

# 3. Remove the stray tree a previous run of this script created via the
#    openrsync bug above, then verify: expect 7 encoding tensors
#    (1.5M–192M each) plus the 500M llm_embedding.pt.
echo
echo "--- files now on $REMOTE ---"
ssh "$REMOTE" 'rm -rf ~/git/Reindex-Then-Adapt/Users; ls -lh ~/git/Reindex-Then-Adapt/reindex_step/data/*/*/output-encoding.pt ~/git/Reindex-Then-Adapt/data/llm_embedding.pt'
