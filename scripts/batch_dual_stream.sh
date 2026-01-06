#!/bin/bash
# Batch dual-stream ingestion for Tier 1/2 priority repos

REPOS_DIR="/home/user/work/polymax/data/github_repos"
DUAL_STREAM="/home/user/work/polymax/lib/dual_stream_ingest.py"
LOG_DIR="/home/user/work/polymax/logs"

# Priority repos (Tier 1: Vanderbilt, Tier 2: Virtual Sequencing)
REPOS=(
    "CONCH"           # Tier 2: Foundation model with contrastive learning
    "HIPT"            # Tier 2: Hierarchical vision transformer
    "CLAM"            # Tier 2: Attention-based MIL
    "Map3D"           # Tier 1: Vanderbilt 3D reconstruction
    "SLANTbrainSeg"   # Tier 1: Vanderbilt brain segmentation
    "PreQual"         # Tier 1: Vanderbilt preprocessing (diffusion, denoising)
)

echo "======================================================================"
echo "BATCH DUAL-STREAM INGESTION"
echo "======================================================================"
echo "Repos to process: ${#REPOS[@]}"
echo "Start time: $(date)"
echo ""

TOTAL=0
SUCCESS=0
FAILED=0

for repo in "${REPOS[@]}"; do
    echo ""
    echo "======================================================================"
    echo "Processing: $repo"
    echo "======================================================================"

    REPO_PATH="$REPOS_DIR/$repo"
    LOG_FILE="$LOG_DIR/dual_stream_${repo}_$(date +%Y%m%d_%H%M).log"

    if [ ! -d "$REPO_PATH" ]; then
        echo "ERROR: Repo not found at $REPO_PATH"
        FAILED=$((FAILED + 1))
        continue
    fi

    TOTAL=$((TOTAL + 1))

    # Run dual-stream ingestion
    python3 "$DUAL_STREAM" "$REPO_PATH" \
        --repo-name "${repo}-dual" \
        --max-files 20 \
        2>&1 | tee "$LOG_FILE"

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✓ SUCCESS: $repo ingested"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "✗ FAILED: $repo ingestion failed"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "======================================================================"
echo "BATCH INGESTION COMPLETE"
echo "======================================================================"
echo "Total repos: $TOTAL"
echo "Success: $SUCCESS"
echo "Failed: $FAILED"
echo "End time: $(date)"
echo ""
echo "Logs saved to: $LOG_DIR/dual_stream_*.log"
