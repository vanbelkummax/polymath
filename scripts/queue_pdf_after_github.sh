#!/bin/bash
# Queue PDF ingestion after GitHub repo ingestion completes
# Created: 2026-01-06

GITHUB_PID=19181
PDF_DIR="/home/user/work/polymax/ingest_staging/newpapers_20260106"
LOG_FILE="/home/user/work/polymax/logs/pdf_ingest_$(date +%Y%m%d_%H%M%S).log"

echo "Waiting for GitHub ingestion (PID $GITHUB_PID) to complete..."
echo "PDFs queued: $(ls $PDF_DIR/*.pdf 2>/dev/null | wc -l)"

# Wait for GitHub ingestion to finish
while kill -0 $GITHUB_PID 2>/dev/null; do
    echo "$(date): GitHub ingestion still running..."
    sleep 60
done

echo "$(date): GitHub ingestion complete. Starting PDF ingestion..."

cd /home/user/work/polymax

# Run unified ingestion with move flag
python3 lib/unified_ingest.py "$PDF_DIR" --move 2>&1 | tee "$LOG_FILE"

echo "$(date): PDF ingestion complete. Log: $LOG_FILE"
