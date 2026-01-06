#!/bin/bash
# OCR all textbooks using marker-pdf and ingest to Polymath

OCR_OUTPUT="/home/user/work/polymax/ocr_extracted"
STAGING="/home/user/work/polymax/ingest_staging"

mkdir -p "$OCR_OUTPUT"

echo "Starting OCR for all PDFs in $STAGING..."

for pdf in "$STAGING"/*.pdf; do
    if [ -f "$pdf" ]; then
        name=$(basename "$pdf" .pdf)
        echo "OCRing: $name"

        # Run marker_single
        marker_single "$pdf" --output_format markdown --output_dir "$OCR_OUTPUT/" 2>&1

        echo "Done: $name"
    fi
done

echo "All OCR complete. Output in $OCR_OUTPUT"

# List results
ls -la "$OCR_OUTPUT"/*/
