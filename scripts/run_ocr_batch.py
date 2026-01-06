#!/usr/bin/env python3
"""Run OCR on all PDFs in staging directory."""

import os
import subprocess
from pathlib import Path

STAGING = "/home/user/work/polymax/ingest_staging"
OUTPUT = "/home/user/work/polymax/ocr_output"

os.makedirs(OUTPUT, exist_ok=True)

for pdf in Path(STAGING).glob("*.pdf"):
    name = pdf.stem
    if (Path(OUTPUT) / name).exists():
        print(f"Skipping {name} (already done)")
        continue

    print(f"OCRing: {name}")
    result = subprocess.run(
        ["marker_single", str(pdf), "--output_format", "markdown", "--output_dir", OUTPUT],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  Error: {result.stderr[:200]}")
    else:
        print(f"  Done: {name}")

print(f"\nAll OCR complete. Check {OUTPUT}/")
