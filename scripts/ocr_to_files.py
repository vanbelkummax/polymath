#!/usr/bin/env python3
"""
OCR to Files - Extract text from scanned PDFs, save to disk.
Does NOT touch ChromaDB - just pure OCR work.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

OUTPUT_DIR = Path("/home/user/work/polymax/ocr_extracted")
PROGRESS_FILE = OUTPUT_DIR / "progress.json"


def needs_ocr(pdf_path: Path, min_chars: int = 500) -> bool:
    """Check if PDF needs OCR (scanned/image-based)."""
    import fitz
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc[:5]:  # Check first 5 pages
            text += page.get_text()
        doc.close()
        return len(text.strip()) < min_chars
    except:
        return True


def ocr_pdf(pdf_path: Path) -> tuple:
    """OCR a PDF and return (text, success)."""
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered

    try:
        converter = PdfConverter(artifact_dict=create_model_dict())
        rendered = converter(str(pdf_path))
        text = text_from_rendered(rendered)
        return text, True
    except Exception as e:
        print(f"    OCR error: {e}")
        return "", False


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"completed": [], "failed": []}


def save_progress(progress: dict):
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def main():
    print("=" * 60)
    print("OCR TO FILES (ChromaDB-free)")
    print("=" * 60)
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Find PDFs needing OCR
    pdf_dir = Path("/home/user/work/polymax/ingest_all_waiting")
    all_pdfs = list(pdf_dir.glob("*.pdf"))

    print(f"Scanning {len(all_pdfs)} PDFs for OCR candidates...")

    ocr_needed = []
    for pdf in all_pdfs:
        if needs_ocr(pdf):
            ocr_needed.append(pdf)

    print(f"Found {len(ocr_needed)} PDFs needing OCR\n")

    # Load progress
    progress = load_progress()
    completed = set(progress.get("completed", []))

    # Filter already done
    remaining = [p for p in ocr_needed if p.name not in completed]
    print(f"Remaining after resume: {len(remaining)}")

    if not remaining:
        print("\nAll OCR complete!")
        return

    # Load Marker models once
    print("\nLoading Marker models (one-time)...")
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered

    models = create_model_dict()
    converter = PdfConverter(artifact_dict=models)
    print("Models loaded!\n")

    print("-" * 60)

    for i, pdf in enumerate(remaining, 1):
        print(f"[{i}/{len(remaining)}] OCR: {pdf.name[:50]}...")

        try:
            rendered = converter(str(pdf))
            text = text_from_rendered(rendered)

            # Save to file
            out_file = OUTPUT_DIR / f"{pdf.stem}.txt"
            out_file.write_text(text)

            # Save metadata
            meta_file = OUTPUT_DIR / f"{pdf.stem}.json"
            meta_file.write_text(json.dumps({
                "source": pdf.name,
                "chars": len(text),
                "ocr_date": datetime.now().isoformat()
            }))

            print(f"    ✓ Saved {len(text):,} chars")

            progress["completed"].append(pdf.name)
            save_progress(progress)

        except Exception as e:
            print(f"    ✗ Failed: {e}")
            progress["failed"].append(pdf.name)
            save_progress(progress)

    print("\n" + "=" * 60)
    print("OCR EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Files created: {len(list(OUTPUT_DIR.glob('*.txt')))}")
    print(f"\nRun ingest_ocr_results.py to load into ChromaDB")


if __name__ == "__main__":
    main()
