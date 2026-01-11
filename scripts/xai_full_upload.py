#!/usr/bin/env python3
"""
Upload ALL PDFs to xAI Collections.

Creates a fresh collection and uploads all 3,158 PDFs from the archive.
Runs with progress tracking and resume capability.

Usage:
    python3 scripts/xai_full_upload.py

    # Resume from checkpoint
    python3 scripts/xai_full_upload.py --resume
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from datetime import datetime

# Load .env
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

try:
    from xai_sdk import Client
except ImportError:
    print("❌ xai-sdk not installed. Run: pip install xai-sdk")
    sys.exit(1)

# Configuration
PDF_DIR = Path("/scratch/polymath_archive/pdfs")
COLLECTION_NAME = "polymath_full"
CHECKPOINT_FILE = Path("/home/user/work/polymax/logs/xai_full_upload_checkpoint.json")
LOG_FILE = Path("/home/user/work/polymax/logs/xai_full_upload.log")

# Ensure log directory exists
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def log(msg: str):
    """Log to both console and file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def get_client() -> Client:
    """Initialize xAI client."""
    api_key = os.getenv("XAI_API_KEY")
    mgmt_key = os.getenv("XAI_MANAGEMENT_API_KEY")

    if not api_key or not mgmt_key:
        log("❌ Missing XAI_API_KEY or XAI_MANAGEMENT_API_KEY")
        sys.exit(1)

    return Client(
        api_key=api_key,
        management_api_key=mgmt_key,
        timeout=300  # 5 min timeout for large files
    )


def create_collection(client: Client) -> str:
    """Create a new collection for full upload."""
    # Check existing collections
    response = client.collections.list()
    collections = response.collections if hasattr(response, 'collections') else []

    for c in collections:
        cname = c.collection_name if hasattr(c, 'collection_name') else ''
        cid = c.collection_id if hasattr(c, 'collection_id') else ''
        if cname == COLLECTION_NAME:
            docs = c.documents_count if hasattr(c, 'documents_count') else 0
            log(f"Found existing collection '{COLLECTION_NAME}': {docs} docs, ID: {cid}")
            return cid

    # Create new
    log(f"Creating new collection: {COLLECTION_NAME}")
    collection = client.collections.create(name=COLLECTION_NAME)
    cid = collection.collection_id if hasattr(collection, 'collection_id') else collection.id
    log(f"Created collection: {cid}")
    return cid


def load_checkpoint() -> dict:
    """Load checkpoint for resume capability."""
    if CHECKPOINT_FILE.exists():
        return json.loads(CHECKPOINT_FILE.read_text())
    return {"uploaded": [], "failed": [], "collection_id": None}


def save_checkpoint(checkpoint: dict):
    """Save checkpoint."""
    CHECKPOINT_FILE.write_text(json.dumps(checkpoint, indent=2))


def extract_metadata(pdf_path: Path) -> dict:
    """Extract metadata from PDF filename."""
    name = pdf_path.name

    # Year extraction
    year = None
    if match := re.search(r'(19|20)\d{2}', name):
        year = match.group(0)

    # Source type
    source_type = "unknown"
    if name.startswith("10."):
        source_type = "doi"
    elif re.match(r'\d{4}\.\d+', name):
        source_type = "arxiv"
    elif name.startswith("1-s2.0"):
        source_type = "sciencedirect"
    elif "PMC" in name:
        source_type = "pmc"
    elif "nature" in name.lower() or "s41" in name:
        source_type = "nature"

    # Size
    size_mb = pdf_path.stat().st_size / (1024 * 1024)

    return {
        "filename": name,
        "year": year or "",
        "source_type": source_type,
        "size_mb": f"{size_mb:.2f}"
    }


def upload_all(client: Client, collection_id: str, resume: bool = False):
    """Upload all PDFs with progress tracking."""
    # Get all PDFs
    all_pdfs = sorted(PDF_DIR.glob("*.pdf"))
    total = len(all_pdfs)

    log(f"Found {total} PDFs in {PDF_DIR}")

    # Load checkpoint
    checkpoint = load_checkpoint() if resume else {"uploaded": [], "failed": [], "collection_id": collection_id}
    checkpoint["collection_id"] = collection_id

    uploaded_set = set(checkpoint["uploaded"])
    failed_set = set(checkpoint["failed"])

    if resume and uploaded_set:
        log(f"Resuming: {len(uploaded_set)} already uploaded, {len(failed_set)} failed")

    # Upload loop
    success = 0
    failed = 0
    skipped = 0
    start_time = time.time()

    for i, pdf in enumerate(all_pdfs):
        # Skip already uploaded
        if pdf.name in uploaded_set:
            skipped += 1
            continue

        # Skip previously failed (could retry with --retry flag)
        if pdf.name in failed_set:
            skipped += 1
            continue

        # Upload with retry
        max_retries = 3
        uploaded = False

        for attempt in range(max_retries):
            try:
                metadata = extract_metadata(pdf)

                with open(pdf, "rb") as f:
                    client.collections.upload_document(
                        collection_id=collection_id,
                        name=pdf.name,
                        data=f.read(),
                        fields=metadata
                    )

                checkpoint["uploaded"].append(pdf.name)
                uploaded_set.add(pdf.name)
                success += 1
                uploaded = True
                break

            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    log(f"❌ Failed: {pdf.name} - {str(e)[:100]}")
                    checkpoint["failed"].append(pdf.name)
                    failed_set.add(pdf.name)
                    failed += 1

        # Progress logging
        done = success + failed + skipped
        if done % 50 == 0 or done == total:
            elapsed = time.time() - start_time
            rate = (success + failed) / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate / 60 if rate > 0 else 0
            log(f"Progress: {done}/{total} ({100*done/total:.1f}%) | ✓{success} ✗{failed} ⏭{skipped} | {rate:.1f}/s | ETA: {eta:.0f}min")
            save_checkpoint(checkpoint)

        # Rate limiting
        if uploaded:
            time.sleep(0.3)  # ~3 uploads/sec to be safe

    # Final checkpoint
    save_checkpoint(checkpoint)

    # Summary
    elapsed = time.time() - start_time
    log("=" * 60)
    log("UPLOAD COMPLETE")
    log("=" * 60)
    log(f"Total PDFs: {total}")
    log(f"Uploaded: {success}")
    log(f"Failed: {failed}")
    log(f"Skipped: {skipped}")
    log(f"Time: {elapsed/60:.1f} minutes")
    log(f"Collection ID: {collection_id}")
    log("")
    log("Monitor indexing progress with:")
    log(f"  python3 scripts/xai_smoke_test.py")


def main():
    parser = argparse.ArgumentParser(description="Upload all PDFs to xAI Collections")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    log("=" * 60)
    log("xAI Full Upload - 3,158 PDFs")
    log("=" * 60)

    client = get_client()
    log("✅ Client initialized")

    collection_id = create_collection(client)

    upload_all(client, collection_id, resume=args.resume)


if __name__ == "__main__":
    main()
