#!/usr/bin/env python3
"""
Upload ALL PDFs to Zotero.

Creates a "Polymath_Full" collection and uploads all 3,158 PDFs.
Uses filename-based metadata extraction.

Usage:
    python3 scripts/zotero_full_upload.py

    # Resume from checkpoint
    python3 scripts/zotero_full_upload.py --resume
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Load .env
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

try:
    from pyzotero import zotero
except ImportError:
    print("Installing pyzotero...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyzotero", "-q"])
    from pyzotero import zotero

# Configuration
PDF_DIR = Path("/scratch/polymath_archive/pdfs")
COLLECTION_NAME = "Polymath_Full"
CHECKPOINT_FILE = Path("/home/user/work/polymax/logs/zotero_full_upload_checkpoint.json")
LOG_FILE = Path("/home/user/work/polymax/logs/zotero_full_upload.log")

ZOTERO_API_KEY = os.getenv("ZOTERO_API_KEY")
ZOTERO_USER_ID = os.getenv("ZOTERO_USER_ID")

# Ensure log directory exists
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def log(msg: str):
    """Log to both console and file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def get_zotero_client():
    """Initialize Zotero client."""
    if not ZOTERO_API_KEY:
        log("❌ ZOTERO_API_KEY not set")
        sys.exit(1)
    if not ZOTERO_USER_ID:
        log("❌ ZOTERO_USER_ID not set")
        sys.exit(1)

    return zotero.Zotero(ZOTERO_USER_ID, 'user', ZOTERO_API_KEY)


def get_or_create_collection(zot, name: str) -> str:
    """Get existing collection or create new one."""
    collections = zot.collections()

    for c in collections:
        if c['data']['name'] == name:
            log(f"Using existing collection: {name} ({c['key']})")
            return c['key']

    # Create new collection
    result = zot.create_collections([{'name': name}])
    if result and 'successful' in result:
        key = list(result['successful'].values())[0]['key']
        log(f"Created collection: {name} ({key})")
        return key

    raise Exception(f"Failed to create collection: {result}")


def extract_metadata(pdf_path: Path) -> Dict:
    """Extract metadata from PDF filename."""
    name = pdf_path.stem  # filename without extension

    # Year extraction
    year = None
    if match := re.search(r'(19|20)\d{2}', name):
        year = match.group(0)

    # Source type detection and ID extraction
    source_type = "unknown"
    identifier = None

    # DOI pattern (10.xxxx/...)
    if name.startswith("10."):
        source_type = "doi"
        identifier = name.replace("_", "/")

    # arXiv pattern (YYMM.NNNNN)
    elif re.match(r'\d{4}\.\d+', name):
        source_type = "arxiv"
        identifier = re.match(r'(\d{4}\.\d+)', name).group(1)

    # ScienceDirect pattern
    elif name.startswith("1-s2.0"):
        source_type = "sciencedirect"
        if match := re.search(r'S\d+', name):
            identifier = match.group(0)

    # PMC pattern
    elif "PMC" in name:
        source_type = "pmc"
        if match := re.search(r'PMC\d+', name):
            identifier = match.group(0)

    # Nature/Springer pattern
    elif "s41" in name or "nature" in name.lower():
        source_type = "nature"

    # Clean up title from filename
    title = name
    # Remove common prefixes
    title = re.sub(r'^1-s2\.0-S\d+-', '', title)
    title = re.sub(r'^10\.\d+_', '', title)
    title = re.sub(r'^\d{4}\.\d+v?\d*', '', title)
    title = re.sub(r'^PMC\d+_', '', title)
    # Clean up
    title = title.replace('_', ' ').replace('-', ' ')
    title = re.sub(r'\s+', ' ', title).strip()

    if not title or len(title) < 5:
        title = name  # Fall back to filename

    return {
        'title': title[:200],  # Zotero has title length limits
        'year': year,
        'source_type': source_type,
        'identifier': identifier,
        'filename': pdf_path.name
    }


def create_zotero_item(metadata: Dict, pdf_path: Path) -> Dict:
    """Create Zotero item from metadata."""
    source = metadata.get('source_type', 'unknown')

    # Determine item type
    if source == 'arxiv':
        item_type = 'preprint'
    else:
        item_type = 'journalArticle'

    item = {
        'itemType': item_type,
        'title': metadata['title'],
    }

    # Year/Date
    if metadata.get('year'):
        item['date'] = str(metadata['year'])

    # Identifiers
    identifier = metadata.get('identifier')
    if source == 'doi' and identifier:
        item['DOI'] = identifier
    elif source == 'arxiv' and identifier:
        item['archiveID'] = f"arXiv:{identifier}"
        item['url'] = f"https://arxiv.org/abs/{identifier}"

    # Extra field
    extra = [
        f"Source: {source}",
        f"Filename: {metadata['filename']}"
    ]
    if source == 'pmc' and identifier:
        extra.append(f"PMCID: {identifier}")
    item['extra'] = '\n'.join(extra)

    return item


def load_checkpoint() -> dict:
    """Load checkpoint for resume capability."""
    if CHECKPOINT_FILE.exists():
        return json.loads(CHECKPOINT_FILE.read_text())
    return {"uploaded": [], "failed": [], "collection_key": None}


def save_checkpoint(checkpoint: dict):
    """Save checkpoint."""
    CHECKPOINT_FILE.write_text(json.dumps(checkpoint, indent=2))


def upload_all(zot, collection_key: str, resume: bool = False):
    """Upload all PDFs with progress tracking."""
    # Get all PDFs
    all_pdfs = sorted(PDF_DIR.glob("*.pdf"))
    total = len(all_pdfs)

    log(f"Found {total} PDFs in {PDF_DIR}")

    # Load checkpoint
    checkpoint = load_checkpoint() if resume else {"uploaded": [], "failed": [], "collection_key": collection_key}
    checkpoint["collection_key"] = collection_key

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

        # Skip previously failed
        if pdf.name in failed_set:
            skipped += 1
            continue

        # Extract metadata and create item
        try:
            metadata = extract_metadata(pdf)
            item = create_zotero_item(metadata, pdf)
            item['collections'] = [collection_key]

            # Create the item
            result = zot.create_items([item])

            if result and 'successful' in result and result['successful']:
                item_key = list(result['successful'].values())[0]['key']

                # Upload PDF attachment
                try:
                    zot.attachment_simple([str(pdf)], item_key)
                except Exception as e:
                    # Attachment failed but item created
                    log(f"⚠️  Attachment failed for {pdf.name}: {str(e)[:50]}")

                checkpoint["uploaded"].append(pdf.name)
                uploaded_set.add(pdf.name)
                success += 1
            else:
                log(f"❌ Failed to create item: {pdf.name}")
                checkpoint["failed"].append(pdf.name)
                failed_set.add(pdf.name)
                failed += 1

        except Exception as e:
            log(f"❌ Error: {pdf.name} - {str(e)[:100]}")
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

        # Rate limiting (Zotero: ~50 requests per 10 seconds for free tier)
        # With storage plan: more generous limits
        time.sleep(0.3)

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
    log(f"Collection: {COLLECTION_NAME} ({collection_key})")


def main():
    parser = argparse.ArgumentParser(description="Upload all PDFs to Zotero")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    log("=" * 60)
    log("Zotero Full Upload - 3,158 PDFs")
    log("=" * 60)

    zot = get_zotero_client()
    log("✅ Zotero client initialized")

    collection_key = get_or_create_collection(zot, COLLECTION_NAME)

    upload_all(zot, collection_key, resume=args.resume)


if __name__ == "__main__":
    main()
