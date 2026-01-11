#!/usr/bin/env python3
"""
Enrich Zotero items with metadata from GROBID enrichment.

Updates items in Polymath_Full collection with proper titles, authors, DOIs, etc.

Usage:
    python3 scripts/zotero_enrich_metadata.py --dry-run   # Preview changes
    python3 scripts/zotero_enrich_metadata.py             # Apply changes
"""

import argparse
import json
import os
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

from pyzotero import zotero

ZOTERO_API_KEY = os.getenv("ZOTERO_API_KEY")
ZOTERO_USER_ID = os.getenv("ZOTERO_USER_ID")
METADATA_FILE = Path("/home/user/work/polymax/logs/metadata_enrichment_full.jsonl")
LOG_FILE = Path("/home/user/work/polymax/logs/zotero_enrich.log")

LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def log(msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def load_metadata() -> dict:
    """Load metadata indexed by filename."""
    metadata = {}
    for line in METADATA_FILE.read_text().splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            filename = Path(d.get('path', '')).name
            if filename and d.get('confidence', 0) > 0.5:
                metadata[filename] = d
        except json.JSONDecodeError:
            continue
    return metadata


def get_zotero_client():
    if not ZOTERO_API_KEY or not ZOTERO_USER_ID:
        log("Missing ZOTERO_API_KEY or ZOTERO_USER_ID")
        sys.exit(1)
    return zotero.Zotero(ZOTERO_USER_ID, 'user', ZOTERO_API_KEY)


def find_collection(zot, name: str) -> str:
    """Find collection key by name."""
    for c in zot.collections():
        if c['data']['name'] == name:
            return c['key']
    return None


def format_authors(authors: list) -> list:
    """Convert author names to Zotero creator format."""
    creators = []
    for author in (authors or []):
        if isinstance(author, str):
            parts = author.rsplit(' ', 1)
            if len(parts) == 2:
                creators.append({
                    'creatorType': 'author',
                    'firstName': parts[0],
                    'lastName': parts[1]
                })
            else:
                creators.append({
                    'creatorType': 'author',
                    'name': author
                })
    return creators


def enrich_items(zot, collection_key: str, metadata: dict, dry_run: bool = True):
    """Enrich items with metadata."""
    # Get all items in collection (with pagination)
    log(f"Fetching items from collection...")
    items = zot.everything(zot.collection_items(collection_key))
    log(f"Found {len(items)} items")

    updated = 0
    skipped = 0
    no_metadata = 0

    for item in items:
        item_key = item['key']
        item_data = item['data']

        # Skip attachments
        if item_data.get('itemType') == 'attachment':
            continue

        # Get filename from extra field or title
        filename = None
        extra = item_data.get('extra', '')
        for line in extra.split('\n'):
            if line.startswith('Filename:'):
                filename = line.split(':', 1)[1].strip()
                break

        if not filename:
            # Try to get from title if it looks like a filename
            title = item_data.get('title', '')
            if title.endswith('.pdf') or '.' in title:
                filename = title

        if not filename:
            skipped += 1
            continue

        # Look up metadata
        meta = metadata.get(filename)
        if not meta:
            no_metadata += 1
            continue

        # Check if already has good metadata
        if item_data.get('DOI') and item_data.get('creators'):
            skipped += 1
            continue

        # Build updates
        updates = {}

        if meta.get('title') and meta['title'] != filename:
            updates['title'] = meta['title'][:500]

        if meta.get('authors'):
            updates['creators'] = format_authors(meta['authors'])

        if meta.get('year'):
            updates['date'] = str(meta['year'])

        if meta.get('doi'):
            updates['DOI'] = meta['doi']

        if meta.get('abstract'):
            updates['abstractNote'] = meta['abstract'][:10000]

        if meta.get('arxiv_id'):
            updates['url'] = f"https://arxiv.org/abs/{meta['arxiv_id']}"

        if not updates:
            skipped += 1
            continue

        if dry_run:
            log(f"Would update: {filename[:50]} → {updates.get('title', 'N/A')[:40]}")
            updated += 1
        else:
            try:
                # Merge updates into existing data
                for key, value in updates.items():
                    item_data[key] = value

                zot.update_item(item)
                updated += 1

                if updated % 50 == 0:
                    log(f"Progress: {updated} updated")

                time.sleep(0.2)  # Rate limiting

            except Exception as e:
                log(f"Error updating {filename}: {e}")

    log(f"\n{'DRY RUN - ' if dry_run else ''}Summary:")
    log(f"  Updated: {updated}")
    log(f"  Skipped (already good): {skipped}")
    log(f"  No metadata found: {no_metadata}")


def main():
    parser = argparse.ArgumentParser(description="Enrich Zotero items with GROBID metadata")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying")
    parser.add_argument("--collection", default="Polymath_Full", help="Collection name")
    args = parser.parse_args()

    log("=" * 60)
    log("Zotero Metadata Enrichment")
    log("=" * 60)

    # Load metadata
    log(f"Loading metadata from {METADATA_FILE}...")
    metadata = load_metadata()
    log(f"Loaded {len(metadata)} items with confidence > 0.5")

    # Connect to Zotero
    zot = get_zotero_client()
    log("Connected to Zotero")

    # Find collection
    collection_key = find_collection(zot, args.collection)
    if not collection_key:
        log(f"Collection '{args.collection}' not found")
        sys.exit(1)
    log(f"Found collection: {args.collection} ({collection_key})")

    # Enrich items
    enrich_items(zot, collection_key, metadata, dry_run=args.dry_run)

    if args.dry_run:
        log("\nRun without --dry-run to apply changes")


if __name__ == "__main__":
    main()
