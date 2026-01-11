#!/usr/bin/env python3
"""
Upload enriched PDFs to Zotero

Creates a "Polymath" collection and uploads PDFs with extracted metadata.

Usage:
    # Set credentials
    export ZOTERO_API_KEY="your-api-key"
    export ZOTERO_USER_ID="your-user-id"

    # Dry run (show what would be uploaded)
    python3 scripts/upload_to_zotero.py --metadata metadata_enrichment.jsonl

    # Upload
    python3 scripts/upload_to_zotero.py --metadata metadata_enrichment.jsonl --upload

    # Upload only high-confidence matches
    python3 scripts/upload_to_zotero.py --metadata metadata_enrichment.jsonl --upload --min-confidence 0.75
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

try:
    from pyzotero import zotero
except ImportError:
    print("Installing pyzotero...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyzotero", "-q"])
    from pyzotero import zotero

ZOTERO_API_KEY = os.getenv("ZOTERO_API_KEY")
ZOTERO_USER_ID = os.getenv("ZOTERO_USER_ID")
COLLECTION_NAME = "Polymath"


@dataclass
class PaperMetadata:
    path: str
    sha1: str
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    year: Optional[int] = None
    venue: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    pmid: Optional[str] = None
    abstract: Optional[str] = None
    source: str = "unknown"
    confidence: float = 0.0
    match_explain: str = ""
    needs_review: bool = False


def get_zotero_client():
    """Initialize Zotero client."""
    if not ZOTERO_API_KEY:
        print("Error: ZOTERO_API_KEY not set")
        sys.exit(1)
    if not ZOTERO_USER_ID:
        print("Error: ZOTERO_USER_ID not set")
        sys.exit(1)

    return zotero.Zotero(ZOTERO_USER_ID, 'user', ZOTERO_API_KEY)


def get_or_create_collection(zot, name: str) -> str:
    """Get existing collection or create new one."""
    collections = zot.collections()

    for c in collections:
        if c['data']['name'] == name:
            print(f"Using existing collection: {name} ({c['key']})")
            return c['key']

    # Create new collection
    result = zot.create_collections([{'name': name}])
    if result and 'successful' in result:
        key = list(result['successful'].values())[0]['key']
        print(f"Created collection: {name} ({key})")
        return key

    raise Exception(f"Failed to create collection: {result}")


def create_zotero_item(metadata: PaperMetadata) -> Dict:
    """Create Zotero item template from metadata."""
    # Determine item type based on source
    if metadata.arxiv_id:
        item_type = 'preprint'
    elif metadata.venue and 'conference' in metadata.venue.lower():
        item_type = 'conferencePaper'
    else:
        item_type = 'journalArticle'

    item = {
        'itemType': item_type,
        'title': metadata.title or Path(metadata.path).stem,
        'abstractNote': metadata.abstract or '',
    }

    # Authors
    if metadata.authors:
        item['creators'] = []
        for author in metadata.authors:
            parts = author.rsplit(' ', 1)
            if len(parts) == 2:
                item['creators'].append({
                    'creatorType': 'author',
                    'firstName': parts[0],
                    'lastName': parts[1]
                })
            else:
                item['creators'].append({
                    'creatorType': 'author',
                    'name': author
                })

    # Year/Date
    if metadata.year:
        item['date'] = str(metadata.year)

    # Venue
    if metadata.venue:
        if item_type == 'journalArticle':
            item['publicationTitle'] = metadata.venue
        elif item_type == 'conferencePaper':
            item['conferenceName'] = metadata.venue

    # Identifiers
    if metadata.doi:
        item['DOI'] = metadata.doi

    if metadata.arxiv_id:
        item['archiveID'] = f"arXiv:{metadata.arxiv_id}"
        item['url'] = f"https://arxiv.org/abs/{metadata.arxiv_id}"

    # Extra field for additional info
    extra = []
    if metadata.pmid:
        extra.append(f"PMID: {metadata.pmid}")
    if metadata.source:
        extra.append(f"Source: {metadata.source}")
    if metadata.confidence:
        extra.append(f"Confidence: {metadata.confidence:.0%}")
    if extra:
        item['extra'] = '\n'.join(extra)

    return item


def upload_with_attachment(zot, item: Dict, pdf_path: Path, collection_key: str) -> bool:
    """Upload item with PDF attachment."""
    try:
        # Add to collection
        item['collections'] = [collection_key]

        # Create the item first
        result = zot.create_items([item])

        if not result or 'successful' not in result or not result['successful']:
            print(f"  Failed to create item: {result}")
            return False

        item_key = list(result['successful'].values())[0]['key']

        # Upload PDF attachment
        if pdf_path.exists():
            zot.attachment_simple([str(pdf_path)], item_key)
            return True
        else:
            print(f"  Warning: PDF not found at {pdf_path}")
            return True  # Item created, just no attachment

    except Exception as e:
        print(f"  Error: {e}")
        return False


def load_metadata(jsonl_path: Path) -> List[PaperMetadata]:
    """Load metadata from JSONL file."""
    items = []
    with open(jsonl_path) as f:
        for line in f:
            data = json.loads(line)
            items.append(PaperMetadata(**data))
    return items


def main():
    parser = argparse.ArgumentParser(description="Upload PDFs to Zotero")
    parser.add_argument("--metadata", required=True, help="Path to metadata JSONL file")
    parser.add_argument("--upload", action="store_true", help="Actually upload (otherwise dry run)")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="Minimum confidence threshold (0.0-1.0)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of uploads")
    parser.add_argument("--collection", default=COLLECTION_NAME, help="Zotero collection name")
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        print(f"Error: {metadata_path} not found")
        sys.exit(1)

    # Load metadata
    print(f"Loading metadata from {metadata_path}...")
    all_items = load_metadata(metadata_path)

    # Filter by confidence
    items = [m for m in all_items if m.confidence >= args.min_confidence]
    print(f"Found {len(items)} items with confidence >= {args.min_confidence:.0%}")

    if args.limit > 0:
        items = items[:args.limit]
        print(f"Limited to {len(items)} items")

    if not items:
        print("No items to upload")
        return

    # Show sample
    print("\nSample items to upload:")
    for m in items[:5]:
        print(f"  {m.confidence:.0%} | {m.title or Path(m.path).name}")

    if not args.upload:
        print(f"\n[DRY RUN] Would upload {len(items)} items to collection '{args.collection}'")
        print("Run with --upload to actually upload")
        return

    # Initialize Zotero
    print("\nConnecting to Zotero...")
    zot = get_zotero_client()

    # Get or create collection
    collection_key = get_or_create_collection(zot, args.collection)

    # Upload items
    print(f"\nUploading {len(items)} items...")
    success = 0
    failed = 0

    for i, metadata in enumerate(items):
        print(f"[{i+1}/{len(items)}] {metadata.title or Path(metadata.path).name}...")

        item = create_zotero_item(metadata)
        pdf_path = Path(metadata.path)

        if upload_with_attachment(zot, item, pdf_path, collection_key):
            success += 1
        else:
            failed += 1

        # Rate limiting (Zotero allows ~50 requests per 10 seconds)
        if (i + 1) % 10 == 0:
            time.sleep(1)

    print(f"\n{'='*60}")
    print(f"Upload complete: {success} success, {failed} failed")
    print(f"Collection: {args.collection}")


if __name__ == "__main__":
    main()
