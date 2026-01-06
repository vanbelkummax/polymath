#!/usr/bin/env python3
"""
Legacy Metadata Backfill - Adds titles to chunks missing metadata
Runs safely alongside ongoing ingestion (uses batched updates)
"""

import os
import sys
import re
import hashlib
from datetime import datetime

# Avoid GPU contention with OCR process
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import chromadb
from typing import Optional

CHROMADB_PATH = "/home/user/work/polymax/chromadb/polymath_v2"
BATCH_SIZE = 100
PROGRESS_FILE = "/home/user/work/polymax/backfill_progress.json"


def extract_title_from_text(text: str, chunk_id: str) -> str:
    """Extract a reasonable title from chunk text."""
    if not text:
        return f"chunk_{chunk_id[:8]}"

    # Try to find title-like content in first 500 chars
    header = text[:500]

    # Common patterns for paper titles
    patterns = [
        r'^#\s*(.+?)[\n\r]',  # Markdown header
        r'^Title:\s*(.+?)[\n\r]',  # Explicit title
        r'^([A-Z][^.!?\n]{20,150})[.\n]',  # Capitalized sentence
    ]

    for pattern in patterns:
        match = re.search(pattern, header, re.MULTILINE)
        if match:
            title = match.group(1).strip()
            if 10 < len(title) < 200:
                return title[:200]

    # Fall back to first meaningful line
    for line in header.split('\n'):
        line = line.strip().strip('#').strip()
        if 15 < len(line) < 200 and not line.startswith(('http', '!', '|', '-', '*', '{')):
            return line[:200]

    return f"chunk_{chunk_id[:8]}"


def infer_source_from_text(text: str) -> str:
    """Try to infer source type from content."""
    text_lower = text[:1000].lower()

    if any(x in text_lower for x in ['abstract', 'introduction', 'methods', 'results', 'doi:', 'arxiv']):
        return 'paper'
    if any(x in text_lower for x in ['def ', 'class ', 'import ', 'function']):
        return 'code'
    if any(x in text_lower for x in ['chapter', 'section', 'page']):
        return 'textbook'
    return 'document'


def backfill_batch(collection, ids: list, documents: list, existing_metas: list) -> int:
    """Update metadata for a batch of chunks."""
    updated = 0
    new_metas = []

    for i, (chunk_id, doc, meta) in enumerate(zip(ids, documents, existing_metas)):
        # Check if metadata needs updating
        if meta and meta.get('title') and meta.get('source'):
            new_metas.append(meta)  # Keep existing
            continue

        # Build new metadata
        new_meta = dict(meta) if meta else {}

        if not new_meta.get('title'):
            new_meta['title'] = extract_title_from_text(doc, chunk_id)

        if not new_meta.get('source'):
            new_meta['source'] = infer_source_from_text(doc)

        new_meta['backfilled'] = True
        new_meta['backfill_date'] = datetime.now().isoformat()[:10]

        new_metas.append(new_meta)
        updated += 1

    # Batch update
    if updated > 0:
        collection.update(
            ids=ids,
            metadatas=new_metas
        )

    return updated


def load_progress() -> dict:
    """Load progress from file."""
    import json
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE) as f:
                return json.load(f)
        except:
            pass
    return {'offset': 0, 'updated': 0, 'skipped': 0}


def save_progress(progress: dict):
    """Save progress to file."""
    import json
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f)


def main():
    print("=" * 60)
    print("LEGACY METADATA BACKFILL")
    print("=" * 60)
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M}")

    # Connect to ChromaDB
    print("\nConnecting to ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMADB_PATH)
    collection = client.get_collection("polymath_corpus")

    total_chunks = collection.count()
    print(f"Total chunks in collection: {total_chunks:,}")

    # Load progress
    progress = load_progress()
    offset = progress.get('offset', 0)
    total_updated = progress.get('updated', 0)
    total_skipped = progress.get('skipped', 0)

    if offset > 0:
        print(f"Resuming from offset {offset:,} (already updated {total_updated:,})")

    print(f"\nProcessing in batches of {BATCH_SIZE}...")
    print("-" * 60)

    batch_num = 0

    while offset < total_chunks:
        try:
            # Fetch batch
            result = collection.get(
                limit=BATCH_SIZE,
                offset=offset,
                include=['documents', 'metadatas']
            )

            if not result['ids']:
                break

            ids = result['ids']
            docs = result['documents']
            metas = result['metadatas']

            # Count how many need updating
            needs_update = sum(1 for m in metas if not m or not m.get('title'))

            if needs_update > 0:
                updated = backfill_batch(collection, ids, docs, metas)
                total_updated += updated
                total_skipped += len(ids) - updated
            else:
                total_skipped += len(ids)

            offset += len(ids)
            batch_num += 1

            # Progress update every 10 batches
            if batch_num % 10 == 0:
                pct = (offset / total_chunks) * 100
                print(f"[{pct:5.1f}%] Processed {offset:,}/{total_chunks:,} | Updated: {total_updated:,} | Skipped: {total_skipped:,}")

                # Save progress
                save_progress({'offset': offset, 'updated': total_updated, 'skipped': total_skipped})

        except Exception as e:
            print(f"\nError at offset {offset}: {e}")
            save_progress({'offset': offset, 'updated': total_updated, 'skipped': total_skipped})
            raise

    # Final save
    save_progress({'offset': offset, 'updated': total_updated, 'skipped': total_skipped, 'complete': True})

    print("\n" + "=" * 60)
    print("BACKFILL COMPLETE")
    print("=" * 60)
    print(f"Total processed: {offset:,}")
    print(f"Updated: {total_updated:,}")
    print(f"Already had metadata: {total_skipped:,}")
    print(f"Finished: {datetime.now():%Y-%m-%d %H:%M}")


if __name__ == "__main__":
    main()
