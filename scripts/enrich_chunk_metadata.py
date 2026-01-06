#!/usr/bin/env python3
"""
Enrich chunk metadata with citation information.

Fixes: 0% DOI, 0% year, 0% venue coverage.
"""

import sys
sys.path.insert(0, '/home/user/work/polymax')

import chromadb
import psycopg2
import json
from typing import Dict, Optional


def get_artifact_metadata(chunk_id: str, pg_conn) -> Optional[Dict]:
    """Fetch full metadata from Postgres (joins chunks → artifacts + source_items).

    Args:
        chunk_id: The ChromaDB chunk ID (e.g., "doc_1"), corresponds to chunks.id in Postgres
    """
    cursor = pg_conn.cursor()
    cursor.execute("""
        SELECT
            a.title,
            a.authors,
            a.year,
            a.source_url,
            a.artifact_type,
            s.doi,
            s.meta_json->>'venue' as venue,
            s.url as source_url_backup
        FROM chunks c
        JOIN artifacts a ON c.artifact_id = a.id
        LEFT JOIN source_items s ON a.source_item_id = s.id
        WHERE c.id = %s
    """, (chunk_id,))

    row = cursor.fetchone()
    if not row:
        return None

    # Combine authors array into string
    authors_str = None
    if row[1]:  # authors array
        authors_str = ', '.join(row[1][:3])  # First 3 authors

    return {
        'title': row[0],
        'authors': authors_str,
        'year': row[2],
        'doi': row[5],  # from source_items
        'venue': row[6],  # from source_items.meta_json
        'url': row[3] or row[7],  # prefer artifact.source_url, fallback to source_items.url
        'type': row[4]
    }


def enrich_chunks(batch_size: int = 1000, dry_run: bool = True):
    """
    Enrich ChromaDB chunks with Postgres metadata.

    Strategy:
    1. Query chunks with paper_id
    2. Lookup artifact in Postgres
    3. Add doi, year, venue, url to chunk metadata
    4. Update ChromaDB
    """
    # Connect to databases
    chroma_client = chromadb.PersistentClient('/home/user/work/polymax/chromadb/polymath_v2')
    collection = chroma_client.get_collection('polymath_corpus')

    try:
        pg_conn = psycopg2.connect("dbname=polymath user=polymath host=/var/run/postgresql")
    except:
        print("ERROR: Postgres not available")
        return

    # Get total count
    total = collection.count()
    print(f"Total chunks: {total:,}")

    enriched = 0
    skipped = 0
    errors = 0

    # Process in batches
    offset = 0
    while offset < total:
        # Fetch batch
        batch = collection.get(
            limit=batch_size,
            offset=offset,
            include=['metadatas']
        )

        for i, (chunk_id, metadata) in enumerate(zip(batch['ids'], batch['metadatas'])):
            # Skip if already enriched with citation metadata
            if 'doi' in metadata or 'year' in metadata:
                skipped += 1
                continue

            # Fetch from Postgres using ChromaDB chunk ID (e.g., "doc_1")
            # NOT the metadata paper_id field (which is a different hash)
            artifact_meta = get_artifact_metadata(chunk_id, pg_conn)
            if not artifact_meta:
                errors += 1
                continue

            # Prepare enriched metadata
            enriched_meta = {**metadata}  # Copy existing

            # Add citation fields (only if not None)
            if artifact_meta['doi']:
                enriched_meta['doi'] = artifact_meta['doi']
            if artifact_meta['year']:
                enriched_meta['year'] = artifact_meta['year']
            if artifact_meta['venue']:
                enriched_meta['venue'] = artifact_meta['venue']
            if artifact_meta['url']:
                enriched_meta['url'] = artifact_meta['url']
            if artifact_meta['authors']:
                enriched_meta['authors'] = artifact_meta['authors']

            # Update ChromaDB
            if not dry_run:
                collection.update(
                    ids=[chunk_id],
                    metadatas=[enriched_meta]
                )

            enriched += 1

            if enriched % 100 == 0:
                print(f"Progress: {offset + i}/{total} | Enriched: {enriched} | Skipped: {skipped} | Errors: {errors}")

        offset += batch_size

    print(f"\n=== ENRICHMENT COMPLETE ===")
    print(f"Total: {total:,}")
    print(f"Enriched: {enriched:,}")
    print(f"Skipped: {skipped:,}")
    print(f"Errors: {errors:,}")

    if dry_run:
        print("\n⚠️  DRY RUN - No changes made. Run with --execute to apply.")

    pg_conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true', help='Actually update chunks')
    parser.add_argument('--batch-size', type=int, default=1000)

    args = parser.parse_args()

    enrich_chunks(batch_size=args.batch_size, dry_run=not args.execute)
