#!/usr/bin/env python3
"""
Fix Numeric Titles in Documents Table

This script attempts to recover meaningful titles for documents that have
numeric-only titles (a result of failed PDF title extraction).

Strategy:
1. Find documents with numeric titles
2. Get first passage from each document
3. Extract title from passage content using heuristics
4. Update documents table
5. Optionally rebuild affected ChromaDB entries

Usage:
    python3 scripts/fix_numeric_titles.py --dry-run     # Preview changes
    python3 scripts/fix_numeric_titles.py --apply       # Apply to Postgres
    python3 scripts/fix_numeric_titles.py --rebuild     # Apply + rebuild ChromaDB
"""

import argparse
import logging
import re
import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)

POSTGRES_DSN = "dbname=polymath user=polymath host=/var/run/postgresql"


def extract_title_from_passage(passage_text: str) -> str:
    """
    Extract a meaningful title from passage text.

    Returns extracted title or None if no good candidate found.
    """
    if not passage_text:
        return None

    lines = passage_text.strip().split('\n')

    # Strategy 1: Skip markdown header if numeric, look for real content
    for line in lines[:10]:
        line = line.strip()

        # Skip empty, numeric, or very short lines
        if not line or line.isdigit() or len(line) < 15:
            continue

        # Skip markdown headers that are just numbers
        if line.startswith('#'):
            header_text = line.lstrip('#').strip()
            if header_text.isdigit():
                continue
            # Real header found
            if len(header_text) > 10 and len(header_text) < 300:
                return header_text[:200]

        # Skip common metadata patterns
        skip_patterns = [
            r'^abstract', r'^keywords', r'^doi:', r'^copyright',
            r'^www\.', r'^http', r'^figure \d', r'^table \d',
            r'^©', r'^\d+\s*$', r'^et al', r'^page \d',
            r'^\s*\d+\s*$', r'^references$', r'^introduction$'
        ]
        if any(re.match(p, line.lower()) for p in skip_patterns):
            continue

        # Candidate found
        if len(line) > 300:
            # Try to find sentence boundary
            for sep in ['. ', ': ', ' - ']:
                if sep in line[:250]:
                    idx = line.index(sep)
                    if idx > 20:
                        return line[:idx]
            return line[:200] + "..."

        return line

    return None


def get_numeric_title_docs(conn, limit=None):
    """Get documents with numeric-only titles."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        query = """
            SELECT d.doc_id, d.title,
                   (SELECT passage_text FROM passages p
                    WHERE p.doc_id = d.doc_id
                    ORDER BY p.page_num, p.passage_id LIMIT 1) as first_passage
            FROM documents d
            WHERE d.title ~ '^[0-9]+$'
            ORDER BY d.doc_id
        """
        if limit:
            query += f" LIMIT {limit}"
        cur.execute(query)
        return [dict(row) for row in cur.fetchall()]


def update_document_title(conn, doc_id: str, new_title: str):
    """Update a document's title."""
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE documents SET title = %s, updated_at = NOW() WHERE doc_id = %s",
            (new_title, doc_id)
        )
        # Also update artifacts table
        cur.execute(
            "UPDATE artifacts SET title = %s WHERE doc_id = %s",
            (new_title, doc_id)
        )
    conn.commit()


def main():
    parser = argparse.ArgumentParser(description="Fix numeric titles in documents")
    parser.add_argument('--dry-run', action='store_true', help="Preview changes without applying")
    parser.add_argument('--apply', action='store_true', help="Apply changes to Postgres")
    parser.add_argument('--rebuild', action='store_true', help="Apply + rebuild ChromaDB")
    parser.add_argument('--limit', type=int, default=None, help="Limit documents to process")
    args = parser.parse_args()

    if not (args.dry_run or args.apply or args.rebuild):
        parser.print_help()
        return

    conn = psycopg2.connect(POSTGRES_DSN)

    logger.info("Finding documents with numeric titles...")
    docs = get_numeric_title_docs(conn, args.limit)
    logger.info(f"Found {len(docs)} documents with numeric titles")

    fixed = 0
    unfixable = 0

    for doc in docs:
        old_title = doc['title']
        passage = doc['first_passage']

        if not passage:
            logger.warning(f"No passages for doc {doc['doc_id']}")
            unfixable += 1
            continue

        new_title = extract_title_from_passage(passage)

        if new_title and new_title != old_title:
            if args.dry_run:
                logger.info(f"Would fix: {old_title} -> {new_title[:60]}...")
            else:
                update_document_title(conn, doc['doc_id'], new_title)
                logger.info(f"Fixed: {old_title} -> {new_title[:60]}...")
            fixed += 1
        else:
            unfixable += 1

    logger.info(f"\nSummary:")
    logger.info(f"  Fixed: {fixed}")
    logger.info(f"  Unfixable: {unfixable}")
    logger.info(f"  Total: {len(docs)}")

    if args.rebuild and fixed > 0:
        logger.info("\nTo rebuild ChromaDB, run:")
        logger.info("  python3 scripts/rebuild_chromadb_v2.py")

    conn.close()


if __name__ == "__main__":
    main()
