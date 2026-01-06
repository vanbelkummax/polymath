#!/usr/bin/env python3
"""
Metadata Enrichment Script for ChromaDB

This script enriches existing chunks with metadata extracted from their content.
It runs non-destructively in batches to avoid memory issues.

The key insight: Most chunks have format "Title: X\n\nAbstract: Y..."
We can parse this to recover paper titles and add minimal metadata.

Author: Librarian Agent
Date: 2026-01-04

Usage:
    python3 enrich_metadata.py --dry-run        # Preview only
    python3 enrich_metadata.py --batch-size 500 # Process in batches
    python3 enrich_metadata.py --live           # Actually update
"""

import sys
import re
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import argparse

# Lazy imports to speed up --help
def get_chromadb():
    import chromadb
    return chromadb

CHROMADB_PATH = "/home/user/work/polymax/chromadb/polymath_v2"
COLLECTION_NAME = "polymath_corpus"

def extract_title_from_content(content: str) -> Optional[str]:
    """
    Extract paper title from chunk content.

    Common patterns:
    - "Title: Some Paper Title\n\nAbstract: ..."
    - "Title: Some Paper Title\nAbstract: ..."
    - First line is title (if reasonable length)
    """
    if not content:
        return None

    # Pattern 1: Explicit "Title:" prefix
    if content.startswith("Title:"):
        # Find end of title (usually followed by Abstract or newline)
        match = re.match(r"Title:\s*(.+?)(?:\n\n|\nAbstract:|\n\n?$)", content, re.DOTALL)
        if match:
            title = match.group(1).strip()
            # Clean up multi-line titles
            title = re.sub(r'\s+', ' ', title)
            if 10 < len(title) < 300:
                return title

    # Pattern 2: First meaningful line
    lines = content.split('\n')
    for line in lines[:3]:  # Check first 3 lines
        line = line.strip()
        # Skip empty lines and short lines
        if not line or len(line) < 10:
            continue
        # Skip lines that look like metadata
        if line.startswith(('Abstract:', 'Keywords:', 'Author:', 'Date:')):
            continue
        # Accept reasonable title length
        if 10 < len(line) < 200:
            return line

    return None


def generate_paper_id(title: str) -> str:
    """Generate a stable paper ID from title."""
    normalized = title.lower().strip()
    normalized = re.sub(r'[^a-z0-9]+', '', normalized)
    return hashlib.md5(normalized.encode()).hexdigest()[:16]


def extract_authors_from_content(content: str) -> Optional[List[str]]:
    """Try to extract author names (best effort)."""
    # Look for common patterns
    patterns = [
        r"(?:Authors?|By|Written by):?\s*(.+?)(?:\n|$)",
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+et\s+al\.",
    ]

    for pattern in patterns:
        match = re.search(pattern, content[:1000], re.IGNORECASE)
        if match:
            authors = match.group(1)
            # Split on common delimiters
            author_list = re.split(r'[,;]|\s+and\s+', authors)
            author_list = [a.strip() for a in author_list if a.strip()]
            if author_list:
                return author_list[:5]  # Limit to 5 authors

    return None


class MetadataEnricher:
    """Enriches ChromaDB chunks with extracted metadata."""

    def __init__(self, dry_run: bool = True, batch_size: int = 100, verbose: bool = True):
        self.dry_run = dry_run
        self.batch_size = batch_size
        self.verbose = verbose
        self.client = None
        self.collection = None
        self.stats = {
            'total_processed': 0,
            'already_enriched': 0,
            'newly_enriched': 0,
            'no_title_found': 0,
            'errors': 0
        }

    def connect(self):
        """Connect to ChromaDB."""
        print(f"[{datetime.now():%H:%M:%S}] Connecting to ChromaDB...")
        chromadb = get_chromadb()
        self.client = chromadb.PersistentClient(path=CHROMADB_PATH)
        self.collection = self.client.get_collection(COLLECTION_NAME)
        print(f"  Collection: {COLLECTION_NAME}")
        print(f"  Total chunks: {self.collection.count():,}")

    def check_chunk_needs_enrichment(self, metadata: Optional[Dict]) -> bool:
        """Check if chunk metadata needs enrichment."""
        if metadata is None:
            return True
        # Check if it has the key fields we want
        if not metadata.get('title') or metadata.get('title') == 'Unknown':
            return True
        if not metadata.get('paper_id'):
            return True
        return False

    def process_batch(self, offset: int) -> int:
        """
        Process a batch of chunks starting at offset.
        Returns number of chunks actually processed.
        """
        # Get batch of chunks
        result = self.collection.get(
            offset=offset,
            limit=self.batch_size,
            include=['documents', 'metadatas']
        )

        if not result['ids']:
            return 0

        ids_to_update = []
        new_metadatas = []

        for i, (chunk_id, doc, meta) in enumerate(zip(
            result['ids'],
            result['documents'],
            result['metadatas']
        )):
            self.stats['total_processed'] += 1

            # Handle None metadata
            if meta is None:
                meta = {}

            # Check if needs enrichment
            if not self.check_chunk_needs_enrichment(meta):
                self.stats['already_enriched'] += 1
                continue

            # Try to extract title
            title = extract_title_from_content(doc)

            if not title:
                self.stats['no_title_found'] += 1
                # Still add minimal metadata
                if meta is None or meta == {}:
                    new_meta = {
                        'enriched_at': datetime.now().isoformat(),
                        'enrichment_version': 'v1',
                        'title_extracted': False
                    }
                    ids_to_update.append(chunk_id)
                    new_metadatas.append(new_meta)
                continue

            # Build enriched metadata
            new_meta = meta.copy() if meta else {}
            new_meta['title'] = title
            new_meta['paper_id'] = generate_paper_id(title)
            new_meta['enriched_at'] = datetime.now().isoformat()
            new_meta['enrichment_version'] = 'v1'
            new_meta['title_extracted'] = True

            # Try to extract authors (convert list to comma-separated string for ChromaDB)
            authors = extract_authors_from_content(doc)
            if authors:
                new_meta['authors'] = ', '.join(authors) if isinstance(authors, list) else str(authors)

            ids_to_update.append(chunk_id)
            new_metadatas.append(new_meta)
            self.stats['newly_enriched'] += 1

        # Update metadata in ChromaDB
        if ids_to_update and not self.dry_run:
            try:
                self.collection.update(
                    ids=ids_to_update,
                    metadatas=new_metadatas
                )
            except Exception as e:
                print(f"  ERROR updating batch: {e}")
                self.stats['errors'] += 1

        return len(result['ids'])

    def run(self, sample_only: int = None) -> Dict:
        """
        Run the enrichment process.

        Args:
            sample_only: If set, only process this many chunks (for testing)
        """
        self.connect()

        total_chunks = self.collection.count()
        if sample_only:
            total_chunks = min(sample_only, total_chunks)

        print(f"\n{'='*70}")
        print(f"METADATA ENRICHMENT - {'DRY RUN' if self.dry_run else 'LIVE MODE'}")
        print(f"{'='*70}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Chunks to process: {total_chunks:,}")
        print()

        offset = 0
        processed = 0

        while processed < total_chunks:
            chunk_count = self.process_batch(offset)
            if chunk_count == 0:
                break

            processed += chunk_count
            offset += chunk_count

            # Progress update
            if self.verbose and processed % (self.batch_size * 10) == 0:
                pct = (processed / total_chunks) * 100
                print(f"  Progress: {processed:,}/{total_chunks:,} ({pct:.1f}%) - "
                      f"enriched: {self.stats['newly_enriched']}")

            if sample_only and processed >= sample_only:
                break

        # Final summary
        print(f"\n{'='*70}")
        print(f"SUMMARY")
        print(f"{'='*70}")
        print(f"  Total processed: {self.stats['total_processed']:,}")
        print(f"  Already enriched: {self.stats['already_enriched']:,}")
        print(f"  Newly enriched: {self.stats['newly_enriched']:,}")
        print(f"  No title found: {self.stats['no_title_found']:,}")
        print(f"  Errors: {self.stats['errors']}")

        if self.dry_run:
            print(f"\n  [DRY RUN] Would update {self.stats['newly_enriched'] + self.stats['no_title_found']:,} chunks")
        else:
            print(f"\n  [LIVE] Updated {self.stats['newly_enriched'] + self.stats['no_title_found']:,} chunks")

        return self.stats


def main():
    parser = argparse.ArgumentParser(
        description="Enrich ChromaDB chunks with extracted metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview what would be enriched (safe):
  python3 enrich_metadata.py --dry-run

  # Test on first 1000 chunks:
  python3 enrich_metadata.py --dry-run --sample 1000

  # Process all chunks in small batches:
  python3 enrich_metadata.py --live --batch-size 100

  # Full enrichment:
  python3 enrich_metadata.py --live
        """
    )

    parser.add_argument('--dry-run', action='store_true', default=True,
                       help='Preview changes without updating (default: True)')
    parser.add_argument('--live', action='store_true',
                       help='Actually update metadata in ChromaDB')
    parser.add_argument('--batch-size', type=int, default=500,
                       help='Number of chunks per batch (default: 500)')
    parser.add_argument('--sample', type=int,
                       help='Only process first N chunks (for testing)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Minimal output')
    parser.add_argument('--output', '-o',
                       help='Save stats to JSON file')

    args = parser.parse_args()

    # Determine mode
    dry_run = not args.live

    # Create enricher
    enricher = MetadataEnricher(
        dry_run=dry_run,
        batch_size=args.batch_size,
        verbose=not args.quiet
    )

    try:
        stats = enricher.run(sample_only=args.sample)

        # Save stats if requested
        if args.output:
            stats['timestamp'] = datetime.now().isoformat()
            stats['dry_run'] = dry_run
            with open(args.output, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"\nStats saved to: {args.output}")

        return 0

    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
