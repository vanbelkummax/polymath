#!/usr/bin/env python3
"""
Batch populate passages table from existing PDFs.

This script parses PDFs using enhanced_pdf_parser and populates the passages
table with page-local coordinates for evidence-bound citations.

Usage:
    python3 scripts/populate_passages_batch.py --limit 100 --dry-run
    python3 scripts/populate_passages_batch.py --all
    python3 scripts/populate_passages_batch.py --dir /path/to/pdfs --limit 50
"""
import argparse
import logging
import sys
import uuid
from pathlib import Path
from typing import Optional, List, Tuple
import psycopg2
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.enhanced_pdf_parser import EnhancedPDFParser
from lib.doc_identity import compute_doc_id, normalize_title

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# PDF directories to process
PDF_DIRS = [
    Path('/home/user/work/polymath_pdfs'),
    Path('/home/user/work/polymax/ingest_staging'),
    Path('/home/user/work/polymax/ingest_staging/yuankai_huo'),
]


def get_db_connection():
    """Get database connection."""
    return psycopg2.connect(
        dbname='polymath',
        user='polymath',
        host='/var/run/postgresql'
    )


def find_all_pdfs(directories: List[Path]) -> List[Path]:
    """Find all PDFs in given directories."""
    pdfs = []
    for dir_path in directories:
        if dir_path.exists():
            pdfs.extend(dir_path.glob('*.pdf'))
    return sorted(set(pdfs))


def get_docs_with_passages(conn) -> set:
    """Get set of doc_ids that already have passages."""
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT doc_id::text FROM passages")
    return {row[0] for row in cursor.fetchall()}


def find_or_create_doc(conn, pdf_path: Path) -> Optional[uuid.UUID]:
    """
    Find existing doc or create new one based on PDF filename.

    Returns doc_id if found/created, None if cannot determine identity.
    """
    cursor = conn.cursor()

    # Extract title from filename
    title = pdf_path.stem
    # Clean up common filename patterns
    title = title.replace('_', ' ').replace('-', ' ')

    # Try to find existing document by title similarity
    title_hash = normalize_title(title)[:16] if len(title) > 16 else normalize_title(title)

    cursor.execute("""
        SELECT doc_id, title FROM documents
        WHERE LOWER(title) LIKE %s
        OR title_hash LIKE %s
        LIMIT 1
    """, (f'%{title[:30].lower()}%', f'{title_hash}%'))

    result = cursor.fetchone()
    if result:
        return uuid.UUID(result[0])

    # Create new document entry
    doc_id = compute_doc_id(
        title=title,
        authors=['Unknown'],
        year=2024  # Default year
    )

    try:
        cursor.execute("""
            INSERT INTO documents (doc_id, title, authors, year, parser_version)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (doc_id) DO NOTHING
            RETURNING doc_id
        """, (str(doc_id), title, ['Unknown'], 2024, 'batch_populate_v1'))
        conn.commit()
        return doc_id
    except Exception as e:
        logger.warning(f"Could not create document for {pdf_path.name}: {e}")
        conn.rollback()
        return None


def process_pdf(pdf_path: Path, parser: EnhancedPDFParser = None) -> Tuple[int, str]:
    """
    Process a single PDF and return passages.

    Returns: (passage_count, error_message or '')
    """
    # Create parser if not provided (needed for multiprocessing)
    if parser is None:
        parser = EnhancedPDFParser()

    conn = get_db_connection()

    try:
        # Find or create document
        doc_id = find_or_create_doc(conn, pdf_path)
        if not doc_id:
            return (0, f"Could not create document for {pdf_path.name}")

        # Check if already processed
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM passages WHERE doc_id = %s",
            (str(doc_id),)
        )
        existing_count = cursor.fetchone()[0]
        if existing_count > 0:
            return (0, f"Already has {existing_count} passages")

        # Parse PDF
        passages = parser.extract_with_provenance(str(pdf_path), doc_id)

        if not passages:
            return (0, "No passages extracted")

        # Insert passages
        inserted = 0
        for p in passages:
            try:
                cursor.execute("""
                    INSERT INTO passages
                    (passage_id, doc_id, page_num, page_char_start, page_char_end,
                     section, passage_text, quality_score, parser_version)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (passage_id) DO NOTHING
                """, (
                    str(p.passage_id), str(p.doc_id), p.page_num,
                    p.page_char_start, p.page_char_end, p.section,
                    p.passage_text, p.quality_score, p.parser_version
                ))
                inserted += 1
            except Exception as e:
                logger.debug(f"Failed to insert passage: {e}")

        conn.commit()
        return (inserted, '')

    except Exception as e:
        conn.rollback()
        return (0, str(e))
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description='Batch populate passages table from PDFs'
    )
    parser.add_argument('--limit', type=int, default=100,
                        help='Max number of PDFs to process')
    parser.add_argument('--all', action='store_true',
                        help='Process all PDFs (ignore limit)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be processed without doing it')
    parser.add_argument('--dir', type=str, default=None,
                        help='Process specific directory only')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers')

    args = parser.parse_args()

    # Determine directories
    if args.dir:
        directories = [Path(args.dir)]
    else:
        directories = PDF_DIRS

    # Find all PDFs
    logger.info(f"Scanning directories: {[str(d) for d in directories]}")
    all_pdfs = find_all_pdfs(directories)
    logger.info(f"Found {len(all_pdfs)} PDFs")

    # Get already processed docs
    conn = get_db_connection()
    processed_docs = get_docs_with_passages(conn)
    logger.info(f"Already have passages for {len(processed_docs)} documents")
    conn.close()

    # Apply limit
    if not args.all:
        all_pdfs = all_pdfs[:args.limit]

    logger.info(f"Will process {len(all_pdfs)} PDFs")

    if args.dry_run:
        for pdf in all_pdfs[:20]:
            logger.info(f"Would process: {pdf.name}")
        if len(all_pdfs) > 20:
            logger.info(f"... and {len(all_pdfs) - 20} more")
        return

    # Process PDFs in parallel
    total_passages = 0
    processed = 0
    errors = 0
    completed = 0

    start_time = time.time()
    num_workers = args.workers

    logger.info(f"Starting parallel processing with {num_workers} workers...")

    # Use ProcessPoolExecutor for CPU-bound PDF parsing
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all jobs
        future_to_pdf = {executor.submit(process_pdf, pdf_path): pdf_path for pdf_path in all_pdfs}

        # Process results as they complete
        for future in as_completed(future_to_pdf):
            pdf_path = future_to_pdf[future]
            completed += 1

            try:
                count, error = future.result()
                if error:
                    if 'Already has' not in error:
                        logger.warning(f"[{completed}/{len(all_pdfs)}] {pdf_path.name}: {error}")
                        errors += 1
                else:
                    total_passages += count
                    processed += 1
                    if count > 0:
                        logger.info(f"[{completed}/{len(all_pdfs)}] {pdf_path.name}: {count} passages")
            except Exception as e:
                logger.error(f"[{completed}/{len(all_pdfs)}] {pdf_path.name}: {e}")
                errors += 1

            # Progress update every 50 PDFs
            if completed % 50 == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed
                remaining = (len(all_pdfs) - completed) / rate if rate > 0 else 0
                logger.info(f"Progress: {completed}/{len(all_pdfs)} ({rate:.1f} PDFs/sec, ~{remaining/60:.1f} min remaining)")

    elapsed = time.time() - start_time

    # Final stats
    logger.info(f"\n{'='*60}")
    logger.info(f"BATCH POPULATION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"PDFs processed: {processed}")
    logger.info(f"Errors: {errors}")
    logger.info(f"Total passages created: {total_passages}")
    logger.info(f"Time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    logger.info(f"Rate: {len(all_pdfs)/elapsed:.2f} PDFs/second")

    # Check final count
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM passages")
    final_count = cursor.fetchone()[0]
    conn.close()

    logger.info(f"Total passages in database: {final_count}")


if __name__ == '__main__':
    main()
