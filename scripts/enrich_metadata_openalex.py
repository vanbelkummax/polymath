#!/usr/bin/env python3
"""
Metadata Enrichment via OpenAlex - Fills DOI/PMID gaps using title matching.

Target: 84.4% of documents missing DOI (3,046 of 3,602 docs with passages)

Strategy:
1. Query OpenAlex API by title (fuzzy match)
2. Match by normalized title similarity (threshold: 0.85)
3. Update doi, pmid, openalex_id columns

Usage:
  python3 scripts/enrich_metadata_openalex.py --batch 100 --dry-run
  python3 scripts/enrich_metadata_openalex.py --batch 100
"""
import sys
import time
import argparse
import logging
import requests
from pathlib import Path
from difflib import SequenceMatcher
from typing import Optional, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))
import psycopg2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OPENALEX_API = "https://api.openalex.org/works"
USER_AGENT = "mailto:max.vanbelkum@vanderbilt.edu"  # Required by OpenAlex


def normalize_title(title: str) -> str:
    """Normalize title for comparison."""
    import re
    t = title.lower()
    t = re.sub(r'[^a-z0-9\s]', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def title_similarity(t1: str, t2: str) -> float:
    """Calculate similarity between two titles."""
    n1 = normalize_title(t1)
    n2 = normalize_title(t2)
    return SequenceMatcher(None, n1, n2).ratio()


def clean_title_for_search(title: str) -> str:
    """Clean title for API search - remove OCR artifacts, special chars."""
    import re
    # Strip leading lowercase words (usually truncated)
    title = re.sub(r'^[a-z][a-z\s]{0,30}(?=[A-Z])', '', title)
    # Remove only the chars that break OpenAlex (not colons!)
    title = re.sub(r'[%&;{}[\]<>|]', ' ', title)
    # Collapse whitespace
    title = re.sub(r'\s+', ' ', title).strip()
    # Take only first 150 chars for search
    return title[:150]


def search_openalex(title: str, year: Optional[int] = None, year_window: int = 2, min_score: float = 0.85) -> Optional[Dict[str, Any]]:
    """
    Search OpenAlex for a paper by title.

    Returns best match with DOI, PMID, OpenAlex ID if similarity > threshold.
    """
    clean_title = clean_title_for_search(title)
    if len(clean_title) < 15:
        return None

    # Use search endpoint instead of filter (more lenient)
    params = {
        "search": clean_title,  # Full-text search instead of filter
        "per_page": 5,
        "select": "id,doi,ids,title,publication_year"
    }

    if year:
        # Allow ±year_window tolerance (year in DB can be wrong)
        params["filter"] = f"publication_year:{year-year_window}-{year+year_window}"

    headers = {"User-Agent": USER_AGENT}

    try:
        resp = requests.get(OPENALEX_API, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", [])
        if not results:
            return None

        # Find best match by title similarity
        best_match = None
        best_score = 0

        for work in results:
            work_title = work.get("title", "")
            if not work_title:
                continue

            score = title_similarity(title, work_title)
            if score > best_score:
                best_score = score
                best_match = work

        if best_match and best_score >= min_score:
            ids = best_match.get("ids", {})
            return {
                "openalex_id": best_match.get("id", "").replace("https://openalex.org/", ""),
                "doi": ids.get("doi", "").replace("https://doi.org/", "") if ids.get("doi") else None,
                "pmid": ids.get("pmid", "").replace("https://pubmed.ncbi.nlm.nih.gov/", "") if ids.get("pmid") else None,
                "matched_title": best_match.get("title"),
                "similarity": best_score
            }

        return None

    except Exception as e:
        logger.warning(f"OpenAlex API error: {e}")
        return None


def is_valid_title(title: str) -> bool:
    """Check if title looks like a real paper title (not OCR garbage)."""
    if not title or len(title) < 20:
        return False
    # Too many special chars = OCR garbage
    alpha_ratio = sum(c.isalpha() or c.isspace() for c in title) / len(title)
    if alpha_ratio < 0.75:
        return False
    # Title should have some uppercase letters (real titles have capitals)
    if not any(c.isupper() for c in title[:50]):
        return False
    # Avoid truncated titles
    if title.endswith(',') or title.endswith('&') or title.endswith('('):
        return False
    # Skip common false positives
    garbage_markers = [
        'HHS Public Access', 'These authors contributed equally',
        'Corresponding author', 'Supplementary', 'won the Nobel',
        'DOI:', 'http://', 'PMID:', 'ChineseAcademyofSciences'
    ]
    title_lower = title.lower()
    for marker in garbage_markers:
        if marker.lower() in title_lower:
            return False
    # Must have at least 3 words
    if len(title.split()) < 3:
        return False
    return True


def get_docs_missing_doi(conn, limit: int = 100, offset: int = 0) -> list:
    """Get documents with passages but missing DOI - prioritize good titles."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT doc_id, title, year FROM (
            SELECT DISTINCT d.doc_id, d.title, d.year, LENGTH(d.title) as title_len
            FROM documents d
            INNER JOIN passages p ON d.doc_id = p.doc_id
            WHERE d.doi IS NULL
              AND d.title IS NOT NULL
              AND LENGTH(d.title) > 40
              AND d.title ~ '^[A-Z]'
              AND d.year IS NOT NULL
              AND (LENGTH(REGEXP_REPLACE(d.title, '[^a-zA-Z ]', '', 'g'))::float
                   / NULLIF(LENGTH(d.title), 0)) > 0.75
        ) sq
        ORDER BY title_len DESC
        LIMIT %(limit)s OFFSET %(offset)s
    """, {'limit': limit, 'offset': offset})
    return cursor.fetchall()


def update_document_metadata(conn, doc_id: str, doi: str = None, pmid: str = None, openalex_id: str = None) -> bool:
    """Update document with resolved identifiers."""
    cursor = conn.cursor()
    try:
        cursor.execute("""
            UPDATE documents
            SET doi = COALESCE(%s, doi),
                pmid = COALESCE(%s, pmid),
                openalex_id = COALESCE(%s, openalex_id)
            WHERE doc_id = %s
        """, (doi, pmid, openalex_id, doc_id))
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to update {doc_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Enrich metadata via OpenAlex")
    parser.add_argument("--batch", type=int, default=100, help="Batch size")
    parser.add_argument("--offset", type=int, default=0, help="Starting offset")
    parser.add_argument("--max-docs", type=int, help="Max documents to process")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between API calls (seconds)")
    parser.add_argument("--dry-run", action="store_true", help="Don't update database")
    parser.add_argument("--min-score", type=float, default=0.85, help="Min title similarity threshold (0-1)")
    parser.add_argument("--year-window", type=int, default=2, help="Year tolerance ±N years")
    parser.add_argument("--log-file", type=str, help="Log changes to CSV (doc_id,title,old_doi,new_doi,score)")
    args = parser.parse_args()

    conn = psycopg2.connect("dbname=polymath user=polymath host=/var/run/postgresql")

    offset = args.offset
    total_processed = 0
    total_enriched = 0
    total_not_found = 0

    print(f"Starting metadata enrichment (batch={args.batch}, offset={offset})")
    print(f"Thresholds: min_score={args.min_score}, year_window=±{args.year_window}")
    if args.log_file:
        print(f"Logging changes to: {args.log_file}")
        # Write CSV header
        with open(args.log_file, 'w') as f:
            f.write("doc_id,title,old_doi,new_doi,new_pmid,score\n")
    print("-" * 60)

    while True:
        docs = get_docs_missing_doi(conn, args.batch, offset)
        if not docs:
            print("\nNo more documents to process")
            break

        for doc_id, title, year in docs:
            # Skip OCR garbage titles
            if not is_valid_title(title):
                continue

            total_processed += 1

            if args.max_docs and total_processed > args.max_docs:
                break

            # Truncate title for display
            title_short = title[:60] + "..." if len(title) > 60 else title
            print(f"[{total_processed}] {title_short}", end=" ")

            # Search OpenAlex
            result = search_openalex(title, year, year_window=args.year_window, min_score=args.min_score)

            if result:
                doi = result.get("doi")
                pmid = result.get("pmid")
                openalex_id = result.get("openalex_id")
                similarity = result.get("similarity", 0)

                # Log change before update
                if args.log_file and not args.dry_run:
                    with open(args.log_file, 'a') as f:
                        # CSV: doc_id,title,old_doi,new_doi,new_pmid,score
                        safe_title = title.replace(',', ';').replace('\n', ' ')[:100]
                        f.write(f"{doc_id},{safe_title},,{doi or ''},{pmid or ''},{similarity:.3f}\n")

                if not args.dry_run:
                    update_document_metadata(conn, doc_id, doi, pmid, openalex_id)

                total_enriched += 1
                print(f"✓ DOI={doi or 'N/A'} PMID={pmid or 'N/A'} ({similarity:.2f})")
            else:
                total_not_found += 1
                print("✗ Not found")

            time.sleep(args.delay)

        if args.max_docs and total_processed >= args.max_docs:
            break

        offset += args.batch

    conn.close()

    print("\n" + "=" * 60)
    print("ENRICHMENT COMPLETE")
    print(f"Processed: {total_processed}")
    print(f"Enriched: {total_enriched} ({100*total_enriched/max(1,total_processed):.1f}%)")
    print(f"Not found: {total_not_found}")
    if args.dry_run:
        print("(DRY RUN - no changes made)")


if __name__ == "__main__":
    main()
