#!/usr/bin/env python3
"""
Identify high-priority metadata-only documents for PDF acquisition.
Usage: python3 scripts/identify_metadata_targets.py --output /path/to/TARGETS.csv --limit 50
"""
import csv
import re
import psycopg2
from pathlib import Path

FACULTY_PATTERNS = [
    r'\bhwang\b', r'\blandman\b', r'\blau\b', r'\bsarkar\b',
    r'\bvanderbilt\b'
]

TOPIC_KEYWORDS = [
    'spatial transcriptomics', 'single-cell', 'computational pathology',
    'holotomography', 'foundation model', 'image registration',
    'MRI', 'brain imaging', 'segmentation', 'colorectal', 'CRC',
    'spatial biology', 'multimodal', 'deep learning pathology',
    'gene expression prediction', 'H&E', 'histology', 'histopathology',
    'whole slide', 'digital pathology', 'tumor microenvironment',
    'immunotherapy', 'cancer', 'attention mechanism', 'transformer',
    'graph neural', 'convolutional', 'self-supervised', 'contrastive learning',
    'vision transformer', 'multi-scale', 'multi-resolution'
]

def score_document(row):
    """Score a metadata-only doc by relevance to target faculty."""
    score = 0
    reasons = []

    authors = (row['authors'] or '').lower()
    title = (row['title'] or '').lower()

    # Author matches (highest priority)
    for pattern in FACULTY_PATTERNS[:4]:  # hwang, landman, lau, sarkar
        if re.search(pattern, authors):
            score += 100
            reasons.append(f"author:{pattern.strip(chr(92)+'b')}")

    # Vanderbilt affiliation
    if 'vanderbilt' in authors:
        score += 50
        reasons.append("affiliation:vanderbilt")

    # Topic keyword matches
    for kw in TOPIC_KEYWORDS:
        if kw.lower() in title:
            score += 20
            reasons.append(f"topic:{kw}")

    # Has DOI/PMID (easier to acquire)
    if row['doi']:
        score += 10
        reasons.append("has_doi")
    if row['pmid']:
        score += 10
        reasons.append("has_pmid")

    # Recency bonus
    if row['year'] and str(row['year']).isdigit() and int(row['year']) >= 2020:
        score += 5
        reasons.append("recent")

    return score, '|'.join(reasons) if reasons else 'none'

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int, default=50)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    conn = psycopg2.connect(dbname='polymath', user='polymath')
    cur = conn.cursor()

    # Get metadata-only docs
    cur.execute("""
        SELECT
            d.doc_id::text,
            d.title,
            d.year,
            d.doi,
            d.pmid,
            array_to_string(d.authors, '; ') as authors,
            d.title_hash
        FROM documents d
        LEFT JOIN passages p ON d.doc_id = p.doc_id
        WHERE p.passage_id IS NULL
          AND d.title IS NOT NULL
        ORDER BY d.year DESC NULLS LAST
    """)

    rows = [dict(zip(['doc_id', 'title', 'year', 'doi', 'pmid', 'authors', 'title_hash'], r))
            for r in cur.fetchall()]

    # Score and rank
    scored = []
    for row in rows:
        score, reason = score_document(row)
        if score > 0:
            scored.append({**row, 'score': score, 'matched_reason': reason})

    scored.sort(key=lambda x: -x['score'])
    top = scored[:args.limit]

    print(f"Found {len(scored)} relevant metadata-only docs")
    print(f"Top {len(top)} selected for acquisition")

    if args.dry_run:
        for i, doc in enumerate(top[:10], 1):
            print(f"  {i}. [{doc['score']}] {doc['title'][:60]}...")
        return

    # Write to CSV
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['rank', 'doc_id', 'title', 'year', 'doi', 'pmid', 'matched_reason', 'score'])
        for i, doc in enumerate(top, 1):
            writer.writerow([i, doc['doc_id'], doc['title'], doc['year'],
                           doc['doi'], doc['pmid'], doc['matched_reason'], doc['score']])

    print(f"Written to {args.output}")
    conn.close()

if __name__ == '__main__':
    main()
