#!/usr/bin/env python3
"""
Enrich remaining PDFs without metadata using multiple strategies:
1. arXiv API - for preprints with arXiv IDs in filename
2. Semantic Scholar API - free, good for CS/ML papers
3. Title extraction + search - extract title from PDF, search APIs

Usage:
    python3 scripts/enrich_remaining.py --dry-run
    python3 scripts/enrich_remaining.py --strategy arxiv
    python3 scripts/enrich_remaining.py --strategy s2
    python3 scripts/enrich_remaining.py --all
"""

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from datetime import datetime
import httpx

# Semantic Scholar API key (1 req/sec with key vs 100 req/5min without)
S2_API_KEY = os.getenv("SEMANTICSCHOLAR_API_KEY", "")

CITATIONS_FILE = Path("/home/user/polymath-repo/data/citations/pdf_citations.json")
NEEDS_REVIEW_FILE = Path("/home/user/polymath-repo/data/citations/needs_review.json")
PDF_DIR = Path("/scratch/polymath_archive/pdfs")
LOG_FILE = Path("/home/user/work/polymax/logs/enrich_remaining.log")

LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def log(msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def extract_arxiv_id(filename: str) -> str:
    """Extract arXiv ID from filename."""
    # Pattern: YYMM.NNNNN or YYMM.NNNNNvN
    match = re.match(r'^(\d{4}\.\d{4,5})(v\d+)?', filename)
    if match:
        return match.group(1)
    return None


def lookup_arxiv(arxiv_id: str) -> dict:
    """Look up paper metadata from arXiv API."""
    try:
        url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
        with httpx.Client(timeout=10) as client:
            resp = client.get(url)
            if resp.status_code == 200:
                text = resp.text

                # Parse XML (simple regex extraction)
                title_match = re.search(r'<title>(.+?)</title>', text, re.DOTALL)
                authors = re.findall(r'<name>(.+?)</name>', text)
                abstract_match = re.search(r'<summary>(.+?)</summary>', text, re.DOTALL)
                published_match = re.search(r'<published>(\d{4})', text)
                doi_match = re.search(r'doi.org/([^<"]+)', text)

                if title_match:
                    title = title_match.group(1).strip().replace('\n', ' ')
                    # Skip if it's the "Error" title
                    if title.lower().startswith('error'):
                        return None

                    return {
                        'title': title,
                        'authors': authors,
                        'year': int(published_match.group(1)) if published_match else None,
                        'arxiv_id': arxiv_id,
                        'doi': doi_match.group(1) if doi_match else None,
                        'abstract': abstract_match.group(1).strip().replace('\n', ' ')[:2000] if abstract_match else None,
                        'url': f"https://arxiv.org/abs/{arxiv_id}",
                        'source': 'arxiv'
                    }
    except Exception as e:
        pass
    return None


def extract_title_from_pdf(pdf_path: Path) -> str:
    """Extract likely title from first page of PDF."""
    try:
        result = subprocess.run(
            ['pdftotext', '-f', '1', '-l', '1', str(pdf_path), '-'],
            capture_output=True, text=True, timeout=10
        )
        text = result.stdout

        # Get first few non-empty lines (likely title)
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if lines:
            # Title is usually in first 1-3 lines, concatenate if short
            title_parts = []
            for line in lines[:5]:
                # Skip lines that look like headers/footers
                if any(x in line.lower() for x in ['arxiv', 'preprint', 'submitted', 'accepted', 'published']):
                    continue
                if re.match(r'^\d+$', line):  # Just a number
                    continue
                if len(line) < 10:  # Too short
                    continue
                title_parts.append(line)
                if len(' '.join(title_parts)) > 50:
                    break

            if title_parts:
                return ' '.join(title_parts)[:200]
    except:
        pass
    return None


def lookup_semantic_scholar(title: str) -> dict:
    """Look up paper by title on Semantic Scholar."""
    try:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            'query': title[:200],
            'limit': 1,
            'fields': 'title,authors,year,abstract,externalIds,url'
        }
        headers = {}
        if S2_API_KEY:
            headers['x-api-key'] = S2_API_KEY

        with httpx.Client(timeout=15) as client:
            resp = client.get(url, params=params, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                if data.get('data'):
                    paper = data['data'][0]

                    # Check if title matches reasonably
                    found_title = paper.get('title', '').lower()
                    query_title = title.lower()
                    # Simple word overlap check
                    query_words = set(query_title.split())
                    found_words = set(found_title.split())
                    overlap = len(query_words & found_words) / max(len(query_words), 1)

                    if overlap < 0.5:  # Title doesn't match well enough
                        return None

                    authors = [a.get('name') for a in paper.get('authors', [])]
                    ext_ids = paper.get('externalIds', {})

                    return {
                        'title': paper.get('title'),
                        'authors': authors,
                        'year': paper.get('year'),
                        'doi': ext_ids.get('DOI'),
                        'arxiv_id': ext_ids.get('ArXiv'),
                        'pmid': ext_ids.get('PubMed'),
                        'abstract': paper.get('abstract', '')[:2000] if paper.get('abstract') else None,
                        'url': paper.get('url'),
                        'source': 'semantic_scholar'
                    }
    except Exception as e:
        pass
    return None


def enrich_with_arxiv(needs_review: list, citations: dict, dry_run: bool = False) -> int:
    """Enrich items that have arXiv IDs in filename."""
    log("=== Strategy 1: arXiv API ===")

    enriched = 0
    for item in needs_review:
        filename = item['filename']
        arxiv_id = extract_arxiv_id(filename)

        if not arxiv_id:
            continue

        meta = lookup_arxiv(arxiv_id)
        if meta:
            if dry_run:
                log(f"Would enrich: {filename} → {meta['title'][:50]}")
            else:
                citations[filename].update(meta)
                citations[filename]['confidence'] = 0.9
            enriched += 1
            time.sleep(0.5)  # Rate limit

    log(f"arXiv: {enriched} items enriched")
    return enriched


def enrich_with_semantic_scholar(needs_review: list, citations: dict, dry_run: bool = False) -> int:
    """Enrich items by extracting title and searching Semantic Scholar."""
    log("=== Strategy 2: Semantic Scholar ===")
    log(f"API key: {'present' if S2_API_KEY else 'missing (rate limited)'}")

    enriched = 0
    for i, item in enumerate(needs_review):
        filename = item['filename']
        pdf_path = PDF_DIR / filename

        if not pdf_path.exists():
            continue

        # Skip if already has good data
        if citations.get(filename, {}).get('confidence', 0) >= 0.75:
            continue

        # Extract title from PDF
        title = extract_title_from_pdf(pdf_path)
        if not title:
            continue

        # Search Semantic Scholar
        meta = lookup_semantic_scholar(title)
        if meta:
            if dry_run:
                log(f"Would enrich: {filename} → {meta['title'][:50]}")
            else:
                citations[filename].update(meta)
                citations[filename]['confidence'] = 0.85
            enriched += 1

        # Rate limit: 1 req/sec with API key, 100 req/5min without
        time.sleep(1.1 if S2_API_KEY else 3)

        if (i + 1) % 50 == 0:
            log(f"Progress: {i+1}/{len(needs_review)}, enriched: {enriched}")

    log(f"Semantic Scholar: {enriched} items enriched")
    return enriched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--strategy", choices=['arxiv', 's2', 'all'], default='all')
    parser.add_argument("--limit", type=int, default=0, help="Limit items to process")
    args = parser.parse_args()

    log("=" * 60)
    log("Enriching Remaining PDFs")
    log("=" * 60)

    # Load data
    citations = json.loads(CITATIONS_FILE.read_text())
    needs_review = json.loads(NEEDS_REVIEW_FILE.read_text())

    if args.limit:
        needs_review = needs_review[:args.limit]

    log(f"Items needing review: {len(needs_review)}")

    total_enriched = 0

    if args.strategy in ['arxiv', 'all']:
        total_enriched += enrich_with_arxiv(needs_review, citations, args.dry_run)

    if args.strategy in ['s2', 'all']:
        total_enriched += enrich_with_semantic_scholar(needs_review, citations, args.dry_run)

    # Save updated citations
    if not args.dry_run and total_enriched > 0:
        CITATIONS_FILE.write_text(json.dumps(citations, indent=2))

        # Update needs_review
        still_needs = [c for c in citations.values() if c.get('confidence', 0) < 0.5]
        NEEDS_REVIEW_FILE.write_text(json.dumps(still_needs, indent=2))

        log(f"\nSaved updates. Still needs review: {len(still_needs)}")

    log(f"\n{'DRY RUN - ' if args.dry_run else ''}Total enriched: {total_enriched}")


if __name__ == "__main__":
    main()
