#!/usr/bin/env python3
"""
Enrich PDFs using LLM (Gemini) to extract metadata from first page.

Extracts title, authors, year, abstract, and attempts DOI/arXiv ID detection.

Usage:
    python3 scripts/enrich_with_llm.py --dry-run --limit 10
    python3 scripts/enrich_with_llm.py --limit 100
    python3 scripts/enrich_with_llm.py  # Process all
"""

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load .env
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

import google.generativeai as genai

CITATIONS_FILE = Path("/home/user/polymath-repo/data/citations/pdf_citations.json")
NEEDS_REVIEW_FILE = Path("/home/user/polymath-repo/data/citations/needs_review.json")
PDF_DIR = Path("/scratch/polymath_archive/pdfs")
LOG_FILE = Path("/home/user/work/polymax/logs/enrich_llm.log")
CACHE_FILE = Path("/home/user/work/polymax/logs/llm_extraction_cache.json")

LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash-exp')

EXTRACTION_PROMPT = """Extract bibliographic metadata from this academic paper's first page text.

Return ONLY valid JSON with these fields (use null for unknown):
{
  "title": "Full paper title",
  "authors": ["Author One", "Author Two"],
  "year": 2024,
  "journal": "Journal or Conference Name",
  "doi": "10.xxxx/xxxxx",
  "arxiv_id": "2401.12345",
  "abstract": "First 500 chars of abstract if visible"
}

IMPORTANT:
- Extract the EXACT title as written (preserve capitalization)
- Authors should be full names, not abbreviations
- Year should be publication year (integer)
- DOI format: 10.xxxx/xxxxx (no URL prefix)
- arXiv ID format: YYMM.NNNNN
- If abstract is cut off, include what's visible

TEXT FROM FIRST PAGE:
"""


def log(msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def extract_text_from_pdf(pdf_path: Path, pages: int = 2) -> str:
    """Extract text from first N pages of PDF."""
    try:
        result = subprocess.run(
            ['pdftotext', '-f', '1', '-l', str(pages), str(pdf_path), '-'],
            capture_output=True, text=True, timeout=30
        )
        return result.stdout[:8000]  # Limit to ~8K chars
    except Exception as e:
        return None


def extract_metadata_with_llm(text: str) -> dict:
    """Use Gemini to extract metadata from PDF text."""
    try:
        response = model.generate_content(
            EXTRACTION_PROMPT + text,
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=1024,
            )
        )

        # Parse JSON from response
        response_text = response.text.strip()

        # Handle markdown code blocks
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0]
        elif '```' in response_text:
            response_text = response_text.split('```')[1].split('```')[0]

        data = json.loads(response_text)

        # Validate and clean
        result = {
            'title': data.get('title'),
            'authors': data.get('authors') if isinstance(data.get('authors'), list) else None,
            'year': int(data['year']) if data.get('year') and str(data['year']).isdigit() else None,
            'journal': data.get('journal'),
            'doi': data.get('doi'),
            'arxiv_id': data.get('arxiv_id'),
            'abstract': data.get('abstract'),
            'source': 'gemini_extraction'
        }

        # Only return if we got at least a title
        if result.get('title') and len(result['title']) > 5:
            return result

    except json.JSONDecodeError as e:
        pass
    except Exception as e:
        pass

    return None


def load_cache() -> dict:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}


def save_cache(cache: dict):
    CACHE_FILE.write_text(json.dumps(cache, indent=2))


def process_pdf(filename: str, cache: dict) -> tuple:
    """Process a single PDF and return (filename, metadata)."""
    # Check cache
    if filename in cache:
        return filename, cache[filename]

    pdf_path = PDF_DIR / filename
    if not pdf_path.exists():
        return filename, None

    # Extract text
    text = extract_text_from_pdf(pdf_path)
    if not text or len(text) < 100:
        return filename, None

    # Extract metadata with LLM
    metadata = extract_metadata_with_llm(text)

    # Cache result
    cache[filename] = metadata

    return filename, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--delay", type=float, default=0.2, help="Delay between API calls")
    args = parser.parse_args()

    log("=" * 60)
    log("LLM-based PDF Metadata Extraction (Gemini)")
    log("=" * 60)

    # Load data
    citations = json.loads(CITATIONS_FILE.read_text())
    needs_review = json.loads(NEEDS_REVIEW_FILE.read_text())
    cache = load_cache()

    log(f"Loaded {len(cache)} cached extractions")

    if args.limit:
        needs_review = needs_review[:args.limit]

    log(f"Items to process: {len(needs_review)}")

    enriched = 0
    failed = 0
    cached_hits = 0

    for i, item in enumerate(needs_review):
        filename = item['filename']

        # Check if already has good metadata
        if citations.get(filename, {}).get('confidence', 0) >= 0.75:
            continue

        # Process
        filename, metadata = process_pdf(filename, cache)

        if metadata:
            if metadata == cache.get(filename):
                cached_hits += 1

            if args.dry_run:
                log(f"Would enrich: {filename[:40]} → {metadata.get('title', 'N/A')[:40]}")
            else:
                # Update citations
                citations[filename].update(metadata)
                citations[filename]['confidence'] = 0.8

            enriched += 1
        else:
            failed += 1

        # Progress logging
        if (i + 1) % 50 == 0:
            log(f"Progress: {i+1}/{len(needs_review)} | Enriched: {enriched} | Failed: {failed} | Cached: {cached_hits}")
            if not args.dry_run:
                save_cache(cache)

        # Rate limiting
        time.sleep(args.delay)

    # Save results
    if not args.dry_run:
        save_cache(cache)

        if enriched > 0:
            CITATIONS_FILE.write_text(json.dumps(citations, indent=2))

            # Update needs_review
            still_needs = [c for c in citations.values() if c.get('confidence', 0) < 0.5]
            NEEDS_REVIEW_FILE.write_text(json.dumps(still_needs, indent=2))

            log(f"Saved updates. Still needs review: {len(still_needs)}")

    log("=" * 60)
    log(f"{'DRY RUN - ' if args.dry_run else ''}COMPLETE")
    log(f"  Processed: {len(needs_review)}")
    log(f"  Enriched: {enriched}")
    log(f"  Failed: {failed}")
    log(f"  Cache hits: {cached_hits}")
    log("=" * 60)


if __name__ == "__main__":
    main()
