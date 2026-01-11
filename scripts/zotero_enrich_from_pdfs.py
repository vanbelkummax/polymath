#!/usr/bin/env python3
"""
Enrich Zotero items by extracting DOIs from PDFs and looking up metadata via Crossref.

Usage:
    python3 scripts/zotero_enrich_from_pdfs.py --dry-run
    python3 scripts/zotero_enrich_from_pdfs.py
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import httpx

# Load .env
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

from pyzotero import zotero

ZOTERO_API_KEY = os.getenv("ZOTERO_API_KEY")
ZOTERO_USER_ID = os.getenv("ZOTERO_USER_ID")
PDF_DIR = Path("/scratch/polymath_archive/pdfs")
LOG_FILE = Path("/home/user/work/polymax/logs/zotero_enrich_pdfs.log")
CACHE_FILE = Path("/home/user/work/polymax/logs/doi_cache.json")

LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

DOI_PATTERN = re.compile(r'10\.\d{4,}/[^\s\]>"\']+')


def log(msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def extract_doi_from_pdf(pdf_path: Path) -> str:
    """Extract DOI from PDF text."""
    try:
        result = subprocess.run(
            ['pdftotext', '-f', '1', '-l', '2', str(pdf_path), '-'],
            capture_output=True, text=True, timeout=15
        )
        dois = DOI_PATTERN.findall(result.stdout[:8000])
        if dois:
            # Clean up DOI
            doi = dois[0].rstrip('.,;:)]}')
            return doi
    except:
        pass
    return None


def lookup_crossref(doi: str) -> dict:
    """Look up metadata from Crossref."""
    try:
        url = f"https://api.crossref.org/works/{doi}"
        headers = {"User-Agent": "Polymath/1.0 (mailto:research@example.com)"}

        with httpx.Client(timeout=10) as client:
            resp = client.get(url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()['message']

                # Extract authors
                authors = []
                for author in data.get('author', []):
                    given = author.get('given', '')
                    family = author.get('family', '')
                    if given and family:
                        authors.append({'firstName': given, 'lastName': family, 'creatorType': 'author'})
                    elif family:
                        authors.append({'name': family, 'creatorType': 'author'})

                # Extract year
                year = None
                if 'published-print' in data:
                    year = data['published-print']['date-parts'][0][0]
                elif 'published-online' in data:
                    year = data['published-online']['date-parts'][0][0]
                elif 'created' in data:
                    year = data['created']['date-parts'][0][0]

                return {
                    'title': data.get('title', [''])[0],
                    'creators': authors,
                    'DOI': doi,
                    'date': str(year) if year else None,
                    'publicationTitle': data.get('container-title', [''])[0],
                    'volume': data.get('volume'),
                    'issue': data.get('issue'),
                    'pages': data.get('page'),
                    'abstractNote': data.get('abstract', '').replace('<jats:p>', '').replace('</jats:p>', ''),
                    'url': data.get('URL'),
                }
    except Exception as e:
        pass
    return None


def get_zotero_client():
    if not ZOTERO_API_KEY or not ZOTERO_USER_ID:
        log("Missing ZOTERO_API_KEY or ZOTERO_USER_ID")
        sys.exit(1)
    return zotero.Zotero(ZOTERO_USER_ID, 'user', ZOTERO_API_KEY)


def find_collection(zot, name: str) -> str:
    for c in zot.collections():
        if c['data']['name'] == name:
            return c['key']
    return None


def load_doi_cache() -> dict:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}


def save_doi_cache(cache: dict):
    CACHE_FILE.write_text(json.dumps(cache))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--collection", default="Polymath_Full")
    parser.add_argument("--limit", type=int, default=0, help="Limit items to process (0=all)")
    args = parser.parse_args()

    log("=" * 60)
    log("Zotero PDF → DOI → Crossref Enrichment")
    log("=" * 60)

    # Load DOI cache
    doi_cache = load_doi_cache()
    log(f"Loaded {len(doi_cache)} cached DOI lookups")

    # Connect to Zotero
    zot = get_zotero_client()
    collection_key = find_collection(zot, args.collection)
    if not collection_key:
        log(f"Collection '{args.collection}' not found")
        sys.exit(1)

    # Get all items
    log("Fetching Zotero items...")
    items = zot.everything(zot.collection_items(collection_key))
    log(f"Found {len(items)} items")

    # Filter to items needing enrichment (no DOI)
    to_process = []
    for item in items:
        data = item['data']
        if data.get('itemType') == 'attachment':
            continue
        if data.get('DOI'):  # Already has DOI
            continue

        # Get filename from extra
        filename = None
        for line in data.get('extra', '').split('\n'):
            if line.startswith('Filename:'):
                filename = line.split(':', 1)[1].strip()
                break

        if filename:
            to_process.append((item, filename))

    log(f"Items needing enrichment: {len(to_process)}")

    if args.limit:
        to_process = to_process[:args.limit]
        log(f"Limited to {args.limit} items")

    # Extract DOIs from PDFs
    log("\nExtracting DOIs from PDFs...")
    filename_to_doi = {}

    def extract_for_file(filename):
        pdf_path = PDF_DIR / filename
        if pdf_path.exists():
            doi = extract_doi_from_pdf(pdf_path)
            return filename, doi
        return filename, None

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(extract_for_file, f): f for _, f in to_process}
        for i, future in enumerate(as_completed(futures)):
            filename, doi = future.result()
            if doi:
                filename_to_doi[filename] = doi
            if (i + 1) % 200 == 0:
                log(f"  Extracted {i+1}/{len(to_process)}, found {len(filename_to_doi)} DOIs")

    log(f"Found DOIs in {len(filename_to_doi)} PDFs")

    # Look up metadata from Crossref
    log("\nLooking up metadata from Crossref...")
    updated = 0
    no_doi = 0
    lookup_failed = 0

    for i, (item, filename) in enumerate(to_process):
        doi = filename_to_doi.get(filename)

        if not doi:
            no_doi += 1
            continue

        # Check cache
        if doi in doi_cache:
            metadata = doi_cache[doi]
        else:
            metadata = lookup_crossref(doi)
            doi_cache[doi] = metadata
            time.sleep(0.1)  # Rate limit Crossref

        if not metadata:
            lookup_failed += 1
            continue

        if args.dry_run:
            log(f"Would update: {filename[:40]} → {metadata.get('title', 'N/A')[:40]}")
            updated += 1
        else:
            try:
                # Update item
                item_data = item['data']
                for key, value in metadata.items():
                    if value:
                        item_data[key] = value

                zot.update_item(item)
                updated += 1

                if updated % 50 == 0:
                    log(f"Progress: {updated} updated")
                    save_doi_cache(doi_cache)

                time.sleep(0.2)  # Zotero rate limit

            except Exception as e:
                log(f"Error updating {filename}: {e}")

        if (i + 1) % 200 == 0:
            log(f"Processed {i+1}/{len(to_process)}")

    # Save cache
    save_doi_cache(doi_cache)

    log(f"\n{'DRY RUN - ' if args.dry_run else ''}Summary:")
    log(f"  Updated: {updated}")
    log(f"  No DOI in PDF: {no_doi}")
    log(f"  Crossref lookup failed: {lookup_failed}")


if __name__ == "__main__":
    main()
