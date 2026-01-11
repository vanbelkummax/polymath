#!/usr/bin/env python3
"""
PDF Metadata Enrichment Pipeline

Extracts metadata from PDFs using:
1. Filename pattern matching (fast path for arXiv, DOI, ScienceDirect)
2. GROBID header extraction (for unknown patterns)
3. Crossref/OpenAlex API enrichment

Usage:
    # Start GROBID first:
    docker run --rm --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.8.2-crf

    # Then run:
    python3 scripts/enrich_pdf_metadata.py --pdf-dir /scratch/polymath_archive/pdfs/
    python3 scripts/enrich_pdf_metadata.py --pdf-dir /scratch/polymath_archive/pdfs/ --apply
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import psycopg2
from psycopg2.extras import execute_values

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.config import POSTGRES_DSN

GROBID_URL = os.getenv("GROBID_URL", "http://localhost:8070")
CROSSREF_EMAIL = os.getenv("CROSSREF_EMAIL", "polymath@vanderbilt.edu")
OPENALEX_EMAIL = os.getenv("OPENALEX_EMAIL", "polymath@vanderbilt.edu")


@dataclass
class PaperMetadata:
    path: str
    sha1: str
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    year: Optional[int] = None
    venue: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    pmid: Optional[str] = None
    abstract: Optional[str] = None
    source: str = "unknown"
    confidence: float = 0.0
    match_explain: str = ""
    needs_review: bool = False


def compute_sha1(filepath: Path) -> str:
    """Compute SHA1 hash of file."""
    h = hashlib.sha1()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def extract_id_from_filename(filename: str) -> Optional[Dict]:
    """Extract identifiers from PDF filename patterns."""
    basename = Path(filename).stem

    # arXiv new format: 2301.12345v2 → arXiv:2301.12345
    if match := re.match(r'^(\d{4}\.\d{4,5})(v\d+)?$', basename):
        return {'arxiv_id': match.group(1), 'source': 'filename_arxiv'}

    # arXiv old format: 0712598v1 or 0712.1598v4
    if match := re.match(r'^(\d{7})(v\d+)?$', basename):
        return {'arxiv_id': basename.rstrip('v0123456789'), 'source': 'filename_arxiv_old'}

    # DOI format: 10.1007_s10462-024-11033-5 → 10.1007/s10462-024-11033-5
    if match := re.match(r'^(10\.\d{4,})[-_](.+)$', basename):
        doi = f"{match.group(1)}/{match.group(2).replace('_', '/').replace('-', '/')}"
        # Fix common DOI patterns
        doi = re.sub(r'/+', '/', doi)  # Remove double slashes
        return {'doi': doi, 'source': 'filename_doi'}

    # ScienceDirect PII: 1-s2.0-S0092867421006279-main
    if match := re.match(r'^1-s2\.0-([A-Z0-9]+)(-main)?$', basename):
        return {'pii': match.group(1), 'source': 'filename_pii'}

    return None


def grobid_extract_header(pdf_path: Path) -> Optional[Dict]:
    """Extract header metadata using GROBID."""
    try:
        with open(pdf_path, 'rb') as f:
            response = requests.post(
                f"{GROBID_URL}/api/processHeaderDocument",
                files={'input': f},
                timeout=60
            )

        if response.status_code != 200:
            return None

        # Parse TEI XML response
        return parse_tei_header(response.text)
    except Exception as e:
        print(f"GROBID error for {pdf_path.name}: {e}")
        return None


def parse_tei_header(tei_xml: str) -> Optional[Dict]:
    """Parse GROBID TEI XML to extract metadata."""
    import xml.etree.ElementTree as ET

    try:
        # Define namespaces
        ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
        root = ET.fromstring(tei_xml)

        result = {'source': 'grobid'}

        # Title
        title_elem = root.find('.//tei:titleStmt/tei:title', ns)
        if title_elem is not None and title_elem.text:
            result['title'] = title_elem.text.strip()

        # Authors
        authors = []
        for author in root.findall('.//tei:sourceDesc//tei:author', ns):
            forename = author.find('.//tei:forename', ns)
            surname = author.find('.//tei:surname', ns)
            if surname is not None:
                name = surname.text or ""
                if forename is not None and forename.text:
                    name = f"{forename.text} {name}"
                authors.append(name.strip())
        if authors:
            result['authors'] = authors

        # Year
        date_elem = root.find('.//tei:sourceDesc//tei:date[@when]', ns)
        if date_elem is not None:
            when = date_elem.get('when', '')
            if match := re.match(r'(\d{4})', when):
                result['year'] = int(match.group(1))

        # DOI
        for idno in root.findall('.//tei:idno', ns):
            if idno.get('type') == 'DOI' and idno.text:
                result['doi'] = idno.text.strip()
            elif idno.get('type') == 'arXiv' and idno.text:
                result['arxiv_id'] = idno.text.strip()

        # Abstract
        abstract_elem = root.find('.//tei:abstract', ns)
        if abstract_elem is not None:
            abstract_text = ''.join(abstract_elem.itertext()).strip()
            if abstract_text:
                result['abstract'] = abstract_text

        return result if len(result) > 1 else None

    except ET.ParseError:
        return None


def enrich_from_crossref(doi: str) -> Optional[Dict]:
    """Fetch metadata from Crossref API."""
    try:
        headers = {'User-Agent': f'Polymath/1.0 (mailto:{CROSSREF_EMAIL})'}
        response = requests.get(
            f"https://api.crossref.org/works/{doi}",
            headers=headers,
            timeout=10
        )

        if response.status_code != 200:
            return None

        data = response.json().get('message', {})

        result = {'source': 'crossref', 'doi': doi}

        # Title
        if titles := data.get('title'):
            result['title'] = titles[0]

        # Authors
        if authors := data.get('author'):
            result['authors'] = [
                f"{a.get('given', '')} {a.get('family', '')}".strip()
                for a in authors
            ]

        # Year
        if published := data.get('published-print') or data.get('published-online'):
            if parts := published.get('date-parts', [[]])[0]:
                result['year'] = parts[0]

        # Venue
        if container := data.get('container-title'):
            result['venue'] = container[0] if container else None

        # Abstract
        if abstract := data.get('abstract'):
            # Strip JATS tags
            result['abstract'] = re.sub(r'<[^>]+>', '', abstract)

        return result

    except Exception as e:
        print(f"Crossref error for {doi}: {e}")
        return None


def enrich_from_openalex(doi: str = None, title: str = None, year: int = None) -> Optional[Dict]:
    """Fetch metadata from OpenAlex API."""
    try:
        headers = {'User-Agent': f'mailto:{OPENALEX_EMAIL}'}

        if doi:
            url = f"https://api.openalex.org/works/https://doi.org/{doi}"
        elif title:
            # Search by title
            params = {'search': title}
            if year:
                params['filter'] = f'publication_year:{year}'
            url = "https://api.openalex.org/works"
            response = requests.get(url, params=params, headers=headers, timeout=10)
            if response.status_code == 200:
                results = response.json().get('results', [])
                if results:
                    data = results[0]  # Best match
                else:
                    return None
            else:
                return None
        else:
            return None

        if doi:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                return None
            data = response.json()

        result = {'source': 'openalex'}

        # DOI
        if data.get('doi'):
            result['doi'] = data['doi'].replace('https://doi.org/', '')

        # Title
        if data.get('title'):
            result['title'] = data['title']

        # Authors
        if authorships := data.get('authorships'):
            result['authors'] = [
                a.get('author', {}).get('display_name', '')
                for a in authorships
            ]

        # Year
        if data.get('publication_year'):
            result['year'] = data['publication_year']

        # Venue
        if location := data.get('primary_location'):
            if source := location.get('source'):
                result['venue'] = source.get('display_name')

        # PMID
        if ids := data.get('ids'):
            if pmid := ids.get('pmid'):
                result['pmid'] = pmid.replace('https://pubmed.ncbi.nlm.nih.gov/', '')

        return result

    except Exception as e:
        print(f"OpenAlex error: {e}")
        return None


def enrich_from_arxiv(arxiv_id: str) -> Optional[Dict]:
    """Fetch metadata from arXiv API."""
    try:
        # Clean arXiv ID
        arxiv_id = arxiv_id.replace('arXiv:', '').strip()

        response = requests.get(
            f"http://export.arxiv.org/api/query?id_list={arxiv_id}",
            timeout=10
        )

        if response.status_code != 200:
            return None

        import xml.etree.ElementTree as ET
        root = ET.fromstring(response.text)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}

        entry = root.find('.//atom:entry', ns)
        if entry is None:
            return None

        result = {'source': 'arxiv', 'arxiv_id': arxiv_id}

        # Title
        title = entry.find('atom:title', ns)
        if title is not None and title.text:
            result['title'] = ' '.join(title.text.split())

        # Authors
        authors = []
        for author in entry.findall('atom:author', ns):
            name = author.find('atom:name', ns)
            if name is not None and name.text:
                authors.append(name.text)
        if authors:
            result['authors'] = authors

        # Year
        published = entry.find('atom:published', ns)
        if published is not None and published.text:
            if match := re.match(r'(\d{4})', published.text):
                result['year'] = int(match.group(1))

        # Abstract
        summary = entry.find('atom:summary', ns)
        if summary is not None and summary.text:
            result['abstract'] = ' '.join(summary.text.split())

        # DOI (if available)
        for link in entry.findall('atom:link', ns):
            if 'doi.org' in link.get('href', ''):
                result['doi'] = link.get('href').split('doi.org/')[-1]

        return result

    except Exception as e:
        print(f"arXiv error for {arxiv_id}: {e}")
        return None


def process_pdf(pdf_path: Path) -> PaperMetadata:
    """Process a single PDF through the enrichment pipeline."""
    sha1 = compute_sha1(pdf_path)
    metadata = PaperMetadata(path=str(pdf_path), sha1=sha1)

    # Phase 1: Try filename extraction (fast)
    if ids := extract_id_from_filename(pdf_path.name):
        metadata.source = ids.get('source', 'filename')

        if arxiv_id := ids.get('arxiv_id'):
            metadata.arxiv_id = arxiv_id
            if enriched := enrich_from_arxiv(arxiv_id):
                metadata.title = enriched.get('title')
                metadata.authors = enriched.get('authors')
                metadata.year = enriched.get('year')
                metadata.abstract = enriched.get('abstract')
                metadata.doi = enriched.get('doi')
                metadata.confidence = 0.95
                metadata.match_explain = f"arXiv ID {arxiv_id} exact match"
                return metadata

        if doi := ids.get('doi'):
            metadata.doi = doi
            # Try Crossref first
            if enriched := enrich_from_crossref(doi):
                metadata.title = enriched.get('title')
                metadata.authors = enriched.get('authors')
                metadata.year = enriched.get('year')
                metadata.venue = enriched.get('venue')
                metadata.abstract = enriched.get('abstract')
                metadata.confidence = 0.98
                metadata.match_explain = f"DOI {doi} exact match (Crossref)"
                return metadata
            # Fallback to OpenAlex
            if enriched := enrich_from_openalex(doi=doi):
                metadata.title = enriched.get('title')
                metadata.authors = enriched.get('authors')
                metadata.year = enriched.get('year')
                metadata.venue = enriched.get('venue')
                metadata.pmid = enriched.get('pmid')
                metadata.confidence = 0.95
                metadata.match_explain = f"DOI {doi} match (OpenAlex)"
                return metadata

    # Phase 2: GROBID extraction
    if grobid_data := grobid_extract_header(pdf_path):
        metadata.source = 'grobid'
        metadata.title = grobid_data.get('title')
        metadata.authors = grobid_data.get('authors')
        metadata.year = grobid_data.get('year')
        metadata.doi = grobid_data.get('doi')
        metadata.arxiv_id = grobid_data.get('arxiv_id')
        metadata.abstract = grobid_data.get('abstract')

        # If GROBID found a DOI, enrich it
        if metadata.doi:
            if enriched := enrich_from_crossref(metadata.doi):
                metadata.venue = enriched.get('venue')
                metadata.confidence = 0.90
                metadata.match_explain = "GROBID DOI → Crossref"
                return metadata

        # Try fuzzy title match in OpenAlex
        if metadata.title:
            if enriched := enrich_from_openalex(title=metadata.title, year=metadata.year):
                # Verify title similarity
                if enriched.get('title') and metadata.title:
                    # Simple similarity check
                    t1 = metadata.title.lower()
                    t2 = enriched['title'].lower()
                    if t1 in t2 or t2 in t1 or len(set(t1.split()) & set(t2.split())) > 3:
                        metadata.doi = enriched.get('doi')
                        metadata.venue = enriched.get('venue')
                        metadata.pmid = enriched.get('pmid')
                        metadata.confidence = 0.75
                        metadata.match_explain = "GROBID title → OpenAlex fuzzy"
                        return metadata

        metadata.confidence = 0.50
        metadata.match_explain = "GROBID only (no API verification)"
        return metadata

    # Phase 3: No metadata found
    metadata.needs_review = True
    metadata.confidence = 0.0
    metadata.match_explain = "No metadata extracted"
    return metadata


def update_database(metadata_list: List[PaperMetadata], apply: bool = False):
    """Update Postgres documents table with enriched metadata."""
    if not apply:
        print("\n[DRY RUN] Would update database with:")
        for m in metadata_list[:10]:
            print(f"  {Path(m.path).name}: {m.title or 'NO TITLE'} ({m.confidence:.0%})")
        print(f"  ... and {len(metadata_list) - 10} more")
        return

    conn = psycopg2.connect(POSTGRES_DSN)
    cur = conn.cursor()

    updated = 0
    for m in metadata_list:
        if not m.title:
            continue

        # Find matching document by path or sha1
        cur.execute("""
            UPDATE documents SET
                title = COALESCE(%s, title),
                authors = COALESCE(%s, authors),
                year = COALESCE(%s, year),
                doi = COALESCE(%s, doi),
                pmid = COALESCE(%s, pmid),
                abstract = COALESCE(%s, abstract),
                venue = COALESCE(%s, venue)
            WHERE file_path LIKE %s OR sha1 = %s
            RETURNING id
        """, (
            m.title, m.authors, m.year, m.doi, m.pmid, m.abstract, m.venue,
            f"%{Path(m.path).name}", m.sha1
        ))

        if cur.fetchone():
            updated += 1

    conn.commit()
    cur.close()
    conn.close()

    print(f"\n[APPLIED] Updated {updated} documents in database")


def main():
    parser = argparse.ArgumentParser(description="Enrich PDF metadata")
    parser.add_argument("--pdf-dir", required=True, help="Directory containing PDFs")
    parser.add_argument("--apply", action="store_true", help="Apply changes to database")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of PDFs to process")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--output", default="metadata_enrichment.jsonl", help="Output JSONL file")
    parser.add_argument("--skip-grobid", action="store_true", help="Skip GROBID (filename extraction only)")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.exists():
        print(f"Error: {pdf_dir} does not exist")
        sys.exit(1)

    pdfs = list(pdf_dir.glob("*.pdf"))
    if args.limit > 0:
        pdfs = pdfs[:args.limit]

    print(f"Processing {len(pdfs)} PDFs from {pdf_dir}")

    # Check GROBID availability
    if not args.skip_grobid:
        try:
            r = requests.get(f"{GROBID_URL}/api/isalive", timeout=5)
            if r.status_code == 200:
                print(f"GROBID available at {GROBID_URL}")
            else:
                print(f"Warning: GROBID not responding, will skip GROBID extraction")
                args.skip_grobid = True
        except:
            print(f"Warning: Cannot connect to GROBID at {GROBID_URL}")
            args.skip_grobid = True

    # Process PDFs
    results = []
    stats = {'filename': 0, 'grobid': 0, 'needs_review': 0}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_pdf, pdf): pdf for pdf in pdfs}

        for i, future in enumerate(as_completed(futures)):
            try:
                metadata = future.result()
                results.append(metadata)

                if metadata.needs_review:
                    stats['needs_review'] += 1
                elif 'filename' in metadata.source:
                    stats['filename'] += 1
                else:
                    stats['grobid'] += 1

                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(pdfs)} PDFs...")

            except Exception as e:
                pdf = futures[future]
                print(f"Error processing {pdf.name}: {e}")

    # Write results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        for m in results:
            f.write(json.dumps(asdict(m)) + '\n')

    print(f"\n{'='*60}")
    print(f"Results written to {output_path}")
    print(f"\nStats:")
    print(f"  Filename extraction: {stats['filename']}")
    print(f"  GROBID extraction: {stats['grobid']}")
    print(f"  Needs review: {stats['needs_review']}")
    print(f"  Total: {len(results)}")

    # High confidence results
    high_conf = [m for m in results if m.confidence >= 0.75]
    print(f"\nHigh confidence (≥75%): {len(high_conf)}")

    # Update database
    if high_conf:
        update_database(high_conf, apply=args.apply)

    # Write needs_review to separate file
    needs_review = [m for m in results if m.needs_review]
    if needs_review:
        review_path = output_path.with_suffix('.needs_review.csv')
        with open(review_path, 'w') as f:
            f.write("path,sha1\n")
            for m in needs_review:
                f.write(f"{m.path},{m.sha1}\n")
        print(f"\nNeeds review written to {review_path}")


if __name__ == "__main__":
    main()
