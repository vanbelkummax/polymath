#!/usr/bin/env python3
"""
Build a mapping between PDFs on disk and doc_ids in Postgres.

Outputs a CSV report so auditors can locate the exact PDF used for citations.
Optionally stores file paths as doc_aliases (alias_type='file_path').
"""
import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import psycopg2

# Add lib to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.doc_identity import compute_doc_id, normalize_title

DEFAULT_DIRS = [
    Path('/home/user/work/polymath_pdfs'),
    Path('/home/user/work/polymax/ingest_staging'),
    Path('/home/user/work/polymax/ingest_staging/yuankai_huo'),
]


def get_db_connection():
    return psycopg2.connect(
        dbname='polymath',
        user='polymath',
        host='/var/run/postgresql'
    )


def iter_pdfs(dirs: List[Path], recursive: bool) -> Iterable[Path]:
    for d in dirs:
        if not d.exists():
            continue
        if recursive:
            yield from sorted(d.rglob('*.pdf'))
            yield from sorted(d.rglob('*.PDF'))
        else:
            yield from sorted(d.glob('*.pdf'))
            yield from sorted(d.glob('*.PDF'))


def title_from_filename(pdf_path: Path) -> str:
    title = pdf_path.stem
    return title.replace('_', ' ').replace('-', ' ')


def find_doc_id_by_title(cursor, title: str) -> Optional[str]:
    title_hash = normalize_title(title)
    title_hash = title_hash[:16] if len(title_hash) > 16 else title_hash
    cursor.execute("""
        SELECT doc_id
        FROM documents
        WHERE LOWER(title) LIKE %s
           OR title_hash LIKE %s
        LIMIT 1
    """, (f'%{title[:30].lower()}%', f'{title_hash}%'))
    row = cursor.fetchone()
    return row[0] if row else None


def find_doc_id_exact(cursor, doc_id: str) -> Optional[str]:
    cursor.execute("SELECT doc_id FROM documents WHERE doc_id = %s", (doc_id,))
    row = cursor.fetchone()
    return row[0] if row else None


def doc_has_passages(cursor, doc_id: str) -> bool:
    cursor.execute("SELECT 1 FROM passages WHERE doc_id = %s LIMIT 1", (doc_id,))
    return cursor.fetchone() is not None


def insert_file_alias(cursor, doc_id: str, file_path: str) -> None:
    cursor.execute("""
        INSERT INTO doc_aliases (doc_id, alias_type, alias_value)
        VALUES (%s, 'file_path', %s)
        ON CONFLICT (alias_type, alias_value) DO NOTHING
    """, (doc_id, file_path))


def main():
    parser = argparse.ArgumentParser(description="Map PDFs to doc_ids")
    parser.add_argument('--dir', action='append', default=None,
                        help='Directory to scan (repeatable)')
    parser.add_argument('--recursive', action='store_true',
                        help='Scan subdirectories recursively')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of PDFs processed')
    parser.add_argument('--write-doc-aliases', action='store_true',
                        help='Persist file_path aliases in doc_aliases')
    parser.add_argument('--output', type=str, default=None,
                        help='CSV output path')
    args = parser.parse_args()

    dirs = [Path(d) for d in args.dir] if args.dir else DEFAULT_DIRS
    pdfs = list(iter_pdfs(dirs, args.recursive))
    if args.limit:
        pdfs = pdfs[:args.limit]

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = Path(args.output) if args.output else Path(
        f"/home/user/work/polymax/reports/pdf_doc_map_{timestamp}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    conn = get_db_connection()
    cursor = conn.cursor()

    matched = 0
    matched_with_passages = 0
    unmatched = 0

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "pdf_path",
            "title_guess",
            "doc_id",
            "in_documents",
            "has_passages",
            "note",
        ])

        for pdf in pdfs:
            title = title_from_filename(pdf)
            doc_id = find_doc_id_by_title(cursor, title)
            note = ""

            if not doc_id:
                # Fallback to deterministic compute_doc_id (legacy batch behavior)
                fallback_id = compute_doc_id(title, ["Unknown"], 2024)
                doc_id = find_doc_id_exact(cursor, str(fallback_id))
                if doc_id:
                    note = "matched_by_computed_id"

            if doc_id:
                matched += 1
                has_passages = doc_has_passages(cursor, doc_id)
                if has_passages:
                    matched_with_passages += 1
                if args.write_doc_aliases:
                    insert_file_alias(cursor, doc_id, str(pdf))
            else:
                unmatched += 1
                has_passages = False

            writer.writerow([
                str(pdf),
                title,
                doc_id or "",
                "yes" if doc_id else "no",
                "yes" if has_passages else "no",
                note,
            ])

    if args.write_doc_aliases:
        conn.commit()
    conn.close()

    print("PDF → doc_id mapping complete")
    print(f"Scanned PDFs: {len(pdfs)}")
    print(f"Matched docs: {matched}")
    print(f"Matched with passages: {matched_with_passages}")
    print(f"Unmatched: {unmatched}")
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
