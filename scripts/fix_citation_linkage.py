#!/usr/bin/env python3
"""
Fix citation linkage by connecting orphan passage docs to artifacts.
"""
import psycopg2
from pathlib import Path
import sys

# PDF directories
PDF_DIRS = [
    Path('/home/user/work/polymath_pdfs'),
    Path('/home/user/work/polymax/ingest_staging'),
]

def get_conn():
    return psycopg2.connect(dbname='polymath', user='polymath', host='/var/run/postgresql')

def find_pdf_path(title: str) -> str | None:
    """Find PDF file matching the title (which is often just a number like '18443')."""
    for dir_path in PDF_DIRS:
        # Try exact match
        pdf_path = dir_path / f"{title}.pdf"
        if pdf_path.exists():
            return str(pdf_path)
        # Try case-insensitive search
        for pdf in dir_path.glob("*.pdf"):
            if pdf.stem.lower() == title.lower():
                return str(pdf)
    return None

def main():
    conn = get_conn()
    cur = conn.cursor()
    
    # Step 1: Find orphan docs (have passages but no artifact link)
    cur.execute("""
        SELECT d.doc_id, d.title
        FROM documents d
        WHERE d.doc_id IN (SELECT DISTINCT doc_id FROM passages)
        AND NOT EXISTS (SELECT 1 FROM artifacts a WHERE a.doc_id = d.doc_id)
    """)
    orphans = cur.fetchall()
    print(f"Found {len(orphans)} orphan docs with passages")
    
    linked = 0
    created = 0
    
    for doc_id, title in orphans:
        # Find the PDF file
        pdf_path = find_pdf_path(title)
        if not pdf_path:
            continue
            
        # Check if artifact exists for this file (by path or by matching name)
        cur.execute("""
            SELECT id FROM artifacts 
            WHERE file_path LIKE %s 
            OR (title = %s AND artifact_type = 'paper')
            LIMIT 1
        """, (f"%{title}.pdf", title))
        
        result = cur.fetchone()
        if result:
            # Link existing artifact to this doc
            artifact_id = result[0]
            cur.execute("""
                UPDATE artifacts SET doc_id = %s, file_path = %s
                WHERE id = %s AND (doc_id IS NULL OR doc_id = %s)
            """, (str(doc_id), pdf_path, artifact_id, str(doc_id)))
            if cur.rowcount > 0:
                linked += 1
        else:
            # Create new artifact for this doc
            cur.execute("""
                INSERT INTO artifacts (title, artifact_type, file_path, doc_id)
                VALUES (%s, 'paper', %s, %s)
                ON CONFLICT DO NOTHING
            """, (title, pdf_path, str(doc_id)))
            if cur.rowcount > 0:
                created += 1
        
        if (linked + created) % 100 == 0:
            print(f"Progress: {linked} linked, {created} created")
            conn.commit()
    
    conn.commit()
    print(f"\nDone! Linked: {linked}, Created: {created}")
    
    # Verify
    cur.execute("""
        SELECT COUNT(DISTINCT p.doc_id) as citable_docs
        FROM passages p
        JOIN documents d ON d.doc_id = p.doc_id
        JOIN artifacts a ON a.doc_id = d.doc_id
        WHERE a.file_path IS NOT NULL AND a.file_path != ''
    """)
    print(f"Fully citable docs now: {cur.fetchone()[0]}")
    
    conn.close()

if __name__ == "__main__":
    main()
