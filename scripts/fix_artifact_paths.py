#!/usr/bin/env python3
"""
Populate missing file_path for artifacts by searching PDF directories.
"""
import psycopg2
from pathlib import Path
import re

PDF_DIRS = [
    Path('/home/user/work/polymath_pdfs'),
    Path('/home/user/work/polymax/ingest_staging'),
]

# Build index of all PDFs
def build_pdf_index():
    """Build filename -> full path index."""
    index = {}
    for dir_path in PDF_DIRS:
        if dir_path.exists():
            for pdf in dir_path.rglob("*.pdf"):
                # Index by stem (without extension)
                key = pdf.stem.lower()
                if key not in index:
                    index[key] = str(pdf)
                # Also index by full filename
                index[pdf.name.lower()] = str(pdf)
    return index

def normalize_title(title: str) -> str:
    """Normalize title for matching."""
    if not title:
        return ""
    # Remove special chars, lowercase
    return re.sub(r'[^a-z0-9]', '', title.lower())[:50]

def main():
    conn = psycopg2.connect(dbname='polymath', user='polymath', host='/var/run/postgresql')
    cur = conn.cursor()
    
    print("Building PDF index...")
    pdf_index = build_pdf_index()
    print(f"Indexed {len(pdf_index)} PDF files")
    
    # Get artifacts with empty file_path
    cur.execute("""
        SELECT id, title FROM artifacts 
        WHERE artifact_type = 'paper'
        AND (file_path IS NULL OR file_path = '')
    """)
    empty_artifacts = cur.fetchall()
    print(f"Found {len(empty_artifacts)} artifacts with empty file_path")
    
    fixed = 0
    for artifact_id, title in empty_artifacts:
        if not title:
            continue
            
        # Try various matching strategies
        path = None
        
        # 1. Exact stem match
        key = title.lower()
        if key in pdf_index:
            path = pdf_index[key]
        
        # 2. Normalized title match
        if not path:
            norm = normalize_title(title)
            for pdf_key, pdf_path in pdf_index.items():
                if normalize_title(pdf_key) == norm:
                    path = pdf_path
                    break
        
        # 3. Title starts with number (like "18443")
        if not path and title.isdigit():
            key = f"{title}.pdf".lower()
            if key in pdf_index:
                path = pdf_index[key]
        
        if path:
            cur.execute("UPDATE artifacts SET file_path = %s WHERE id = %s", 
                       (path, artifact_id))
            fixed += 1
            if fixed % 500 == 0:
                print(f"Fixed {fixed} artifacts")
                conn.commit()
    
    conn.commit()
    print(f"\nDone! Fixed {fixed} artifact file_paths")
    
    # Verify
    cur.execute("""
        SELECT 
            COUNT(*) as total,
            COUNT(CASE WHEN file_path IS NOT NULL AND file_path != '' THEN 1 END) as with_path
        FROM artifacts WHERE artifact_type = 'paper'
    """)
    total, with_path = cur.fetchone()
    print(f"Artifacts with file_path: {with_path}/{total}")
    
    conn.close()

if __name__ == "__main__":
    main()
