#!/usr/bin/env python3
"""
Check provenance orphan rate.

HARDENING_VERSION: 2026-01-05
Expected return: Provenance linkage % (NOT storage mismatch)
Logs: stdout only
Exit codes: 0 = pass (≥20% provenance), 1 = fail (<20% provenance)
"""
import sys
sys.path.insert(0, '/home/user/work/polymax')

import chromadb
import psycopg2

def check_provenance_linkage(sample_size: int = 1000):
    """
    Check what % of chunks have provenance linkage.

    Provenance = any of: doc_id, source_item_id, doi, pmid, arxiv_id
    NOT just existence in Postgres chunks table.
    """
    # Check ChromaDB provenance metadata
    client = chromadb.PersistentClient('/home/user/work/polymax/chromadb/polymath_v2')
    collection = client.get_collection('polymath_corpus')

    total_chunks = collection.count()
    sample = collection.get(limit=sample_size, include=['metadatas'])

    # Count chunks WITH provenance linkage
    has_provenance = 0
    for meta in sample['metadatas']:
        # Provenance = any of: doc_id, source_item_id, doi, pmid, arxiv_id
        if any(meta.get(field) for field in ['doc_id', 'source_item_id', 'doi', 'pmid', 'arxiv_id']):
            has_provenance += 1

    provenance_rate = 100.0 * has_provenance / sample_size

    print(f"=== PROVENANCE LINKAGE CHECK ===")
    print(f"Total chunks in ChromaDB: {total_chunks:,}")
    print(f"Sample size: {sample_size}")
    print(f"With provenance: {has_provenance} ({provenance_rate:.1f}%)")
    print(f"Orphan (no linkage): {sample_size - has_provenance} ({100 - provenance_rate:.1f}%)")

    # Also check Postgres side
    try:
        pg_conn = psycopg2.connect("dbname=polymath user=polymath host=/var/run/postgresql")
        cursor = pg_conn.cursor()

        # Artifacts with source_item_id
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                COUNT(source_item_id) as with_source,
                ROUND(100.0 * COUNT(source_item_id) / COUNT(*), 1) as pct
            FROM artifacts
        """)
        total, with_source, pct = cursor.fetchone()

        print(f"\n=== POSTGRES SOURCE LINKAGE ===")
        print(f"Total artifacts: {total}")
        print(f"With source_item_id: {with_source} ({pct}%)")

        pg_conn.close()
    except Exception as e:
        print(f"\n⚠️  Could not check Postgres: {e}")

    # STOP THE LINE check
    if provenance_rate < 20:
        print(f"\n🛑 STOP THE LINE: Provenance linkage {provenance_rate:.1f}% < 20% threshold")
        print("DO NOT ingest new content until Phase 1-2 migration complete")
        print("Run: python3 scripts/create_document_registry.py --execute")
        return 1
    else:
        print(f"\n✅ Provenance linkage OK ({provenance_rate:.1f}% >= 20%)")
        return 0


if __name__ == "__main__":
    exit_code = check_provenance_linkage()
    sys.exit(exit_code)
