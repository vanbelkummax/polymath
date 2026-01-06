#!/usr/bin/env python3
"""
Simple Backfill - Minimal dependencies, runs in separate process
"""
import os
import sys
import re

# Completely isolate from CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

def main():
    print("Simple Backfill Starting...")

    # Import only after env vars are set
    import sqlite3
    from datetime import datetime

    # Direct SQLite access to ChromaDB's underlying database
    db_path = "/home/user/work/polymax/chromadb/polymath_v2/chroma.sqlite3"

    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        return

    print(f"Opening: {db_path}")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Get collection info
    cur.execute("SELECT id, name FROM collections")
    collections = cur.fetchall()
    print(f"Collections: {collections}")

    # Find polymath_corpus collection
    corpus_id = None
    for cid, name in collections:
        if name == 'polymath_corpus':
            corpus_id = cid
            break

    if not corpus_id:
        print("polymath_corpus not found")
        conn.close()
        return

    # Count segments needing metadata
    cur.execute("""
        SELECT COUNT(*) FROM embedding_metadata
        WHERE segment_id IN (SELECT id FROM segments WHERE collection_id = ?)
    """, (corpus_id,))
    total_meta = cur.fetchone()[0]
    print(f"Metadata entries: {total_meta}")

    # For now, just report status - actual updates need ChromaDB API
    print("Note: Direct SQLite update skipped - use ChromaDB API for safety")
    print(f"Finished: {datetime.now():%H:%M}")

    conn.close()

if __name__ == "__main__":
    main()
