#!/usr/bin/env python3
"""
Migrate ChromaDB: Extract docs from sqlite, re-embed, create fresh DB.
"""

import sqlite3
import sys
import os
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

OLD_DB = "/home/user/work/polymax/chromadb/polymath_v2/chroma.sqlite3"
NEW_PATH = "/home/user/work/polymax/chromadb/polymath_v2"

def extract_documents():
    """Extract all documents with metadata from old sqlite."""
    conn = sqlite3.connect(OLD_DB)
    
    # Get all (id, embedding_id) pairs
    cursor = conn.execute("SELECT id, embedding_id FROM embeddings ORDER BY id;")
    id_map = {row[0]: row[1] for row in cursor.fetchall()}
    print(f"Found {len(id_map)} embeddings")
    
    # Get all metadata grouped by id
    cursor = conn.execute("""
        SELECT id, key, string_value, int_value, float_value
        FROM embedding_metadata
        ORDER BY id
    """)
    
    records = {}
    for row in cursor.fetchall():
        num_id, key, sv, iv, fv = row
        if num_id not in records:
            records[num_id] = {'document': None, 'metadata': {}}
        
        if key == 'chroma:document':
            records[num_id]['document'] = sv
        else:
            val = sv or iv or fv
            if val is not None:
                records[num_id]['metadata'][key] = val
    
    # Convert to list with proper embedding_id
    result = []
    for num_id, data in records.items():
        if data['document'] and num_id in id_map:
            result.append({
                'id': id_map[num_id],
                'document': data['document'],
                'metadata': data['metadata']
            })
    
    conn.close()
    return result


def migrate():
    print("=" * 60)
    print("CHROMADB MIGRATION")
    print("=" * 60)
    print(f"Source: {OLD_DB}")
    print(f"Target: {NEW_PATH}")
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M}")
    print()
    
    # Extract data
    print("Extracting documents from sqlite...")
    records = extract_documents()
    print(f"Valid records: {len(records)}")
    
    if not records:
        print("No data to migrate!")
        return
    
    # Load embedding model
    print("\nLoading embedding model...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-mpnet-base-v2')
    
    # Create fresh ChromaDB
    print(f"Creating fresh ChromaDB at {NEW_PATH}...")
    import chromadb
    Path(NEW_PATH).mkdir(exist_ok=True)
    
    client = chromadb.PersistentClient(path=NEW_PATH)
    coll = client.get_or_create_collection(
        name="polymath_corpus",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Add in batches with fresh embeddings
    batch_size = 200
    added = 0
    
    for i in tqdm(range(0, len(records), batch_size), desc="Embedding & adding"):
        batch = records[i:i+batch_size]
        docs = [r['document'] for r in batch]
        
        # Generate embeddings
        embeddings = model.encode(docs, show_progress_bar=False).tolist()
        
        try:
            coll.add(
                ids=[r['id'] for r in batch],
                embeddings=embeddings,
                documents=docs,
                metadatas=[r['metadata'] for r in batch]
            )
            added += len(batch)
        except Exception as e:
            print(f"\nError adding batch: {e}")
            # Try one by one
            for j, r in enumerate(batch):
                try:
                    coll.add(
                        ids=[r['id']],
                        embeddings=[embeddings[j]],
                        documents=[r['document']],
                        metadatas=[r['metadata']]
                    )
                    added += 1
                except:
                    pass
    
    print(f"\n{'='*60}")
    print(f"MIGRATION COMPLETE")
    print(f"  Migrated: {added}/{len(records)} embeddings")
    print(f"  New DB: {NEW_PATH}")
    print(f"  Collection count: {coll.count()}")


if __name__ == "__main__":
    migrate()
