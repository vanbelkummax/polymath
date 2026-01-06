#!/usr/bin/env python3
"""
Sync chunks from ChromaDB to Postgres for BM25/FTS search.
One-time backfill to enable hybrid retrieval.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Set
import hashlib

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

CHROMA_PATH = "/home/user/work/polymax/chromadb/polymath_v2"
POSTGRES_URL = "dbname=polymath user=polymath host=/var/run/postgresql"
BATCH_SIZE = 500


def sync_chunks_to_postgres():
    """Sync all chunks from ChromaDB to Postgres."""
    import chromadb
    import psycopg2
    from tqdm import tqdm

    # Connect to both stores
    print("Connecting to ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection("polymath_corpus")

    print("Connecting to Postgres...")
    pg = psycopg2.connect(POSTGRES_URL)
    cursor = pg.cursor()

    # Get total chunk count
    total_chunks = collection.count()
    print(f"Total chunks in ChromaDB: {total_chunks}")

    # Get existing chunks in Postgres to avoid duplicates
    cursor.execute("SELECT id FROM chunks")
    existing_ids = {row[0] for row in cursor.fetchall()}
    print(f"Existing chunks in Postgres: {len(existing_ids)}")

    # Check artifact columns for optional fields
    cursor.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_name = 'artifacts'
    """)
    artifact_columns = {row[0] for row in cursor.fetchall()}
    has_ingest_run = "ingest_run_id" in artifact_columns
    has_ingestion_method = "ingestion_method" in artifact_columns

    # Get existing artifacts for reference
    cursor.execute("SELECT file_hash, id FROM artifacts")
    artifact_map: Dict[str, str] = {row[0]: row[1] for row in cursor.fetchall()}
    print(f"Existing artifacts: {len(artifact_map)}")

    # Process in batches
    offset = 0
    inserted = 0
    skipped = 0
    artifacts_created = 0

    with tqdm(total=total_chunks, desc="Syncing chunks") as pbar:
        while offset < total_chunks:
            # Fetch batch from ChromaDB
            results = collection.get(
                limit=BATCH_SIZE,
                offset=offset,
                include=["documents", "metadatas"]
            )

            if not results['ids']:
                break

            for i, chunk_id in enumerate(results['ids']):
                if chunk_id in existing_ids:
                    skipped += 1
                    pbar.update(1)
                    continue

                content = results['documents'][i]
                meta = results['metadatas'][i]

                # Get or create artifact
                file_hash = meta.get('file_hash', '')
                if not file_hash:
                    # Generate hash from title
                    title = meta.get('title', 'Unknown')
                    file_hash = hashlib.md5(title.encode()).hexdigest()[:16]

                if file_hash not in artifact_map:
                    # Create artifact
                    title = meta.get('title', 'Unknown')
                    source = meta.get('source', '')
                    artifact_type = meta.get('type', 'paper')

                    try:
                        ingest_run_id = meta.get("ingest_run_id")
                        ingestion_method = meta.get("ingestion_method")

                        columns = ["artifact_type", "title", "file_path", "file_hash", "indexed_at"]
                        placeholders = ["%s", "%s", "%s", "%s", "NOW()"]
                        values = [artifact_type, title, source, file_hash]

                        if has_ingest_run and ingest_run_id:
                            columns.append("ingest_run_id")
                            placeholders.append("%s")
                            values.append(ingest_run_id)

                        if has_ingestion_method and ingestion_method:
                            columns.append("ingestion_method")
                            placeholders.append("%s")
                            values.append(ingestion_method)

                        update_fields = ["indexed_at = NOW()"]
                        if has_ingest_run:
                            update_fields.append(
                                "ingest_run_id = COALESCE(EXCLUDED.ingest_run_id, artifacts.ingest_run_id)"
                            )
                        if has_ingestion_method:
                            update_fields.append(
                                "ingestion_method = COALESCE(EXCLUDED.ingestion_method, artifacts.ingestion_method)"
                            )

                        query = f"""
                            INSERT INTO artifacts ({", ".join(columns)})
                            VALUES ({", ".join(placeholders)})
                            ON CONFLICT (file_hash) DO UPDATE SET {", ".join(update_fields)}
                            RETURNING id
                        """
                        cursor.execute(query, values)
                        artifact_id = cursor.fetchone()[0]
                        artifact_map[file_hash] = artifact_id
                        artifacts_created += 1
                    except Exception as e:
                        print(f"Error creating artifact: {e}")
                        continue

                artifact_id = artifact_map[file_hash]
                chunk_index = meta.get('chunk_index', 0)

                # Insert chunk
                try:
                    cursor.execute("""
                        INSERT INTO chunks (id, artifact_id, chunk_index, content, content_tsv)
                        VALUES (%s, %s, %s, %s, to_tsvector('english', %s))
                        ON CONFLICT (id) DO NOTHING
                    """, (chunk_id, artifact_id, chunk_index, content, content))
                    inserted += 1
                except Exception as e:
                    print(f"Error inserting chunk {chunk_id}: {e}")
                    pg.rollback()
                    continue

                pbar.update(1)

            # Commit batch
            pg.commit()
            offset += BATCH_SIZE

    # Final commit
    pg.commit()

    # Verify
    cursor.execute("SELECT COUNT(*) FROM chunks")
    final_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM chunks WHERE content_tsv IS NOT NULL")
    fts_count = cursor.fetchone()[0]

    print(f"\n{'='*50}")
    print(f"SYNC COMPLETE")
    print(f"Chunks inserted: {inserted}")
    print(f"Chunks skipped (already existed): {skipped}")
    print(f"Artifacts created: {artifacts_created}")
    print(f"Total chunks in Postgres: {final_count}")
    print(f"Chunks with FTS: {fts_count}")

    cursor.close()
    pg.close()


if __name__ == "__main__":
    sync_chunks_to_postgres()
