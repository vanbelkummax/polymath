#!/usr/bin/env python3
"""
Rebuild ChromaDB with BGE-M3 embeddings (1024-dim).
Creates a NEW collection to avoid dimension conflicts.
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from lib.config import CHROMADB_PATH, PAPERS_COLLECTION, CODE_COLLECTION, EMBEDDING_MODEL
from lib.kb_derived import ensure_derived_tables, update_migration_checkpoint, get_migration_checkpoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def rebuild_chroma(
    pg_dsn: str,
    chroma_dir: str,
    collection_name: str,
    batch_size: int = 128,
    limit: int = None,
    resume: bool = True,
    dry_run: bool = False,
    target: str = "passages"
):
    """Rebuild ChromaDB collection with BGE-M3."""
    import psycopg2

    conn = psycopg2.connect(pg_dsn)
    ensure_derived_tables(conn)

    # Load embedding model
    logger.info(f"Loading {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=chroma_dir)

    # Get or create collection
    try:
        collection = client.get_collection(collection_name)
        logger.info(f"Using existing collection: {collection_name}")
    except:
        collection = client.create_collection(
            name=collection_name,
            metadata={"embedding_model": EMBEDDING_MODEL, "embedding_dim": 1024}
        )
        logger.info(f"Created new collection: {collection_name}")

    job_name = f"rebuild_chroma_{collection_name}"

    # Get checkpoint
    start_after = None
    if resume:
        checkpoint = get_migration_checkpoint(conn, job_name)
        if checkpoint and checkpoint["status"] != "completed":
            start_after = checkpoint["cursor_position"]
            logger.info(f"Resuming from {start_after}")

    # Query items
    cursor = conn.cursor()

    if target == "passages":
        if start_after:
            query = "SELECT passage_id::text, passage_text FROM passages WHERE passage_id::text > %s ORDER BY passage_id"
            params = (start_after,)
        else:
            query = "SELECT passage_id::text, passage_text FROM passages ORDER BY passage_id"
            params = None
    else:  # chunks
        if start_after:
            query = "SELECT id, content FROM chunks WHERE id > %s ORDER BY id"
            params = (start_after,)
        else:
            query = "SELECT id, content FROM chunks ORDER BY id"
            params = None

    if limit:
        query += f" LIMIT {limit}"

    cursor.execute(query, params)

    batch_ids = []
    batch_texts = []
    processed = 0

    for row in tqdm(cursor, desc=f"Embedding {target}"):
        item_id, text = row

        if not text or len(text) < 10:
            continue

        batch_ids.append(item_id)
        batch_texts.append(text)

        if len(batch_ids) >= batch_size:
            # Embed batch
            embeddings = model.encode(batch_texts, show_progress_bar=False)

            if not dry_run:
                # Upsert to ChromaDB
                collection.upsert(
                    ids=batch_ids,
                    embeddings=embeddings.tolist(),
                    documents=batch_texts,
                    metadatas=[{"item_id": iid} for iid in batch_ids]
                )

                # Checkpoint
                update_migration_checkpoint(
                    conn, job_name, batch_ids[-1], "running",
                    items_processed=len(batch_ids),
                    cursor_type="passage_id" if target == "passages" else "chunk_id"
                )

            processed += len(batch_ids)
            batch_ids = []
            batch_texts = []

    # Process remaining
    if batch_ids:
        embeddings = model.encode(batch_texts, show_progress_bar=False)

        if not dry_run:
            collection.upsert(
                ids=batch_ids,
                embeddings=embeddings.tolist(),
                documents=batch_texts,
                metadatas=[{"item_id": iid} for iid in batch_ids]
            )

        processed += len(batch_ids)

    # Final checkpoint
    if not dry_run:
        update_migration_checkpoint(
            conn, job_name, None, "completed",
            notes=f"Completed: {processed} items"
        )

    cursor.close()
    conn.close()

    logger.info(f"Rebuild complete: {processed} items embedded")
    logger.info(f"Collection size: {collection.count()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pg-dsn", default="dbname=polymath user=polymath host=/var/run/postgresql")
    parser.add_argument("--chroma-dir", default=CHROMADB_PATH)
    parser.add_argument("--collection", default=PAPERS_COLLECTION)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--target", choices=["passages", "chunks"], default="passages")

    args = parser.parse_args()

    rebuild_chroma(
        args.pg_dsn, args.chroma_dir, args.collection,
        batch_size=args.batch_size,
        limit=args.limit,
        resume=args.resume,
        dry_run=args.dry_run,
        target=args.target
    )


if __name__ == "__main__":
    main()
