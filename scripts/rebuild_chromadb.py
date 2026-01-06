#!/usr/bin/env python3
"""
Rebuild ChromaDB from Postgres data.

Rebuilds the vector index from:
- passages (532K, paper content)
- code_chunks (412K, code content)

Usage:
    python3 scripts/rebuild_chromadb.py --batch-size 500 --workers 4

    # Resume from checkpoint
    python3 scripts/rebuild_chromadb.py --resume
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib

import chromadb
import psycopg2
from psycopg2.extras import RealDictCursor
from sentence_transformers import SentenceTransformer

# Config
CHROMA_PATH = "/home/user/work/polymax/chromadb/polymath_v2"
POSTGRES_DSN = "dbname=polymath user=polymath host=/var/run/postgresql"
EMBEDDING_MODEL = "all-mpnet-base-v2"
CHECKPOINT_FILE = "/home/user/work/polymax/chromadb_rebuild_checkpoint.json"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RebuildStats:
    passages_total: int = 0
    passages_done: int = 0
    code_chunks_total: int = 0
    code_chunks_done: int = 0
    errors: int = 0
    start_time: float = 0


def get_postgres_connection():
    return psycopg2.connect(POSTGRES_DSN)


def load_checkpoint() -> Dict[str, Any]:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {
        'passages_offset': 0,
        'code_chunks_offset': 0,
        'completed': False
    }


def save_checkpoint(checkpoint: Dict[str, Any]):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f)


def fetch_passages_batch(conn, offset: int, limit: int) -> List[Dict]:
    """Fetch a batch of passages from Postgres."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT
                p.passage_id::text as id,
                p.passage_text as content,
                p.page_num,
                p.section,
                d.title,
                d.doc_id::text,
                d.year,
                d.authors
            FROM passages p
            JOIN documents d ON p.doc_id = d.doc_id
            WHERE p.citable = true
            ORDER BY p.passage_id
            OFFSET %s LIMIT %s
        """, (offset, limit))
        return [dict(row) for row in cur.fetchall()]


def fetch_code_chunks_batch(conn, offset: int, limit: int) -> List[Dict]:
    """Fetch a batch of code chunks from Postgres."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT
                cc.chunk_id::text as id,
                cc.content,
                cc.name,
                cc.chunk_type,
                cc.class_name,
                cc.symbol_qualified_name,
                cc.docstring,
                cc.signature,
                cc.concepts,
                cc.start_line,
                cc.end_line,
                cf.file_path,
                cf.repo_name,
                cf.commit_sha
            FROM code_chunks cc
            JOIN code_files cf ON cc.file_id = cf.file_id
            ORDER BY cc.chunk_id
            OFFSET %s LIMIT %s
        """, (offset, limit))
        return [dict(row) for row in cur.fetchall()]


def prepare_passage_for_chroma(passage: Dict) -> tuple:
    """Prepare a passage for ChromaDB insertion."""
    # Create searchable text
    text = passage['content']
    if passage.get('title'):
        text = f"# {passage['title']}\n\n{text}"

    metadata = {
        'source': 'passage',
        'doc_id': passage['doc_id'],
        'title': passage.get('title', ''),
        'page_num': passage.get('page_num', -1),
        'section': passage.get('section', ''),
        'year': passage.get('year') or 0,
    }

    return passage['id'], text, metadata


def prepare_code_chunk_for_chroma(chunk: Dict) -> tuple:
    """Prepare a code chunk for ChromaDB insertion."""
    # Create searchable text with context
    parts = []
    if chunk.get('repo_name'):
        parts.append(f"Repository: {chunk['repo_name']}")
    if chunk.get('file_path'):
        parts.append(f"File: {chunk['file_path']}")
    if chunk.get('name'):
        parts.append(f"Name: {chunk['name']}")
    if chunk.get('docstring'):
        parts.append(f"Docstring: {chunk['docstring']}")
    if chunk.get('signature'):
        parts.append(f"Signature: {chunk['signature']}")
    parts.append(f"Code:\n{chunk['content']}")

    text = "\n".join(parts)

    metadata = {
        'source': 'code',
        'repo_name': chunk.get('repo_name', ''),
        'file_path': chunk.get('file_path', ''),
        'name': chunk.get('name', ''),
        'chunk_type': chunk.get('chunk_type', ''),
        'class_name': chunk.get('class_name', ''),
        'start_line': chunk.get('start_line', 0),
        'end_line': chunk.get('end_line', 0),
        'concepts': ','.join(chunk.get('concepts') or []),
    }

    return chunk['id'], text, metadata


def rebuild_chromadb(
    batch_size: int = 500,
    resume: bool = False,
    passages_only: bool = False,
    code_only: bool = False
):
    """Main rebuild function."""

    # Load checkpoint
    checkpoint = load_checkpoint() if resume else {
        'passages_offset': 0,
        'code_chunks_offset': 0,
        'completed': False
    }

    if checkpoint.get('completed'):
        logger.info("Rebuild already completed. Use --resume=false to start fresh.")
        return

    # Initialize ChromaDB
    logger.info(f"Initializing ChromaDB at {CHROMA_PATH}")
    os.makedirs(CHROMA_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Create or get collection
    try:
        collection = client.get_collection("polymath_corpus")
        logger.info(f"Using existing collection with {collection.count()} items")
    except Exception:
        collection = client.create_collection(
            name="polymath_corpus",
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Created new collection")

    # Initialize embedder
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedder = SentenceTransformer(EMBEDDING_MODEL, device='cuda')

    # Get counts
    conn = get_postgres_connection()
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM passages WHERE citable = true")
        total_passages = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM code_chunks")
        total_code = cur.fetchone()[0]

    logger.info(f"Total passages: {total_passages:,}")
    logger.info(f"Total code chunks: {total_code:,}")

    stats = RebuildStats(
        passages_total=total_passages,
        code_chunks_total=total_code,
        passages_done=checkpoint['passages_offset'],
        code_chunks_done=checkpoint['code_chunks_offset'],
        start_time=time.time()
    )

    # Process passages
    if not code_only:
        logger.info(f"Processing passages from offset {checkpoint['passages_offset']}")
        offset = checkpoint['passages_offset']

        while offset < total_passages:
            batch = fetch_passages_batch(conn, offset, batch_size)
            if not batch:
                break

            # Prepare data
            ids, texts, metadatas = [], [], []
            for passage in batch:
                try:
                    pid, text, meta = prepare_passage_for_chroma(passage)
                    ids.append(pid)
                    texts.append(text[:8000])  # Truncate very long passages
                    metadatas.append(meta)
                except Exception as e:
                    logger.warning(f"Error preparing passage: {e}")
                    stats.errors += 1

            if texts:
                # Embed
                embeddings = embedder.encode(texts, show_progress_bar=False).tolist()

                # Upsert to ChromaDB
                collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas
                )

            offset += batch_size
            stats.passages_done = offset

            # Save checkpoint
            checkpoint['passages_offset'] = offset
            save_checkpoint(checkpoint)

            # Progress
            elapsed = time.time() - stats.start_time
            rate = stats.passages_done / elapsed if elapsed > 0 else 0
            eta = (total_passages - stats.passages_done) / rate if rate > 0 else 0
            logger.info(
                f"Passages: {stats.passages_done:,}/{total_passages:,} "
                f"({100*stats.passages_done/total_passages:.1f}%) "
                f"Rate: {rate:.1f}/s ETA: {eta/60:.1f}min"
            )

    # Process code chunks
    if not passages_only:
        logger.info(f"Processing code chunks from offset {checkpoint['code_chunks_offset']}")
        offset = checkpoint['code_chunks_offset']

        while offset < total_code:
            batch = fetch_code_chunks_batch(conn, offset, batch_size)
            if not batch:
                break

            # Prepare data
            ids, texts, metadatas = [], [], []
            for chunk in batch:
                try:
                    cid, text, meta = prepare_code_chunk_for_chroma(chunk)
                    ids.append(cid)
                    texts.append(text[:8000])  # Truncate
                    metadatas.append(meta)
                except Exception as e:
                    logger.warning(f"Error preparing code chunk: {e}")
                    stats.errors += 1

            if texts:
                # Embed
                embeddings = embedder.encode(texts, show_progress_bar=False).tolist()

                # Upsert to ChromaDB
                collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas
                )

            offset += batch_size
            stats.code_chunks_done = offset

            # Save checkpoint
            checkpoint['code_chunks_offset'] = offset
            save_checkpoint(checkpoint)

            # Progress
            elapsed = time.time() - stats.start_time
            total_done = stats.passages_done + stats.code_chunks_done
            total_items = total_passages + total_code
            rate = total_done / elapsed if elapsed > 0 else 0
            eta = (total_items - total_done) / rate if rate > 0 else 0
            logger.info(
                f"Code: {stats.code_chunks_done:,}/{total_code:,} "
                f"({100*stats.code_chunks_done/total_code:.1f}%) "
                f"Rate: {rate:.1f}/s ETA: {eta/60:.1f}min"
            )

    # Mark complete
    checkpoint['completed'] = True
    save_checkpoint(checkpoint)

    # Final stats
    elapsed = time.time() - stats.start_time
    logger.info("="*60)
    logger.info("REBUILD COMPLETE")
    logger.info(f"Passages indexed: {stats.passages_done:,}")
    logger.info(f"Code chunks indexed: {stats.code_chunks_done:,}")
    logger.info(f"Total items: {collection.count():,}")
    logger.info(f"Errors: {stats.errors}")
    logger.info(f"Time: {elapsed/60:.1f} minutes")
    logger.info("="*60)

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild ChromaDB from Postgres")
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--passages-only", action="store_true", help="Only rebuild passages")
    parser.add_argument("--code-only", action="store_true", help="Only rebuild code")

    args = parser.parse_args()

    rebuild_chromadb(
        batch_size=args.batch_size,
        resume=args.resume,
        passages_only=args.passages_only,
        code_only=args.code_only
    )
