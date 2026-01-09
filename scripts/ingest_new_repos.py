#!/usr/bin/env python3
"""
Ingest new repos from retrieval directory into Postgres + ChromaDB.

1. Ingests repo code into Postgres (code_files, code_chunks tables)
2. Embeds chunks into ChromaDB BGE-M3 collection

Usage:
    python3 scripts/ingest_new_repos.py
    python3 scripts/ingest_new_repos.py --repo Starlitnightly_omicverse
    python3 scripts/ingest_new_repos.py --dry-run
"""

import argparse
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import List, Set

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
import psycopg2
from psycopg2.extras import RealDictCursor
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from lib.config import CHROMADB_PATH, CODE_COLLECTION, EMBEDDING_MODEL, POSTGRES_DSN
from lib.code_ingest import ingest_repo, scan_repo

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)

RETRIEVAL_DIR = Path("/databases/polymath_retrieval_2026_01_07/repos")
BATCH_SIZE = 128


def get_postgres_repos() -> Set[str]:
    """Get repo names already in Postgres."""
    conn = psycopg2.connect(POSTGRES_DSN)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT repo_name FROM code_files")
    repos = set(row[0] for row in cur.fetchall())
    cur.close()
    conn.close()
    return repos


def get_chromadb_repos() -> Set[str]:
    """Get repo names already in ChromaDB."""
    client = chromadb.PersistentClient(path=str(CHROMADB_PATH))
    code_coll = client.get_collection(CODE_COLLECTION)

    repos = set()
    offset = 0
    batch_size = 10000
    while True:
        results = code_coll.get(limit=batch_size, offset=offset, include=['metadatas'])
        if not results['metadatas']:
            break
        for m in results['metadatas']:
            if m and 'repo_name' in m:
                repos.add(m['repo_name'])
        offset += batch_size
    return repos


def dir_to_repo_name(dir_name: str) -> str:
    """Convert directory name (org_repo) to repo name (org/repo)."""
    parts = dir_name.split('_', 1)
    if len(parts) == 2:
        return f"{parts[0]}/{parts[1]}"
    return dir_name


def embed_repo_chunks(repo_name: str, model: SentenceTransformer, dry_run: bool = False):
    """Embed all chunks from a repo into ChromaDB."""
    conn = psycopg2.connect(POSTGRES_DSN)

    # Get chunks for this repo
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT
                cc.chunk_id::text as id,
                cc.content,
                cc.name,
                cc.chunk_type,
                cc.class_name,
                cc.start_line,
                cc.end_line,
                cc.concepts,
                cf.file_path,
                cf.repo_name,
                cf.language
            FROM code_chunks cc
            JOIN code_files cf ON cc.file_id = cf.file_id
            WHERE cf.repo_name = %s
        """, (repo_name,))
        chunks = list(cur.fetchall())

    conn.close()

    if not chunks:
        logger.warning(f"No chunks found for {repo_name}")
        return 0

    logger.info(f"Embedding {len(chunks)} chunks from {repo_name}")

    if dry_run:
        return len(chunks)

    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=str(CHROMADB_PATH))
    code_coll = client.get_collection(CODE_COLLECTION)

    # Batch embed and upsert
    batch_ids = []
    batch_texts = []
    batch_metas = []
    embedded = 0

    for chunk in tqdm(chunks, desc=f"Embedding {repo_name}"):
        content = chunk['content']
        if not content or len(content) < 10:
            continue

        # Create ID in ChromaDB format
        chunk_id = f"c_{chunk['id']}"

        batch_ids.append(chunk_id)
        batch_texts.append(content[:8000])  # Truncate very long chunks
        batch_metas.append({
            'repo_name': chunk['repo_name'],
            'file_path': chunk['file_path'] or '',
            'language': chunk['language'] or 'unknown',
            'chunk_type': chunk['chunk_type'] or 'module',
            'start_line': chunk['start_line'] or 0,
            'end_line': chunk['end_line'] or 0,
            'source_model': 'bge_m3_v1'
        })

        if len(batch_ids) >= BATCH_SIZE:
            # Embed batch
            embeddings = model.encode(batch_texts, show_progress_bar=False)
            code_coll.upsert(
                ids=batch_ids,
                embeddings=embeddings.tolist(),
                documents=batch_texts,
                metadatas=batch_metas
            )
            embedded += len(batch_ids)
            batch_ids = []
            batch_texts = []
            batch_metas = []

    # Process remaining
    if batch_ids:
        embeddings = model.encode(batch_texts, show_progress_bar=False)
        code_coll.upsert(
            ids=batch_ids,
            embeddings=embeddings.tolist(),
            documents=batch_texts,
            metadatas=batch_metas
        )
        embedded += len(batch_ids)

    return embedded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", help="Specific repo directory to process (e.g., Starlitnightly_omicverse)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--skip-postgres", action="store_true", help="Skip Postgres ingestion")
    parser.add_argument("--skip-chromadb", action="store_true", help="Skip ChromaDB embedding")
    args = parser.parse_args()

    # Get existing repos
    logger.info("Checking existing repos...")
    postgres_repos = get_postgres_repos()
    chromadb_repos = get_chromadb_repos()

    logger.info(f"Postgres repos: {len(postgres_repos)}")
    logger.info(f"ChromaDB repos: {len(chromadb_repos)}")

    # Get repos to process
    if args.repo:
        repo_dirs = [args.repo]
    else:
        repo_dirs = sorted(os.listdir(RETRIEVAL_DIR))

    # Filter to repos not in ChromaDB
    repos_to_process = []
    for dir_name in repo_dirs:
        repo_name = dir_to_repo_name(dir_name)
        if repo_name not in chromadb_repos:
            repos_to_process.append((dir_name, repo_name))

    logger.info(f"Repos to process: {len(repos_to_process)}")

    if args.dry_run:
        for dir_name, repo_name in repos_to_process:
            needs_postgres = repo_name not in postgres_repos
            print(f"  {dir_name} -> {repo_name}")
            print(f"    Postgres: {'INGEST' if needs_postgres else 'SKIP'}")
            print(f"    ChromaDB: EMBED")
        return

    # Load embedding model
    if not args.skip_chromadb:
        logger.info(f"Loading {EMBEDDING_MODEL}...")
        model = SentenceTransformer(EMBEDDING_MODEL)
    else:
        model = None

    # Process each repo
    total_files = 0
    total_chunks = 0
    total_embedded = 0

    for dir_name, repo_name in repos_to_process:
        repo_path = RETRIEVAL_DIR / dir_name

        # Step 1: Postgres ingestion
        if not args.skip_postgres and repo_name not in postgres_repos:
            logger.info(f"Ingesting {repo_name} into Postgres...")
            stats = ingest_repo(repo_path)
            total_files += stats.get('files', 0)
            total_chunks += stats.get('chunks', 0)
            if stats.get('files', 0) > 0:
                postgres_repos.add(repo_name)

        # Step 2: ChromaDB embedding
        if not args.skip_chromadb and model:
            logger.info(f"Embedding {repo_name} into ChromaDB...")
            embedded = embed_repo_chunks(repo_name, model, dry_run=args.dry_run)
            total_embedded += embedded

    logger.info(f"\n=== Summary ===")
    logger.info(f"Files ingested to Postgres: {total_files}")
    logger.info(f"Chunks ingested to Postgres: {total_chunks}")
    logger.info(f"Chunks embedded to ChromaDB: {total_embedded}")


if __name__ == "__main__":
    main()
