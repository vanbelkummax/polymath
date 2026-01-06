#!/usr/bin/env python3
"""
Optimized ChromaDB Rebuild v2

Key improvements:
1. TWO COLLECTIONS: polymath_papers, polymath_code
2. RICH METADATA: concepts, year, repo, language, chunk_type for filtering
3. BATCHED EMBEDDING with GPU acceleration
4. RESUMABLE with checkpoints

Collections:
- polymath_papers: 396K citable passages with year, concepts, section
- polymath_code: 412K code chunks with repo, language, chunk_type, concepts

Usage:
    python3 scripts/rebuild_chromadb_v2.py
    python3 scripts/rebuild_chromadb_v2.py --resume
    python3 scripts/rebuild_chromadb_v2.py --papers-only
    python3 scripts/rebuild_chromadb_v2.py --code-only
"""

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional
import hashlib

import chromadb
from chromadb.config import Settings
import psycopg2
from psycopg2.extras import RealDictCursor
from sentence_transformers import SentenceTransformer

# Config
CHROMA_PATH = "/home/user/work/polymax/chromadb"
POSTGRES_DSN = "dbname=polymath user=polymath host=/var/run/postgresql"

# Embedding models - can upgrade code model to unixcoder later
PAPER_MODEL = "all-mpnet-base-v2"  # Good for scientific text
CODE_MODEL = "all-mpnet-base-v2"   # Can switch to microsoft/unixcoder-base

CHECKPOINT_FILE = "/home/user/work/polymax/chromadb_rebuild_v2_checkpoint.json"
BATCH_SIZE = 1000  # Increased for better GPU utilization
ENCODE_BATCH = 256  # Larger encoding batches for GPU

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)


def get_postgres():
    return psycopg2.connect(POSTGRES_DSN)


def load_checkpoint() -> Dict:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {
        'papers_offset': 0,
        'code_offset': 0,
        'papers_done': False,
        'code_done': False
    }


def save_checkpoint(cp: Dict):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(cp, f)


def fetch_passages(conn, offset: int, limit: int) -> List[Dict]:
    """Fetch passages with rich metadata."""
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
                d.authors,
                d.doi,
                d.pmid,
                COALESCE(
                    (SELECT array_agg(c.name)
                     FROM artifact_concepts ac
                     JOIN concepts c ON ac.concept_id = c.id
                     JOIN artifacts a ON ac.artifact_id = a.id
                     WHERE a.doc_id = d.doc_id),
                    '{}'::text[]
                ) as concepts
            FROM passages p
            JOIN documents d ON p.doc_id = d.doc_id
            WHERE p.citable = true AND p.page_num >= 0
            ORDER BY p.passage_id
            OFFSET %s LIMIT %s
        """, (offset, limit))
        return [dict(row) for row in cur.fetchall()]


def fetch_code_chunks(conn, offset: int, limit: int) -> List[Dict]:
    """Fetch code chunks with rich metadata."""
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
                cf.head_commit_sha as commit_sha,
                cf.language
            FROM code_chunks cc
            JOIN code_files cf ON cc.file_id = cf.file_id
            ORDER BY cc.chunk_id
            OFFSET %s LIMIT %s
        """, (offset, limit))
        return [dict(row) for row in cur.fetchall()]


def prepare_paper(p: Dict) -> tuple:
    """Prepare paper passage for ChromaDB."""
    # Searchable text
    text = p['content']
    if p.get('title'):
        text = f"# {p['title']}\n\n{text}"

    # Rich metadata for filtering
    concepts = p.get('concepts') or []
    if isinstance(concepts, list):
        concepts_str = ','.join([c for c in concepts if c])
    else:
        concepts_str = ''

    metadata = {
        'source': 'paper',
        'doc_id': p['doc_id'] or '',
        'title': (p.get('title') or '')[:500],  # Truncate long titles
        'year': p.get('year') or 0,
        'page_num': p.get('page_num') or -1,
        'section': (p.get('section') or '')[:100],
        'concepts': concepts_str[:1000],  # For filtering
        'doi': (p.get('doi') or '')[:100],
        'pmid': (p.get('pmid') or '')[:20],
        'citable': True
    }

    return p['id'], text[:8000], metadata


def prepare_code(c: Dict) -> tuple:
    """Prepare code chunk for ChromaDB."""
    # Build searchable text with context
    parts = []
    if c.get('repo_name'):
        parts.append(f"Repository: {c['repo_name']}")
    if c.get('file_path'):
        parts.append(f"File: {c['file_path']}")
    if c.get('name'):
        parts.append(f"Name: {c['name']}")
    if c.get('class_name'):
        parts.append(f"Class: {c['class_name']}")
    if c.get('docstring'):
        parts.append(f"Docstring: {c['docstring'][:500]}")
    if c.get('signature'):
        parts.append(f"Signature: {c['signature']}")
    parts.append(f"\nCode:\n{c['content']}")

    text = "\n".join(parts)

    # Rich metadata
    concepts = c.get('concepts') or []
    if isinstance(concepts, list):
        concepts_str = ','.join([x for x in concepts if x])
    else:
        concepts_str = ''

    # Extract org from repo_name (e.g., "mahmoodlab/UNI" -> "mahmoodlab")
    repo_name = c.get('repo_name') or ''
    org = repo_name.split('/')[0] if '/' in repo_name else repo_name

    metadata = {
        'source': 'code',
        'repo_name': repo_name[:200],
        'org': org[:100],
        'file_path': (c.get('file_path') or '')[:300],
        'name': (c.get('name') or '')[:200],
        'chunk_type': (c.get('chunk_type') or '')[:50],
        'class_name': (c.get('class_name') or '')[:200],
        'language': (c.get('language') or 'python')[:20],
        'start_line': c.get('start_line') or 0,
        'end_line': c.get('end_line') or 0,
        'concepts': concepts_str[:1000],
        'has_docstring': bool(c.get('docstring'))
    }

    return c['id'], text[:8000], metadata


def rebuild_papers(client, embedder, conn, checkpoint: Dict):
    """Rebuild papers collection."""

    # Get or create collection
    try:
        collection = client.get_or_create_collection(
            name="polymath_papers",
            metadata={"hnsw:space": "cosine", "description": "Scientific papers - 396K passages"}
        )
    except Exception as e:
        logger.error(f"Error creating papers collection: {e}")
        raise

    # Get total count
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM passages WHERE citable = true AND page_num >= 0")
        total = cur.fetchone()[0]

    logger.info(f"Papers collection: {collection.count()} existing, {total} total to index")

    offset = checkpoint['papers_offset']
    start_time = time.time()

    while offset < total:
        batch = fetch_passages(conn, offset, BATCH_SIZE)
        if not batch:
            break

        ids, texts, metadatas = [], [], []
        for p in batch:
            try:
                pid, text, meta = prepare_paper(p)
                ids.append(pid)
                texts.append(text)
                metadatas.append(meta)
            except Exception as e:
                logger.warning(f"Error preparing passage: {e}")

        if texts:
            # Embed batch - use large batch for GPU
            embeddings = embedder.encode(texts, show_progress_bar=False, batch_size=ENCODE_BATCH).tolist()

            # Upsert
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )

        offset += BATCH_SIZE
        checkpoint['papers_offset'] = offset
        save_checkpoint(checkpoint)

        # Progress
        elapsed = time.time() - start_time
        rate = (offset - checkpoint.get('papers_start', 0)) / elapsed if elapsed > 0 else 0
        eta = (total - offset) / rate if rate > 0 else 0
        logger.info(f"Papers: {offset:,}/{total:,} ({100*offset/total:.1f}%) Rate: {rate:.1f}/s ETA: {eta/60:.1f}min")

    checkpoint['papers_done'] = True
    save_checkpoint(checkpoint)
    logger.info(f"Papers collection complete: {collection.count()} items")


def rebuild_code(client, embedder, conn, checkpoint: Dict):
    """Rebuild code collection."""

    # Get or create collection
    try:
        collection = client.get_or_create_collection(
            name="polymath_code",
            metadata={"hnsw:space": "cosine", "description": "Code chunks - 412K from 118 repos"}
        )
    except Exception as e:
        logger.error(f"Error creating code collection: {e}")
        raise

    # Get total count
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM code_chunks")
        total = cur.fetchone()[0]

    logger.info(f"Code collection: {collection.count()} existing, {total} total to index")

    offset = checkpoint['code_offset']
    start_time = time.time()

    while offset < total:
        batch = fetch_code_chunks(conn, offset, BATCH_SIZE)
        if not batch:
            break

        ids, texts, metadatas = [], [], []
        for c in batch:
            try:
                cid, text, meta = prepare_code(c)
                ids.append(cid)
                texts.append(text)
                metadatas.append(meta)
            except Exception as e:
                logger.warning(f"Error preparing code chunk: {e}")

        if texts:
            # Embed batch - use large batch for GPU
            embeddings = embedder.encode(texts, show_progress_bar=False, batch_size=ENCODE_BATCH).tolist()

            # Upsert
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )

        offset += BATCH_SIZE
        checkpoint['code_offset'] = offset
        save_checkpoint(checkpoint)

        # Progress
        elapsed = time.time() - start_time
        rate = (offset - checkpoint.get('code_start', 0)) / elapsed if elapsed > 0 else 0
        eta = (total - offset) / rate if rate > 0 else 0
        logger.info(f"Code: {offset:,}/{total:,} ({100*offset/total:.1f}%) Rate: {rate:.1f}/s ETA: {eta/60:.1f}min")

    checkpoint['code_done'] = True
    save_checkpoint(checkpoint)
    logger.info(f"Code collection complete: {collection.count()} items")


def main():
    parser = argparse.ArgumentParser(description="Optimized ChromaDB Rebuild v2")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--papers-only", action="store_true", help="Only rebuild papers")
    parser.add_argument("--code-only", action="store_true", help="Only rebuild code")
    args = parser.parse_args()

    # Load or init checkpoint
    if args.resume:
        checkpoint = load_checkpoint()
        logger.info(f"Resuming from checkpoint: papers={checkpoint['papers_offset']}, code={checkpoint['code_offset']}")
    else:
        checkpoint = {
            'papers_offset': 0, 'papers_start': 0, 'papers_done': False,
            'code_offset': 0, 'code_start': 0, 'code_done': False
        }

    # Init ChromaDB
    logger.info(f"Initializing ChromaDB at {CHROMA_PATH}")
    os.makedirs(CHROMA_PATH, exist_ok=True)
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False)
    )

    # Init embedder
    logger.info(f"Loading embedding model: {PAPER_MODEL}")
    embedder = SentenceTransformer(PAPER_MODEL, device='cuda')

    # Init Postgres
    conn = get_postgres()

    try:
        # Rebuild papers
        if not args.code_only and not checkpoint.get('papers_done'):
            checkpoint['papers_start'] = checkpoint['papers_offset']
            rebuild_papers(client, embedder, conn, checkpoint)

        # Rebuild code
        if not args.papers_only and not checkpoint.get('code_done'):
            checkpoint['code_start'] = checkpoint['code_offset']
            rebuild_code(client, embedder, conn, checkpoint)

        # Summary
        logger.info("="*60)
        logger.info("REBUILD COMPLETE")
        try:
            papers_coll = client.get_collection("polymath_papers")
            code_coll = client.get_collection("polymath_code")
            logger.info(f"polymath_papers: {papers_coll.count():,} items")
            logger.info(f"polymath_code: {code_coll.count():,} items")
            logger.info(f"TOTAL: {papers_coll.count() + code_coll.count():,} searchable items")
        except Exception as e:
            logger.warning(f"Could not get final counts: {e}")
        logger.info("="*60)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
