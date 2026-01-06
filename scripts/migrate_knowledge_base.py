#!/usr/bin/env python3
"""
Safe Migration Script: Upgrade to BGE-M3 + LLM-based Concept Extraction

This script upgrades the Polymath knowledge base without destructive deletion:
1. Re-embeds all passages with BGE-M3 (1024-dim) into new ChromaDB collection
2. Re-extracts concepts using LocalEntityExtractor (Ollama)
3. MERGEs concept relationships in Neo4j with source_model tagging

The old data remains intact for verification before cleanup.

Usage:
    python scripts/migrate_knowledge_base.py --dry-run          # Preview only
    python scripts/migrate_knowledge_base.py --apply            # Run migration
    python scripts/migrate_knowledge_base.py --resume           # Resume from checkpoint
    python scripts/migrate_knowledge_base.py --verify           # Verify migration
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
import psycopg2
import chromadb

from lib.config import (
    CHROMADB_PATH,
    POSTGRES_DSN,
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    PAPERS_COLLECTION,
    CODE_COLLECTION,
    PAPERS_COLLECTION_LEGACY,
    CODE_COLLECTION_LEGACY,
)
from lib.local_extractor import LocalEntityExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Migration checkpoint file
CHECKPOINT_FILE = Path(__file__).parent.parent / "data" / "migration_checkpoint.json"


class MigrationState:
    """Track migration progress for resumability."""

    def __init__(self, checkpoint_file: Path = CHECKPOINT_FILE):
        self.checkpoint_file = checkpoint_file
        self.state = self._load()

    def _load(self) -> Dict:
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file) as f:
                return json.load(f)
        return {
            "started_at": None,
            "papers_migrated": 0,
            "code_migrated": 0,
            "concepts_extracted": 0,
            "neo4j_merged": 0,
            "last_paper_id": None,
            "last_code_id": None,
            "completed": False
        }

    def save(self):
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.state, f, indent=2, default=str)

    def update(self, **kwargs):
        self.state.update(kwargs)
        self.save()


class KnowledgeBaseMigrator:
    """Migrate Polymath knowledge base to BGE-M3 + LLM concepts."""

    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.state = MigrationState()

        # Initialize connections lazily
        self._pg_conn = None
        self._chroma_client = None
        self._neo4j_driver = None
        self._embedder = None
        self._extractor = None

    @property
    def pg_conn(self):
        if self._pg_conn is None:
            self._pg_conn = psycopg2.connect(POSTGRES_DSN)
        return self._pg_conn

    @property
    def chroma_client(self):
        if self._chroma_client is None:
            self._chroma_client = chromadb.PersistentClient(path=str(CHROMADB_PATH))
        return self._chroma_client

    @property
    def neo4j_driver(self):
        if self._neo4j_driver is None:
            from neo4j import GraphDatabase
            self._neo4j_driver = GraphDatabase.driver(
                NEO4J_URI,
                auth=(NEO4J_USER, NEO4J_PASSWORD) if NEO4J_PASSWORD else None
            )
        return self._neo4j_driver

    @property
    def embedder(self):
        if self._embedder is None:
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            if 'bge-m3' in EMBEDDING_MODEL.lower():
                try:
                    from FlagEmbedding import BGEM3FlagModel
                    self._embedder = BGEM3FlagModel(
                        EMBEDDING_MODEL,
                        use_fp16=True,
                        device='cuda'
                    )
                    self._use_flag_model = True
                    logger.info("Loaded BGEM3FlagModel with GPU acceleration")
                except ImportError:
                    logger.warning("FlagEmbedding not installed, using sentence-transformers")
                    from sentence_transformers import SentenceTransformer
                    self._embedder = SentenceTransformer(EMBEDDING_MODEL, device='cuda')
                    self._use_flag_model = False
            else:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(EMBEDDING_MODEL, device='cuda')
                self._use_flag_model = False
        return self._embedder

    @property
    def extractor(self):
        if self._extractor is None:
            self._extractor = LocalEntityExtractor()
        return self._extractor

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts."""
        if getattr(self, '_use_flag_model', False):
            result = self.embedder.encode(texts, return_dense=True)
            return result['dense_vecs'].tolist()
        else:
            return self.embedder.encode(texts).tolist()

    def _get_or_create_collection(self, name: str):
        """Get or create ChromaDB collection with correct dimensionality."""
        try:
            return self.chroma_client.get_collection(name)
        except Exception:
            return self.chroma_client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine", "dimension": EMBEDDING_DIM}
            )

    def fetch_passages(self, batch_size: int = 1000, offset: int = 0) -> List[Dict]:
        """Fetch passages from Postgres."""
        cursor = self.pg_conn.cursor()
        cursor.execute("""
            SELECT p.id, p.doc_id, p.chunk_index, p.content, p.page_num,
                   a.title, a.doi, a.year
            FROM passages p
            LEFT JOIN artifacts a ON p.doc_id = a.doc_id
            WHERE p.content IS NOT NULL AND LENGTH(p.content) > 50
            ORDER BY p.id
            LIMIT %s OFFSET %s
        """, (batch_size, offset))

        columns = ['id', 'doc_id', 'chunk_index', 'content', 'page_num',
                   'title', 'doi', 'year']
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def fetch_code_chunks(self, batch_size: int = 1000, offset: int = 0) -> List[Dict]:
        """Fetch code chunks from Postgres."""
        cursor = self.pg_conn.cursor()
        cursor.execute("""
            SELECT id, repo_name, file_path, chunk_type, content,
                   start_line, end_line, language
            FROM code_chunks
            WHERE content IS NOT NULL AND LENGTH(content) > 20
            ORDER BY id
            LIMIT %s OFFSET %s
        """, (batch_size, offset))

        columns = ['id', 'repo_name', 'file_path', 'chunk_type', 'content',
                   'start_line', 'end_line', 'language']
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def migrate_papers(self, batch_size: int = 100):
        """Migrate paper passages to new BGE-M3 collection."""
        logger.info(f"Migrating papers to collection: {PAPERS_COLLECTION}")

        collection = self._get_or_create_collection(PAPERS_COLLECTION)

        # Get total count
        cursor = self.pg_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM passages WHERE content IS NOT NULL AND LENGTH(content) > 50")
        total = cursor.fetchone()[0]

        offset = self.state.state.get('papers_migrated', 0)
        pbar = tqdm(total=total, initial=offset, desc="Papers")

        while offset < total:
            passages = self.fetch_passages(batch_size=batch_size, offset=offset)
            if not passages:
                break

            # Prepare batch
            ids = [f"p_{p['id']}" for p in passages]
            texts = [p['content'] for p in passages]
            metadatas = [{
                'doc_id': p['doc_id'],
                'chunk_index': p['chunk_index'],
                'page_num': p['page_num'] or -1,
                'title': p['title'] or '',
                'doi': p['doi'] or '',
                'year': p['year'] or 0,
                'source_model': 'bge_m3_v1'
            } for p in passages]

            if not self.dry_run:
                # Generate embeddings
                embeddings = self._embed_batch(texts)

                # Upsert to ChromaDB
                collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas
                )

            offset += len(passages)
            self.state.update(papers_migrated=offset)
            pbar.update(len(passages))

        pbar.close()
        logger.info(f"Papers migration complete: {offset} passages")
        return offset

    def migrate_code(self, batch_size: int = 100):
        """Migrate code chunks to new BGE-M3 collection."""
        logger.info(f"Migrating code to collection: {CODE_COLLECTION}")

        collection = self._get_or_create_collection(CODE_COLLECTION)

        # Get total count
        cursor = self.pg_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM code_chunks WHERE content IS NOT NULL AND LENGTH(content) > 20")
        total = cursor.fetchone()[0]

        offset = self.state.state.get('code_migrated', 0)
        pbar = tqdm(total=total, initial=offset, desc="Code")

        while offset < total:
            chunks = self.fetch_code_chunks(batch_size=batch_size, offset=offset)
            if not chunks:
                break

            # Prepare batch
            ids = [f"c_{c['id']}" for c in chunks]
            texts = [c['content'] for c in chunks]
            metadatas = [{
                'repo_name': c['repo_name'] or '',
                'file_path': c['file_path'] or '',
                'chunk_type': c['chunk_type'] or '',
                'start_line': c['start_line'] or 0,
                'end_line': c['end_line'] or 0,
                'language': c['language'] or '',
                'source_model': 'bge_m3_v1'
            } for c in chunks]

            if not self.dry_run:
                embeddings = self._embed_batch(texts)
                collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas
                )

            offset += len(chunks)
            self.state.update(code_migrated=offset)
            pbar.update(len(chunks))

        pbar.close()
        logger.info(f"Code migration complete: {offset} chunks")
        return offset

    def extract_and_link_concepts(self, batch_size: int = 50):
        """Extract concepts with LLM and link in Neo4j."""
        logger.info("Extracting concepts with LocalEntityExtractor")

        # Get documents that need concept extraction
        cursor = self.pg_conn.cursor()
        cursor.execute("""
            SELECT DISTINCT doc_id, title,
                   (SELECT content FROM passages WHERE doc_id = a.doc_id ORDER BY chunk_index LIMIT 1) as first_chunk
            FROM artifacts a
            WHERE a.doc_id IS NOT NULL
        """)
        docs = cursor.fetchall()

        concepts_count = self.state.state.get('concepts_extracted', 0)
        neo4j_count = self.state.state.get('neo4j_merged', 0)

        pbar = tqdm(docs, desc="Concepts", initial=concepts_count)

        for doc_id, title, first_chunk in pbar:
            if first_chunk is None:
                continue

            # Combine title and first chunk for context
            text = f"{title or ''}\n\n{first_chunk}"

            if not self.dry_run:
                # Extract concepts using LLM
                concepts = self.extractor.extract_concepts(text)

                if concepts:
                    # MERGE concepts in Neo4j
                    self._merge_concepts_neo4j(doc_id, concepts)
                    neo4j_count += len(concepts)

            concepts_count += 1
            self.state.update(concepts_extracted=concepts_count, neo4j_merged=neo4j_count)

        pbar.close()
        logger.info(f"Concept extraction complete: {concepts_count} docs, {neo4j_count} concept links")
        return concepts_count, neo4j_count

    def _merge_concepts_neo4j(self, doc_id: str, concepts: List[str]):
        """MERGE concept nodes and relationships in Neo4j."""
        with self.neo4j_driver.session() as session:
            for concept in concepts:
                # Normalize concept
                concept_norm = concept.lower().strip().replace(' ', '_')

                session.run("""
                    MERGE (c:Concept {name: $concept})
                    ON CREATE SET c.created_at = datetime()
                    WITH c
                    MERGE (d:Document {doc_id: $doc_id})
                    MERGE (d)-[r:MENTIONS]->(c)
                    ON CREATE SET r.source_model = 'llm_bge_v1', r.created_at = datetime()
                    ON MATCH SET r.source_model = 'llm_bge_v1', r.updated_at = datetime()
                """, concept=concept_norm, doc_id=doc_id)

    def verify_migration(self) -> Dict:
        """Verify migration integrity."""
        logger.info("Verifying migration...")

        results = {
            'papers_collection': {},
            'code_collection': {},
            'neo4j': {},
            'issues': []
        }

        # Check ChromaDB collections
        try:
            papers = self.chroma_client.get_collection(PAPERS_COLLECTION)
            results['papers_collection']['count'] = papers.count()

            # Sample check: verify embedding dimension
            sample = papers.get(limit=1, include=['embeddings'])
            if sample['embeddings']:
                dim = len(sample['embeddings'][0])
                results['papers_collection']['embedding_dim'] = dim
                if dim != EMBEDDING_DIM:
                    results['issues'].append(f"Papers embedding dim mismatch: {dim} vs {EMBEDDING_DIM}")
        except Exception as e:
            results['papers_collection']['error'] = str(e)
            results['issues'].append(f"Papers collection error: {e}")

        try:
            code = self.chroma_client.get_collection(CODE_COLLECTION)
            results['code_collection']['count'] = code.count()
        except Exception as e:
            results['code_collection']['error'] = str(e)

        # Check Neo4j
        try:
            with self.neo4j_driver.session() as session:
                result = session.run("""
                    MATCH (d:Document)-[r:MENTIONS {source_model: 'llm_bge_v1'}]->(c:Concept)
                    RETURN COUNT(r) as count
                """)
                results['neo4j']['llm_relationships'] = result.single()['count']

                # Count concepts
                result = session.run("MATCH (c:Concept) RETURN COUNT(c) as count")
                results['neo4j']['total_concepts'] = result.single()['count']
        except Exception as e:
            results['neo4j']['error'] = str(e)

        return results

    def run(self, resume: bool = False, verify_only: bool = False):
        """Run the full migration."""
        if verify_only:
            results = self.verify_migration()
            print(json.dumps(results, indent=2))
            return results

        if not resume:
            self.state.state = {
                "started_at": datetime.now().isoformat(),
                "papers_migrated": 0,
                "code_migrated": 0,
                "concepts_extracted": 0,
                "neo4j_merged": 0,
                "completed": False
            }
            self.state.save()

        logger.info(f"Starting migration (dry_run={self.dry_run}, resume={resume})")

        # Step 1: Migrate papers
        self.migrate_papers()

        # Step 2: Migrate code
        self.migrate_code()

        # Step 3: Extract concepts and link in Neo4j
        self.extract_and_link_concepts()

        # Mark complete
        self.state.update(completed=True, completed_at=datetime.now().isoformat())

        # Verify
        results = self.verify_migration()
        logger.info("Migration complete!")
        print("\n=== VERIFICATION RESULTS ===")
        print(json.dumps(results, indent=2))

        return results

    def close(self):
        """Close all connections."""
        if self._pg_conn:
            self._pg_conn.close()
        if self._neo4j_driver:
            self._neo4j_driver.close()


def main():
    parser = argparse.ArgumentParser(description="Migrate Polymath KB to BGE-M3 + LLM concepts")
    parser.add_argument('--dry-run', action='store_true', help="Preview without writing")
    parser.add_argument('--apply', action='store_true', help="Apply migration")
    parser.add_argument('--resume', action='store_true', help="Resume from checkpoint")
    parser.add_argument('--verify', action='store_true', help="Verify migration only")
    parser.add_argument('--batch-size', type=int, default=100, help="Batch size for processing")

    args = parser.parse_args()

    if not any([args.dry_run, args.apply, args.verify]):
        parser.print_help()
        print("\nError: Must specify --dry-run, --apply, or --verify")
        sys.exit(1)

    migrator = KnowledgeBaseMigrator(dry_run=not args.apply)

    try:
        migrator.run(resume=args.resume, verify_only=args.verify)
    finally:
        migrator.close()


if __name__ == "__main__":
    main()
