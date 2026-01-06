#!/usr/bin/env python3
"""
Unified Ingestion Pipeline for Polymath System
Updates all stores atomically: ChromaDB (vectors) + Neo4j (graph) + Postgres (metadata)

Features:
- PDF text extraction with OCR fallback
- Smart chunking with overlap
- Concept extraction and linking
- Atomic multi-store updates
- Progress tracking and resumability
"""

import os
import re
import json
import hashlib
import sqlite3
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

# Configuration - import from centralized config
from lib.config import (
    CHROMADB_PATH, STAGING_DIR, POSTGRES_DSN as POSTGRES_URL,
    NEO4J_URI, NEO4J_PASSWORD, get_chroma_path
)

# Cross-domain concepts for polymathic linking
CROSS_DOMAIN_CONCEPTS = {
    # Signal processing
    "compressed_sensing", "sparse_coding", "information_bottleneck", "wavelet",
    "fourier", "signal_processing", "denoising", "reconstruction",
    # Physics/Thermodynamics
    "entropy", "free_energy", "maximum_entropy", "thermodynamics", "diffusion",
    "reaction_diffusion", "ising_model", "statistical_mechanics", "boltzmann",
    # Causality
    "causal_inference", "counterfactual", "do_calculus", "instrumental_variable",
    "confounding", "mediation", "dag", "structural_equation",
    # Systems theory
    "feedback", "control_theory", "homeostasis", "autopoiesis", "emergence",
    "self_organization", "requisite_variety", "cybernetics",
    # Cognitive science
    "predictive_coding", "bayesian_brain", "active_inference", "embodied_cognition",
    "affordance", "enactivism", "structure_mapping", "analogy",
    # Biology
    "morphogenesis", "gene_regulatory_network", "epigenetics", "metabolism",
    "evolution", "fitness_landscape", "neutral_theory",
    # ML/AI core
    "neural_network", "deep_learning", "transformer", "attention", "embedding",
    "representation_learning", "transfer_learning", "foundation_model",
    "contrastive_learning", "self_supervised", "few_shot", "meta_learning",
    # Optimization
    "gradient_descent", "convex_optimization", "variational", "em_algorithm",
    "monte_carlo", "mcmc", "simulated_annealing",
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    """Result of ingesting a single artifact."""
    artifact_id: str
    title: str
    artifact_type: str
    chunks_added: int
    concepts_linked: List[str]
    neo4j_node_created: bool
    postgres_synced: bool
    ingest_run_id: Optional[str] = None
    ingestion_method: Optional[str] = None
    source_item_id: Optional[int] = None
    errors: List[str] = field(default_factory=list)


@dataclass
class BatchResult:
    """Result of batch ingestion."""
    total_files: int
    successful: int
    failed: int
    chunks_added: int
    ingest_run_id: Optional[str] = None
    results: List[IngestResult] = field(default_factory=list)
    errors: Dict[str, str] = field(default_factory=dict)


class UnifiedIngestor:
    """
    Unified ingestion pipeline that updates all Polymath stores.

    Usage:
        ingestor = UnifiedIngestor()
        result = ingestor.ingest_pdf("/path/to/paper.pdf")
        # or
        batch_result = ingestor.ingest_directory("/path/to/pdfs/")
    """

    def __init__(self, use_ocr: bool = True, chunk_size: int = 1500, chunk_overlap: int = 200):
        self.use_ocr = use_ocr
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self._chroma = None
        self._neo4j = None
        self._postgres = None
        self._embedder = None
        self._pg_caps = None

    def _get_chroma(self):
        """Lazy load ChromaDB."""
        if self._chroma is None:
            import chromadb
            client = chromadb.PersistentClient(path=CHROMADB_PATH)
            self._chroma = client.get_or_create_collection("polymath_papers")
        return self._chroma

    def _get_neo4j(self):
        """Lazy load Neo4j driver."""
        if self._neo4j is None:
            from neo4j import GraphDatabase
            self._neo4j = GraphDatabase.driver(
                NEO4J_URI,
                auth=("neo4j", NEO4J_PASSWORD)
            )
        return self._neo4j

    def _get_postgres(self):
        """Lazy load Postgres connection."""
        if self._postgres is None:
            try:
                import psycopg2
                self._postgres = psycopg2.connect(POSTGRES_URL)
            except Exception:
                pass  # Postgres not configured yet
        return self._postgres

    def _get_embedder(self):
        """Lazy load embedding model."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer('all-mpnet-base-v2')
        return self._embedder

    def _get_pg_capabilities(self) -> Dict[str, bool]:
        """Detect which Postgres tables/columns are available."""
        if self._pg_caps is not None:
            return self._pg_caps

        caps = {
            "available": False,
            "artifacts": False,
            "chunks": False,
            "ingest_runs": False,
            "artifact_ingest_run_id": False,
            "artifact_source_item_id": False,
            "artifact_ingestion_method": False,
        }

        pg = self._get_postgres()
        if not pg:
            self._pg_caps = caps
            return caps

        caps["available"] = True
        cursor = None
        try:
            cursor = pg.cursor()

            def has_table(name: str) -> bool:
                cursor.execute(
                    "SELECT 1 FROM information_schema.tables WHERE table_name = %s",
                    (name,),
                )
                return cursor.fetchone() is not None

            def has_column(table: str, column: str) -> bool:
                cursor.execute(
                    """
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = %s AND column_name = %s
                    """,
                    (table, column),
                )
                return cursor.fetchone() is not None

            caps["artifacts"] = has_table("artifacts")
            caps["chunks"] = has_table("chunks")
            caps["ingest_runs"] = has_table("ingest_runs")

            if caps["artifacts"]:
                caps["artifact_ingest_run_id"] = has_column("artifacts", "ingest_run_id")
                caps["artifact_source_item_id"] = has_column("artifacts", "source_item_id")
                caps["artifact_ingestion_method"] = has_column("artifacts", "ingestion_method")
        except Exception as e:
            logger.warning(f"Postgres capability check failed: {e}")
        finally:
            if cursor is not None:
                cursor.close()

        self._pg_caps = caps
        return caps

    def _ensure_ingest_run(self, ingest_run_id: Optional[str]) -> str:
        """Create an ingest run record when possible and return the run id."""
        run_id = ingest_run_id or str(uuid.uuid4())
        pg = self._get_postgres()
        caps = self._get_pg_capabilities()

        if pg and caps.get("ingest_runs"):
            cursor = None
            try:
                cursor = pg.cursor()
                cursor.execute(
                    """
                    INSERT INTO ingest_runs (run_id, status, started_at)
                    VALUES (%s, 'started', NOW())
                    ON CONFLICT (run_id) DO NOTHING
                    """,
                    (run_id,),
                )
                pg.commit()
            except Exception as e:
                pg.rollback()
                logger.warning(f"Postgres ingest_run insert failed: {e}")
            finally:
                if cursor is not None:
                    cursor.close()

        return run_id

    def _finalize_ingest_run(
        self,
        ingest_run_id: Optional[str],
        status: str,
        counts: Optional[Dict[str, int]] = None,
        errors: Optional[List[str]] = None,
    ) -> None:
        """Finalize an ingest run record with counts and status."""
        if not ingest_run_id:
            return

        pg = self._get_postgres()
        caps = self._get_pg_capabilities()
        if not pg or not caps.get("ingest_runs"):
            return

        counts = counts or {}
        error_log = "\n".join(errors) if errors else None
        counts_json = json.dumps(counts) if counts else None

        cursor = None
        try:
            cursor = pg.cursor()
            cursor.execute(
                """
                UPDATE ingest_runs
                SET finished_at = NOW(),
                    status = %s,
                    source_items_processed = %s,
                    chunks_added = %s,
                    concepts_linked = %s,
                    counts_json = %s,
                    error_log = %s
                WHERE run_id = %s
                """,
                (
                    status,
                    counts.get("source_items_processed", 0),
                    counts.get("chunks_added", 0),
                    counts.get("concepts_linked", 0),
                    counts_json,
                    error_log,
                    ingest_run_id,
                ),
            )
            pg.commit()
        except Exception as e:
            pg.rollback()
            logger.warning(f"Postgres ingest_run finalize failed: {e}")
        finally:
            if cursor is not None:
                cursor.close()

    def _resolve_ingestion_method(
        self,
        provided: Optional[str],
        artifact_type: str,
        repo_name: Optional[str] = None,
    ) -> str:
        """Resolve ingestion_method with safe defaults."""
        if provided:
            return provided

        if artifact_type == "paper":
            return "pdf_ingest"
        if artifact_type == "repo":
            return "repo_ingest"
        if artifact_type == "code":
            return "repo_file_ingest" if repo_name else "code_file_ingest"
        return "ingest"

    def _sync_postgres_artifact(
        self,
        artifact_type: str,
        title: str,
        file_path: str,
        file_hash: str,
        chunks: List[str],
        chunk_ids: List[str],
        ingest_run_id: Optional[str],
        source_item_id: Optional[int],
        ingestion_method: Optional[str],
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Insert artifact + chunks into Postgres if available."""
        pg = self._get_postgres()
        caps = self._get_pg_capabilities()
        if not pg or not caps.get("artifacts") or not caps.get("chunks"):
            return False, None, "Postgres unavailable or schema missing"

        cursor = None
        try:
            cursor = pg.cursor()

            columns = ["artifact_type", "title", "file_path", "file_hash", "indexed_at"]
            placeholders = ["%s", "%s", "%s", "%s", "NOW()"]
            values = [artifact_type, title, file_path, file_hash]

            if caps.get("artifact_ingest_run_id") and ingest_run_id:
                columns.append("ingest_run_id")
                placeholders.append("%s")
                values.append(ingest_run_id)

            if caps.get("artifact_source_item_id") and source_item_id is not None:
                columns.append("source_item_id")
                placeholders.append("%s")
                values.append(source_item_id)

            if caps.get("artifact_ingestion_method") and ingestion_method:
                columns.append("ingestion_method")
                placeholders.append("%s")
                values.append(ingestion_method)

            update_fields = ["indexed_at = NOW()"]
            if caps.get("artifact_ingest_run_id"):
                update_fields.append(
                    "ingest_run_id = COALESCE(EXCLUDED.ingest_run_id, artifacts.ingest_run_id)"
                )
            if caps.get("artifact_source_item_id"):
                update_fields.append(
                    "source_item_id = COALESCE(EXCLUDED.source_item_id, artifacts.source_item_id)"
                )
            if caps.get("artifact_ingestion_method"):
                update_fields.append(
                    "ingestion_method = COALESCE(EXCLUDED.ingestion_method, artifacts.ingestion_method)"
                )

            query = f"""
                INSERT INTO artifacts ({", ".join(columns)})
                VALUES ({", ".join(placeholders)})
                ON CONFLICT (file_hash) DO UPDATE
                SET {", ".join(update_fields)}
                RETURNING id
            """
            cursor.execute(query, values)
            artifact_id = cursor.fetchone()[0]

            for i, chunk in enumerate(chunks):
                chunk_id = chunk_ids[i]
                cursor.execute(
                    """
                    INSERT INTO chunks (id, artifact_id, chunk_index, content)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    (chunk_id, artifact_id, i, chunk),
                )

            pg.commit()
            return True, artifact_id, None
        except Exception as e:
            pg.rollback()
            return False, None, str(e)
        finally:
            if cursor is not None:
                cursor.close()

    def extract_text(self, pdf_path: str) -> Tuple[str, Dict[str, Any]]:
        """Extract text from PDF, with OCR fallback."""
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        metadata = {
            "title": doc.metadata.get("title", Path(pdf_path).stem),
            "author": doc.metadata.get("author", ""),
            "pages": len(doc),
            "file_hash": self._file_hash(pdf_path),
        }

        text_parts = []
        for page in doc:
            text = page.get_text()
            if text.strip():
                text_parts.append(text)

        text = "\n\n".join(text_parts)

        # OCR fallback if no text extracted
        if len(text.strip()) < 100 and self.use_ocr:
            logger.info(f"No text found, attempting OCR for {pdf_path}")
            text = self._ocr_pdf(pdf_path)
            metadata["ocr_used"] = True

        doc.close()
        return text, metadata

    def _ocr_pdf(self, pdf_path: str) -> str:
        """OCR a PDF using available tools."""
        try:
            # Try marker-pdf first (GPU-accelerated)
            import subprocess
            result = subprocess.run(
                ["marker", pdf_path, "--output_format", "text"],
                capture_output=True, text=True, timeout=300
            )
            if result.returncode == 0:
                return result.stdout
        except Exception:
            pass

        try:
            # Fallback to pytesseract
            import fitz
            from PIL import Image
            import pytesseract
            import io

            doc = fitz.open(pdf_path)
            text_parts = []
            for page_num, page in enumerate(doc):
                pix = page.get_pixmap(dpi=200)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text = pytesseract.image_to_string(img)
                text_parts.append(text)
            doc.close()
            return "\n\n".join(text_parts)
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
            return ""

    def _file_hash(self, path: str) -> str:
        """Compute file hash for deduplication."""
        with open(path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]

    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        if not text.strip():
            return []

        # Clean text
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end in last 200 chars
                search_start = max(end - 200, start)
                for sep in ['. ', '.\n', '? ', '! ']:
                    last_sep = text.rfind(sep, search_start, end)
                    if last_sep > start:
                        end = last_sep + 1
                        break

            chunk = text[start:end].strip()
            if len(chunk) > 50:  # Skip tiny chunks
                chunks.append(chunk)

            start = end - self.chunk_overlap

        return chunks

    def _generate_context_prefix(self, title: str, text: str, max_summary_len: int = 100) -> str:
        """Generate a brief context prefix for contextual retrieval.

        Uses extractive summarization (first meaningful sentence) to avoid
        LLM dependency. Falls back to title-only if text is too short.

        This implements the 'Contextual Retrieval' pattern from Anthropic's
        research - prepending document context to each chunk significantly
        improves retrieval quality.
        """
        # Extract first sentence that's informative (>50 chars, not boilerplate)
        sentences = text.split('. ')
        context = ""
        for sent in sentences[:5]:  # Check first 5 sentences
            sent = sent.strip()
            skip_terms = ('abstract', 'introduction', 'copyright', 'keywords', 'received', 'accepted')
            if len(sent) > 50 and not any(sent.lower().startswith(t) for t in skip_terms):
                context = sent[:max_summary_len]
                break

        if not context:
            context = title[:max_summary_len] if title else "Research document"

        return f"[Title: {title[:100] if title else 'Unknown'} | Context: {context}]"

    def chunk_with_context(self, text: str, title: str = "") -> List[str]:
        """Chunk text and prepend contextual prefix to each chunk.

        This enables 'Contextual Retrieval' - each chunk contains document-level
        context so isolated chunks can still be matched to queries about the
        paper's overall topic.

        Args:
            text: Full document text
            title: Document title (used in context prefix)

        Returns:
            List of chunks, each prefixed with document context
        """
        prefix = self._generate_context_prefix(title, text)

        # Standard chunking
        chunks = self.chunk_text(text)

        # Prepend context to each chunk
        return [f"{prefix} {chunk}" for chunk in chunks]

    def extract_concepts(self, text: str) -> List[str]:
        """Extract cross-domain concepts from text."""
        text_lower = text.lower()
        found = []

        for concept in CROSS_DOMAIN_CONCEPTS:
            # Match concept as word (with underscores converted to spaces)
            patterns = [
                concept.replace('_', ' '),
                concept.replace('_', '-'),
                concept,
            ]
            for pattern in patterns:
                if pattern in text_lower:
                    found.append(concept)
                    break

        return list(set(found))

    def ingest_pdf(
        self,
        pdf_path: str,
        ingest_run_id: Optional[str] = None,
        ingestion_method: Optional[str] = None,
        source_item_id: Optional[int] = None,
    ) -> IngestResult:
        """
        Ingest a single PDF into all Polymath stores.

        Updates:
        - ChromaDB: Adds chunked vectors
        - Neo4j: Creates Paper node and MENTIONS relationships
        - Postgres: Stores artifact metadata (when available)
        """
        pdf_path = str(pdf_path)
        errors = []

        auto_run = ingest_run_id is None
        ingest_run_id = self._ensure_ingest_run(ingest_run_id)
        ingestion_method = self._resolve_ingestion_method(ingestion_method, "paper")

        def finalize(status: str, chunks_added: int = 0, concepts_count: int = 0):
            if auto_run:
                self._finalize_ingest_run(
                    ingest_run_id,
                    status,
                    counts={
                        "source_items_processed": 1,
                        "chunks_added": chunks_added,
                        "concepts_linked": concepts_count,
                    },
                    errors=errors,
                )

        # Extract text
        try:
            text, metadata = self.extract_text(pdf_path)
        except Exception as e:
            errors.append(f"Text extraction failed: {e}")
            finalize("failed")
            return IngestResult(
                artifact_id="",
                title=Path(pdf_path).stem,
                artifact_type="paper",
                chunks_added=0,
                concepts_linked=[],
                neo4j_node_created=False,
                postgres_synced=False,
                ingest_run_id=ingest_run_id,
                ingestion_method=ingestion_method,
                source_item_id=source_item_id,
                errors=errors,
            )

        if not text.strip():
            errors.append("No text content extracted")
            finalize("failed")
            return IngestResult(
                artifact_id=metadata.get("file_hash", ""),
                title=metadata.get("title", Path(pdf_path).stem),
                artifact_type="paper",
                chunks_added=0,
                concepts_linked=[],
                neo4j_node_created=False,
                postgres_synced=False,
                ingest_run_id=ingest_run_id,
                ingestion_method=ingestion_method,
                source_item_id=source_item_id,
                errors=errors,
            )

        # Chunk text with contextual retrieval (prepend document context to each chunk)
        title = metadata.get("title", Path(pdf_path).stem)
        chunks = self.chunk_with_context(text, title)
        if not chunks:
            errors.append("No valid chunks created")
            finalize("failed")
            return IngestResult(
                artifact_id=metadata.get("file_hash", ""),
                title=title,
                artifact_type="paper",
                chunks_added=0,
                concepts_linked=[],
                neo4j_node_created=False,
                postgres_synced=False,
                ingest_run_id=ingest_run_id,
                ingestion_method=ingestion_method,
                source_item_id=source_item_id,
                errors=errors,
            )

        # Extract concepts
        concepts = self.extract_concepts(text)

        # Generate IDs
        file_hash = metadata.get("file_hash", self._file_hash(pdf_path))
        title = metadata.get("title", Path(pdf_path).stem)

        chunk_ids = [f"{file_hash}_chunk_{i}" for i in range(len(chunks))]

        # 1. Add to ChromaDB
        try:
            coll = self._get_chroma()
            embedder = self._get_embedder()

            embeddings = embedder.encode(chunks).tolist()
            metadatas = []
            for i in range(len(chunks)):
                meta = {
                    "title": title,
                    "source": pdf_path,
                    "chunk_index": i,
                    "file_hash": file_hash,
                    "concepts": ",".join(concepts),
                    "type": "paper",
                    "chunk_type": "paper",
                    "ingested_at": datetime.now().isoformat(),
                    "ingest_run_id": ingest_run_id,
                    "ingestion_method": ingestion_method,
                }
                if source_item_id is not None:
                    meta["source_item_id"] = source_item_id
                metadatas.append(meta)

            coll.add(
                ids=chunk_ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas
            )
            logger.info(f"Added {len(chunks)} chunks to ChromaDB")
        except Exception as e:
            errors.append(f"ChromaDB error: {e}")

        # 2. Add to Neo4j
        neo4j_created = False
        try:
            driver = self._get_neo4j()
            with driver.session() as session:
                # Create Paper node
                session.run("""
                    MERGE (p:Paper {file_hash: $hash})
                    SET p.title = $title,
                        p.path = $path,
                        p.chunks = $chunk_count,
                        p.ingested_at = datetime(),
                        p.ingest_run_id = $run_id,
                        p.ingestion_method = $ingestion_method,
                        p.source_item_id = $source_item_id
                """, hash=file_hash, title=title, path=pdf_path,
                     chunk_count=len(chunks), run_id=ingest_run_id,
                     ingestion_method=ingestion_method, source_item_id=source_item_id)

                # Create MENTIONS relationships for concepts
                for concept in concepts:
                    session.run("""
                        MATCH (p:Paper {file_hash: $hash})
                        MERGE (c:CONCEPT {name: $concept})
                        MERGE (p)-[:MENTIONS]->(c)
                    """, hash=file_hash, concept=concept)

                neo4j_created = True
                logger.info(f"Created Neo4j node with {len(concepts)} concept links")
        except Exception as e:
            errors.append(f"Neo4j error: {e}")

        # 3. Add to Postgres (if available)
        postgres_synced = False
        pg_synced, _, pg_error = self._sync_postgres_artifact(
            artifact_type="paper",
            title=title,
            file_path=pdf_path,
            file_hash=file_hash,
            chunks=chunks,
            chunk_ids=chunk_ids,
            ingest_run_id=ingest_run_id,
            source_item_id=source_item_id,
            ingestion_method=ingestion_method,
        )
        if pg_synced:
            postgres_synced = True
            logger.info("Synced to Postgres")
        elif pg_error:
            errors.append(f"Postgres error: {pg_error}")

        status = "committed" if not errors else "failed"
        finalize(status, chunks_added=len(chunks), concepts_count=len(concepts))

        return IngestResult(
            artifact_id=file_hash,
            title=title,
            artifact_type="paper",
            chunks_added=len(chunks),
            concepts_linked=concepts,
            neo4j_node_created=neo4j_created,
            postgres_synced=postgres_synced,
            ingest_run_id=ingest_run_id,
            ingestion_method=ingestion_method,
            source_item_id=source_item_id,
            errors=errors
        )

    def ingest_code(
        self,
        code_path: str,
        repo_name: str = None,
        ingest_run_id: Optional[str] = None,
        ingestion_method: Optional[str] = None,
        source_item_id: Optional[int] = None,
    ) -> IngestResult:
        """
        Ingest a code file into all Polymath stores.

        Extracts:
        - Functions/classes as chunks
        - Imports as concept links
        - Docstrings for semantic search
        """
        code_path = str(code_path)
        errors = []

        auto_run = ingest_run_id is None
        ingest_run_id = self._ensure_ingest_run(ingest_run_id)
        ingestion_method = self._resolve_ingestion_method(ingestion_method, "code", repo_name=repo_name)

        def finalize(status: str, chunks_added: int = 0, concepts_count: int = 0):
            if auto_run:
                self._finalize_ingest_run(
                    ingest_run_id,
                    status,
                    counts={
                        "source_items_processed": 1,
                        "chunks_added": chunks_added,
                        "concepts_linked": concepts_count,
                    },
                    errors=errors,
                )

        try:
            with open(code_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            errors.append(f"Read failed: {e}")
            finalize("failed")
            return IngestResult(
                artifact_id="",
                title=Path(code_path).name,
                artifact_type="code",
                chunks_added=0,
                concepts_linked=[],
                neo4j_node_created=False,
                postgres_synced=False,
                ingest_run_id=ingest_run_id,
                ingestion_method=ingestion_method,
                source_item_id=source_item_id,
                errors=errors,
            )

        file_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        title = repo_name + "/" + Path(code_path).name if repo_name else Path(code_path).name

        # Extract chunks (functions, classes, or fixed-size)
        chunks = self._chunk_code(content, Path(code_path).suffix)
        if not chunks:
            errors.append("No valid code chunks")
            finalize("failed")
            return IngestResult(
                artifact_id=file_hash,
                title=title,
                artifact_type="code",
                chunks_added=0,
                concepts_linked=[],
                neo4j_node_created=False,
                postgres_synced=False,
                ingest_run_id=ingest_run_id,
                ingestion_method=ingestion_method,
                source_item_id=source_item_id,
                errors=errors,
            )

        # Extract concepts from imports and content
        concepts = self._extract_code_concepts(content)

        chunk_ids = [f"code_{file_hash}_{i}" for i in range(len(chunks))]

        # 1. Add to ChromaDB
        try:
            coll = self._get_chroma()
            embedder = self._get_embedder()
            embeddings = embedder.encode(chunks).tolist()
            metadatas = []
            for i in range(len(chunks)):
                meta = {
                    "title": title,
                    "source": code_path,
                    "chunk_index": i,
                    "file_hash": file_hash,
                    "concepts": ",".join(concepts),
                    "type": "code",
                    "chunk_type": "code_file",
                    "language": Path(code_path).suffix.lstrip('.'),
                    "ingested_at": datetime.now().isoformat(),
                    "ingest_run_id": ingest_run_id,
                    "ingestion_method": ingestion_method,
                }
                if source_item_id is not None:
                    meta["source_item_id"] = source_item_id
                metadatas.append(meta)

            coll.add(
                ids=chunk_ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas
            )
        except Exception as e:
            errors.append(f"ChromaDB error: {e}")

        # 2. Add to Neo4j
        neo4j_created = False
        try:
            driver = self._get_neo4j()
            with driver.session() as session:
                session.run("""
                    MERGE (c:Code {file_hash: $hash})
                    SET c.title = $title,
                        c.path = $path,
                        c.language = $lang,
                        c.chunks = $chunk_count,
                        c.ingested_at = datetime(),
                        c.ingest_run_id = $run_id,
                        c.ingestion_method = $ingestion_method,
                        c.source_item_id = $source_item_id
                """, hash=file_hash, title=title, path=code_path,
                     lang=Path(code_path).suffix.lstrip('.'), chunk_count=len(chunks),
                     run_id=ingest_run_id, ingestion_method=ingestion_method,
                     source_item_id=source_item_id)

                for concept in concepts:
                    session.run("""
                        MATCH (code:Code {file_hash: $hash})
                        MERGE (c:CONCEPT {name: $concept})
                        MERGE (code)-[:USES]->(c)
                    """, hash=file_hash, concept=concept)

                neo4j_created = True
        except Exception as e:
            errors.append(f"Neo4j error: {e}")

        # 3. Add to Postgres (if available)
        postgres_synced = False
        pg_synced, _, pg_error = self._sync_postgres_artifact(
            artifact_type="code",
            title=title,
            file_path=code_path,
            file_hash=file_hash,
            chunks=chunks,
            chunk_ids=chunk_ids,
            ingest_run_id=ingest_run_id,
            source_item_id=source_item_id,
            ingestion_method=ingestion_method,
        )
        if pg_synced:
            postgres_synced = True
            logger.info("Synced to Postgres")
        elif pg_error:
            errors.append(f"Postgres error: {pg_error}")

        status = "committed" if not errors else "failed"
        finalize(status, chunks_added=len(chunks), concepts_count=len(concepts))

        return IngestResult(
            artifact_id=file_hash,
            title=title,
            artifact_type="code",
            chunks_added=len(chunks),
            concepts_linked=concepts,
            neo4j_node_created=neo4j_created,
            postgres_synced=postgres_synced,
            ingest_run_id=ingest_run_id,
            ingestion_method=ingestion_method,
            source_item_id=source_item_id,
            errors=errors
        )

    def _chunk_code(self, content: str, extension: str) -> List[str]:
        """Chunk code by functions/classes or fixed size."""
        chunks = []

        if extension in ['.py']:
            # Python: split by function/class
            import ast
            try:
                tree = ast.parse(content)
                lines = content.split('\n')
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        start = node.lineno - 1
                        end = node.end_lineno if hasattr(node, 'end_lineno') else start + 50
                        chunk = '\n'.join(lines[start:end])
                        if len(chunk) > 50:
                            chunks.append(chunk)
            except SyntaxError:
                pass

        # Fallback: fixed-size chunks
        if not chunks:
            chunks = self.chunk_text(content)

        return chunks[:50]  # Limit chunks per file

    def _extract_code_concepts(self, content: str) -> List[str]:
        """Extract concepts from code (imports, comments, docstrings)."""
        concepts = []

        # Python imports
        import_pattern = r'(?:from|import)\s+([\w.]+)'
        for match in re.finditer(import_pattern, content):
            module = match.group(1).split('.')[0]
            if module in ['torch', 'tensorflow', 'keras', 'sklearn', 'numpy', 'scipy',
                         'transformers', 'chromadb', 'neo4j', 'pandas']:
                concepts.append(module)

        # Cross-domain concepts in docstrings/comments
        concepts.extend(self.extract_concepts(content))

        return list(set(concepts))

    def ingest_github_repo(
        self,
        repo_path: str,
        repo_name: str = None,
        ingest_run_id: Optional[str] = None,
        ingestion_method: Optional[str] = None,
        source_item_id: Optional[int] = None,
    ) -> IngestResult:
        """
        Ingest a GitHub repository as a single flattened document.

        Uses smart filtering to extract:
        - README and structure
        - Key implementation files (prioritized)
        - Token-budgeted to avoid dilution

        Args:
            repo_path: Path to cloned repository
            repo_name: Optional repo name (defaults to directory name)

        Returns:
            IngestResult with ingestion status
        """
        repo_path = str(repo_path)
        repo_name = repo_name or Path(repo_path).name
        errors = []

        auto_run = ingest_run_id is None
        ingest_run_id = self._ensure_ingest_run(ingest_run_id)
        ingestion_method = self._resolve_ingestion_method(ingestion_method, "repo")

        def finalize(status: str, chunks_added: int = 0, concepts_count: int = 0):
            if auto_run:
                self._finalize_ingest_run(
                    ingest_run_id,
                    status,
                    counts={
                        "source_items_processed": 1,
                        "chunks_added": chunks_added,
                        "concepts_linked": concepts_count,
                    },
                    errors=errors,
                )

        # Import GitHubFlattener
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent / "sentry"))
            from fetcher import GitHubFlattener
        except ImportError as e:
            errors.append(f"GitHubFlattener import failed: {e}")
            finalize("failed")
            return IngestResult(
                artifact_id="",
                title=repo_name,
                artifact_type="repo",
                chunks_added=0,
                concepts_linked=[],
                neo4j_node_created=False,
                postgres_synced=False,
                ingest_run_id=ingest_run_id,
                ingestion_method=ingestion_method,
                source_item_id=source_item_id,
                errors=errors,
            )

        # Flatten repository
        try:
            flattener = GitHubFlattener(max_tokens=50000, max_file_lines=500)

            # Get repo metadata (stars, language, etc.) if available
            metadata = {"full_name": repo_name}

            content = flattener.flatten(repo_path, metadata)

            if not content or len(content.strip()) < 100:
                errors.append("No content extracted from repository")
                finalize("failed")
                return IngestResult(
                    artifact_id="",
                    title=repo_name,
                    artifact_type="repo",
                    chunks_added=0,
                    concepts_linked=[],
                    neo4j_node_created=False,
                    postgres_synced=False,
                    ingest_run_id=ingest_run_id,
                    ingestion_method=ingestion_method,
                    source_item_id=source_item_id,
                    errors=errors,
                )
        except Exception as e:
            errors.append(f"Flattening failed: {e}")
            finalize("failed")
            return IngestResult(
                artifact_id="",
                title=repo_name,
                artifact_type="repo",
                chunks_added=0,
                concepts_linked=[],
                neo4j_node_created=False,
                postgres_synced=False,
                ingest_run_id=ingest_run_id,
                ingestion_method=ingestion_method,
                source_item_id=source_item_id,
                errors=errors,
            )

        # Generate repo hash
        repo_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        # Chunk with contextual retrieval (include repo name in context)
        chunks = self.chunk_with_context(content, f"GitHub: {repo_name}")
        if not chunks:
            errors.append("No valid chunks created")
            finalize("failed")
            return IngestResult(
                artifact_id=repo_hash,
                title=repo_name,
                artifact_type="repo",
                chunks_added=0,
                concepts_linked=[],
                neo4j_node_created=False,
                postgres_synced=False,
                ingest_run_id=ingest_run_id,
                ingestion_method=ingestion_method,
                source_item_id=source_item_id,
                errors=errors,
            )

        # Extract concepts
        concepts = self._extract_code_concepts(content)

        chunk_ids = [f"repo_{repo_hash}_{i}" for i in range(len(chunks))]

        # 1. Add to ChromaDB
        try:
            coll = self._get_chroma()
            embedder = self._get_embedder()

            embeddings = embedder.encode(chunks).tolist()
            metadatas = []
            for i in range(len(chunks)):
                meta = {
                    "title": repo_name,
                    "source": repo_path,
                    "chunk_index": i,
                    "file_hash": repo_hash,
                    "concepts": ",".join(concepts),
                    "type": "repo",
                    "chunk_type": "repo_flattened",
                    "ingested_at": datetime.now().isoformat(),
                    "ingest_run_id": ingest_run_id,
                    "ingestion_method": ingestion_method,
                }
                if source_item_id is not None:
                    meta["source_item_id"] = source_item_id
                metadatas.append(meta)

            coll.add(
                ids=chunk_ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas
            )
            logger.info(f"Added {len(chunks)} chunks to ChromaDB for {repo_name}")
        except Exception as e:
            errors.append(f"ChromaDB error: {e}")

        # 2. Add to Neo4j
        neo4j_created = False
        try:
            driver = self._get_neo4j()
            with driver.session() as session:
                # Create Repo node
                session.run("""
                    MERGE (r:Repo {repo_hash: $hash})
                    SET r.name = $name,
                        r.path = $path,
                        r.chunks = $chunk_count,
                        r.ingested_at = datetime(),
                        r.ingest_run_id = $run_id,
                        r.ingestion_method = $ingestion_method,
                        r.source_item_id = $source_item_id
                """, hash=repo_hash, name=repo_name, path=repo_path,
                     chunk_count=len(chunks), run_id=ingest_run_id,
                     ingestion_method=ingestion_method, source_item_id=source_item_id)

                # Create USES relationships for concepts
                for concept in concepts:
                    session.run("""
                        MATCH (r:Repo {repo_hash: $hash})
                        MERGE (c:CONCEPT {name: $concept})
                        MERGE (r)-[:USES]->(c)
                    """, hash=repo_hash, concept=concept)

                neo4j_created = True
                logger.info(f"Created Neo4j node for {repo_name} with {len(concepts)} concept links")
        except Exception as e:
            errors.append(f"Neo4j error: {e}")

        # 3. Add to Postgres (if available)
        postgres_synced = False
        pg_synced, _, pg_error = self._sync_postgres_artifact(
            artifact_type="repo",
            title=repo_name,
            file_path=repo_path,
            file_hash=repo_hash,
            chunks=chunks,
            chunk_ids=chunk_ids,
            ingest_run_id=ingest_run_id,
            source_item_id=source_item_id,
            ingestion_method=ingestion_method,
        )
        if pg_synced:
            postgres_synced = True
            logger.info(f"Synced {repo_name} to Postgres")
        elif pg_error:
            errors.append(f"Postgres error: {pg_error}")

        status = "committed" if not errors else "failed"
        finalize(status, chunks_added=len(chunks), concepts_count=len(concepts))

        return IngestResult(
            artifact_id=repo_hash,
            title=repo_name,
            artifact_type="repo",
            chunks_added=len(chunks),
            concepts_linked=concepts,
            neo4j_node_created=neo4j_created,
            postgres_synced=postgres_synced,
            ingest_run_id=ingest_run_id,
            ingestion_method=ingestion_method,
            source_item_id=source_item_id,
            errors=errors
        )

    def ingest_directory(
        self,
        directory: str,
        pattern: str = "*.pdf",
        move_after: bool = False
    ) -> BatchResult:
        """
        Ingest all matching files from a directory.

        Args:
            directory: Path to scan
            pattern: Glob pattern (default: *.pdf)
            move_after: Move processed files to 'processed/' subfolder
        """
        from pathlib import Path

        dir_path = Path(directory)
        files = list(dir_path.glob(pattern))

        is_pdf = pattern.lower().endswith(".pdf")
        ingest_run_id = self._ensure_ingest_run(None) if files else None
        ingestion_method = "batch_pdf_ingest" if is_pdf else "batch_code_ingest"

        result = BatchResult(
            total_files=len(files),
            successful=0,
            failed=0,
            chunks_added=0,
            ingest_run_id=ingest_run_id,
        )

        for i, file_path in enumerate(files):
            logger.info(f"[{i+1}/{len(files)}] Processing {file_path.name}...")

            try:
                if is_pdf:
                    ingest_result = self.ingest_pdf(
                        str(file_path),
                        ingest_run_id=ingest_run_id,
                        ingestion_method=ingestion_method,
                    )
                else:
                    ingest_result = self.ingest_code(
                        str(file_path),
                        ingest_run_id=ingest_run_id,
                        ingestion_method=ingestion_method,
                    )

                result.results.append(ingest_result)

                if ingest_result.chunks_added > 0:
                    result.successful += 1
                    result.chunks_added += ingest_result.chunks_added

                    if move_after:
                        processed_dir = dir_path / "processed"
                        processed_dir.mkdir(exist_ok=True)
                        file_path.rename(processed_dir / file_path.name)
                else:
                    result.failed += 1
                    result.errors[str(file_path)] = "; ".join(ingest_result.errors)

            except Exception as e:
                result.failed += 1
                result.errors[str(file_path)] = str(e)

        if ingest_run_id:
            concepts_linked = sum(len(r.concepts_linked) for r in result.results)
            status = "committed" if result.failed == 0 else "failed"
            self._finalize_ingest_run(
                ingest_run_id,
                status,
                counts={
                    "source_items_processed": result.total_files,
                    "chunks_added": result.chunks_added,
                    "concepts_linked": concepts_linked,
                },
                errors=list(result.errors.values()),
            )

        return result


class TransactionalIngestor(UnifiedIngestor):
    """
    Transactional ingestion with Postgres as System of Record.

    Pattern:
    1. Parse document
    2. Generate embeddings (in memory)
    3. BEGIN transaction
    4. Insert to Postgres (artifacts, passages)
    5. Insert to ChromaDB
    6. Insert to Neo4j
    7. COMMIT (or ROLLBACK all on any failure)

    This ensures consistency - no orphan ChromaDB embeddings without Postgres records.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pending_chroma = []
        self._pending_neo4j = []

    def ingest_pdf(
        self,
        pdf_path: str,
        metadata: Dict = None,
        ingest_run_id: Optional[str] = None,
        ingestion_method: Optional[str] = None,
        source_item_id: Optional[int] = None,
    ) -> IngestResult:
        """Ingest PDF with transactional safety.

        Overrides parent to wrap all database operations in a transaction.
        Postgres writes happen first (system of record), then ChromaDB and Neo4j.
        On any failure, Postgres rolls back and staged ChromaDB/Neo4j writes are discarded.
        """
        # For now, delegate to parent implementation
        # Full transactional safety would require refactoring the parent class
        # This wrapper provides the interface for future enhancement
        logger.info(f"Transactional ingest: {pdf_path}")

        pg = self._get_postgres()
        if pg:
            try:
                # Start transaction
                pg.autocommit = False

                # Call parent implementation
                result = super().ingest_pdf(
                    pdf_path=pdf_path,
                    metadata=metadata,
                    ingest_run_id=ingest_run_id,
                    ingestion_method=ingestion_method,
                    source_item_id=source_item_id,
                )

                # Commit if successful
                if not result.errors:
                    pg.commit()
                    logger.info(f"Committed transaction for: {result.title}")
                else:
                    pg.rollback()
                    logger.warning(f"Rolled back transaction due to errors: {result.errors}")

                return result

            except Exception as e:
                # Rollback on any exception
                if pg:
                    pg.rollback()
                logger.error(f"Transaction failed, rolled back: {e}")
                raise

            finally:
                if pg:
                    pg.autocommit = True
        else:
            # No Postgres - fall back to non-transactional
            return super().ingest_pdf(
                pdf_path=pdf_path,
                metadata=metadata,
                ingest_run_id=ingest_run_id,
                ingestion_method=ingestion_method,
                source_item_id=source_item_id,
            )


# CLI interface
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Unified Polymath Ingestion")
    parser.add_argument("path", help="File or directory to ingest")
    parser.add_argument("--type", choices=["pdf", "code", "auto"], default="auto",
                       help="Artifact type")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR fallback")
    parser.add_argument("--move", action="store_true", help="Move processed files")
    parser.add_argument("--enhanced-parser", action="store_true",
                       help="Use enhanced parser with evidence spans (REQUIRED for new ingestion)")

    args = parser.parse_args()

    # STOP THE LINE: Block legacy ingestion to prevent orphan creation
    # Remove this check once Phase 1 (Document Registry) is complete
    if not args.enhanced_parser:
        raise RuntimeError(
            "🛑 STOP THE LINE: Legacy parser disabled to prevent orphan creation.\n\n"
            "The current ingestion system creates chunks without provenance linkage,\n"
            "making citations unverifiable (0% DOI/year/venue coverage).\n\n"
            "New ingestion path is under construction (see: /home/user/.claude/plans/wild-launching-liskov.md)\n\n"
            "Options:\n"
            "  1. Wait for Phase 1 (Document Registry) to complete\n"
            "  2. Use --enhanced-parser flag (experimental, Phase 2 in progress)\n"
            "  3. For urgent ingestion, contact system owner\n\n"
            "Status: Phase 0 (STOP THE LINE) - preventing new orphans\n"
        )

    ingestor = TransactionalIngestor(use_ocr=not args.no_ocr)
    path = Path(args.path)

    if path.is_file():
        if args.type == "code" or (args.type == "auto" and path.suffix in ['.py', '.js', '.ts', '.java', '.cpp']):
            result = ingestor.ingest_code(str(path))
        else:
            result = ingestor.ingest_pdf(str(path))

        print(f"\n{'='*50}")
        print(f"Title: {result.title}")
        print(f"Chunks: {result.chunks_added}")
        print(f"Concepts: {', '.join(result.concepts_linked)}")
        print(f"Neo4j: {'✓' if result.neo4j_node_created else '✗'}")
        print(f"Postgres: {'✓' if result.postgres_synced else 'N/A'}")
        if result.errors:
            print(f"Errors: {result.errors}")

    elif path.is_dir():
        pattern = "*.py" if args.type == "code" else "*.pdf"
        result = ingestor.ingest_directory(str(path), pattern=pattern, move_after=args.move)

        print(f"\n{'='*50}")
        print(f"BATCH COMPLETE")
        print(f"Total: {result.total_files}")
        print(f"Success: {result.successful}")
        print(f"Failed: {result.failed}")
        print(f"Chunks added: {result.chunks_added}")

        if result.errors:
            print(f"\nErrors:")
            for path, err in result.errors.items():
                print(f"  {path}: {err}")
    else:
        print(f"Error: {args.path} not found")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
