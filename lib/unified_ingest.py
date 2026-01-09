#!/usr/bin/env python3
"""
Unified Ingestion Pipeline for Polymath System
Postgres-first updates with best-effort ChromaDB + Neo4j indexing

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
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

# Configuration - import from centralized config
from lib.config import (
    CHROMADB_PATH, STAGING_DIR, POSTGRES_DSN as POSTGRES_URL,
    NEO4J_URI, NEO4J_PASSWORD, PAPERS_COLLECTION, CODE_COLLECTION,
    EMBEDDING_MODEL
)

# Local LLM-based entity extraction (replaces regex-based extraction)
from lib.local_extractor import LocalEntityExtractor
from lib.enhanced_pdf_parser import EnhancedPDFParser, Passage
from lib.doc_identity import compute_doc_id, upsert_document

try:
    from psycopg2.extras import execute_values
except Exception:  # pragma: no cover - optional dependency at runtime
    execute_values = None

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
class CodeChunk:
    """Lightweight code chunk with provenance metadata."""
    text: str
    chunk_type: str
    name: str
    class_name: Optional[str]
    start_line: int
    end_line: int
    chunk_id: Optional[str] = None
    chunk_hash: Optional[str] = None


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

        self._papers_coll = None
        self._code_coll = None
        self._neo4j = None
        self._postgres = None
        self._embedder = None
        self._pg_caps = None

        # LLM-based entity extractor (replaces regex)
        self.extractor = LocalEntityExtractor()
        self._pdf_parser = EnhancedPDFParser()
        self._use_bge_m3 = False

    def _get_papers_collection(self):
        """Lazy load ChromaDB papers collection (BGE-M3)."""
        if self._papers_coll is None:
            import chromadb
            client = chromadb.PersistentClient(path=str(CHROMADB_PATH))
            self._papers_coll = client.get_or_create_collection(PAPERS_COLLECTION)
        return self._papers_coll

    def _get_code_collection(self):
        """Lazy load ChromaDB code collection (BGE-M3)."""
        if self._code_coll is None:
            import chromadb
            client = chromadb.PersistentClient(path=str(CHROMADB_PATH))
            self._code_coll = client.get_or_create_collection(CODE_COLLECTION)
        return self._code_coll

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
        """Lazy load embedding model with BGE-M3 support."""
        if self._embedder is None:
            model_name = EMBEDDING_MODEL
            if 'bge-m3' in model_name.lower():
                try:
                    from FlagEmbedding import BGEM3FlagModel
                    self._embedder = BGEM3FlagModel(
                        model_name,
                        use_fp16=True,
                        device='cuda'
                    )
                    self._use_bge_m3 = True
                    logger.info(f"Loaded BGE-M3 model (FlagEmbedding): {model_name}")
                except ImportError:
                    from sentence_transformers import SentenceTransformer
                    self._embedder = SentenceTransformer(model_name, device='cuda')
                    self._use_bge_m3 = False
                    logger.warning("FlagEmbedding not installed, using sentence-transformers fallback")
            else:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(model_name, device='cuda')
                self._use_bge_m3 = False
                logger.info(f"Loaded embedding model: {model_name}")
        return self._embedder

    def _encode_texts(self, texts: List[str]) -> List[List[float]]:
        """Encode texts to embedding vectors."""
        embedder = self._get_embedder()
        if getattr(self, '_use_bge_m3', False):
            result = embedder.encode(texts, return_dense=True)
            return result['dense_vecs'].tolist()
        embeddings = embedder.encode(texts)
        return embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings

    def _get_pg_capabilities(self) -> Dict[str, bool]:
        """Detect which Postgres tables/columns are available."""
        if self._pg_caps is not None:
            return self._pg_caps

        caps = {
            "available": False,
            "artifacts": False,
            "chunks": False,
            "documents": False,
            "doc_aliases": False,
            "passages": False,
            "code_files": False,
            "code_chunks": False,
            "ingest_runs": False,
            "artifact_doc_id": False,
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
            caps["documents"] = has_table("documents")
            caps["doc_aliases"] = has_table("doc_aliases")
            caps["passages"] = has_table("passages")
            caps["code_files"] = has_table("code_files")
            caps["code_chunks"] = has_table("code_chunks")
            caps["ingest_runs"] = has_table("ingest_runs")

            if caps["artifacts"]:
                caps["artifact_doc_id"] = has_column("artifacts", "doc_id")
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
        doc_id: Optional[str] = None,
        insert_chunks: bool = True,
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Insert artifact + chunks into Postgres if available."""
        pg = self._get_postgres()
        caps = self._get_pg_capabilities()
        if not pg or not caps.get("artifacts") or (insert_chunks and not caps.get("chunks")):
            return False, None, "Postgres unavailable or schema missing"

        cursor = None
        try:
            cursor = pg.cursor()

            columns = ["artifact_type", "title", "file_path", "file_hash", "indexed_at"]
            placeholders = ["%s", "%s", "%s", "%s", "NOW()"]
            values = [artifact_type, title, file_path, file_hash]

            if caps.get("artifact_doc_id") and doc_id:
                columns.append("doc_id")
                placeholders.append("%s")
                values.append(doc_id)

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

            if insert_chunks:
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

    def _split_authors(self, raw: str) -> List[str]:
        """Split author string into a list without breaking 'Last, First'."""
        if not raw:
            return []
        parts = re.split(r';|\s+and\s+', raw)
        return [p.strip() for p in parts if p.strip()]

    def _extract_year(self, metadata: Dict[str, Any], text: str) -> int:
        """Extract a publication year from metadata or text."""
        for key in ("creationDate", "modDate"):
            value = metadata.get(key, "")
            match = re.search(r'(19|20)\d{2}', value)
            if match:
                return int(match.group(0))
        match = re.search(r'\b(19|20)\d{2}\b', text)
        return int(match.group(0)) if match else 0

    def _extract_identifiers(self, text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Extract DOI, PMID, and arXiv IDs from text."""
        doi = None
        pmid = None
        arxiv_id = None

        doi_match = re.search(r'\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b', text, flags=re.I)
        if doi_match:
            doi = doi_match.group(0).rstrip(').,;')

        pmid_match = re.search(r'\bPMID[:\s]*([0-9]{4,10})\b', text, flags=re.I)
        if pmid_match:
            pmid = pmid_match.group(1)

        arxiv_match = re.search(r'\barXiv[:\s]*([0-9]{4}\.[0-9]{4,5})(v\d+)?\b', text, flags=re.I)
        if arxiv_match:
            arxiv_id = arxiv_match.group(1)

        return doi, pmid, arxiv_id

    def _extract_pdf_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """Extract basic PDF metadata and identifiers."""
        metadata = {
            "title": Path(pdf_path).stem,
            "author": "",
            "authors": [],
            "pages": 0,
            "file_hash": self._file_hash(pdf_path),
            "doi": None,
            "pmid": None,
            "arxiv_id": None,
            "year": 0,
        }

        first_page_text = ""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            metadata["pages"] = len(doc)
            meta = doc.metadata or {}
            metadata["title"] = meta.get("title") or metadata["title"]
            metadata["author"] = meta.get("author") or ""
            if len(doc) > 0:
                first_page_text = doc[0].get_text() or ""
            doc.close()
        except Exception as e:
            logger.warning(f"PDF metadata read failed: {e}")

        metadata["authors"] = self._split_authors(metadata.get("author", ""))
        metadata["year"] = self._extract_year(meta if 'meta' in locals() else {}, first_page_text)
        doi, pmid, arxiv_id = self._extract_identifiers(first_page_text)
        metadata["doi"] = doi
        metadata["pmid"] = pmid
        metadata["arxiv_id"] = arxiv_id

        return metadata

    def _sync_postgres_passages(self, passages: List[Passage]) -> Tuple[bool, Optional[str]]:
        """Insert citation-eligible passages into Postgres."""
        pg = self._get_postgres()
        caps = self._get_pg_capabilities()
        if not pg or not caps.get("passages"):
            return False, "Postgres passages table unavailable"

        if not passages:
            return False, "No passages to insert"

        cursor = None
        try:
            cursor = pg.cursor()
            rows = [
                (
                    str(p.passage_id),
                    str(p.doc_id),
                    p.page_num,
                    p.page_char_start,
                    p.page_char_end,
                    p.section,
                    p.passage_text,
                    p.quality_score,
                    p.parser_version,
                )
                for p in passages
            ]

            if execute_values is None:
                for row in rows:
                    cursor.execute(
                        """
                        INSERT INTO passages
                        (passage_id, doc_id, page_num, page_char_start, page_char_end,
                         section, passage_text, quality_score, parser_version)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (passage_id) DO NOTHING
                        """,
                        row,
                    )
            else:
                execute_values(
                    cursor,
                    """
                    INSERT INTO passages
                    (passage_id, doc_id, page_num, page_char_start, page_char_end,
                     section, passage_text, quality_score, parser_version)
                    VALUES %s
                    ON CONFLICT (passage_id) DO NOTHING
                    """,
                    rows,
                )

            pg.commit()
            return True, None
        except Exception as e:
            pg.rollback()
            return False, str(e)
        finally:
            if cursor is not None:
                cursor.close()

    def _sync_postgres_code_file(
        self,
        repo_name: str,
        repo_url: Optional[str],
        repo_root: str,
        default_branch: str,
        commit_sha: str,
        rel_path: str,
        language: str,
        file_hash: str,
        file_size: int,
        loc: int,
        modified_time: Optional[datetime],
        chunks: List[CodeChunk],
        concepts: List[str],
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Insert code file + chunks into Postgres."""
        pg = self._get_postgres()
        caps = self._get_pg_capabilities()
        if not pg or not caps.get("code_files") or not caps.get("code_chunks"):
            return False, None, "Postgres code tables unavailable"

        cursor = None
        try:
            cursor = pg.cursor()
            cursor.execute(
                """
                INSERT INTO code_files
                (repo_name, repo_url, repo_root, default_branch, head_commit_sha,
                 file_path, language, file_hash, file_size_bytes, loc, modified_time)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (repo_name, file_path, head_commit_sha) DO UPDATE
                SET file_hash = EXCLUDED.file_hash,
                    file_size_bytes = EXCLUDED.file_size_bytes,
                    loc = EXCLUDED.loc,
                    modified_time = EXCLUDED.modified_time
                RETURNING file_id
                """,
                (
                    repo_name,
                    repo_url,
                    repo_root,
                    default_branch,
                    commit_sha,
                    rel_path,
                    language,
                    file_hash,
                    file_size,
                    loc,
                    modified_time,
                ),
            )
            file_id = cursor.fetchone()[0]

            rows = []
            for chunk in chunks:
                rows.append(
                    (
                        chunk.chunk_id,
                        file_id,
                        chunk.chunk_type,
                        chunk.name,
                        chunk.class_name,
                        f"{rel_path}:{chunk.name}",
                        chunk.start_line,
                        chunk.end_line,
                        chunk.text,
                        chunk.chunk_hash,
                        None,
                        None,
                        [],
                        concepts,
                        f"c_{chunk.chunk_id}",
                        chunk.text,
                    )
                )

            if rows:
                if execute_values is None:
                    for row in rows:
                        cursor.execute(
                            """
                            INSERT INTO code_chunks
                            (chunk_id, file_id, chunk_type, name, class_name, symbol_qualified_name,
                             start_line, end_line, content, chunk_hash, docstring, signature,
                             imports, concepts, embedding_id, search_vector)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                    to_tsvector('english', %s))
                            ON CONFLICT (chunk_id) DO NOTHING
                            """,
                            row,
                        )
                else:
                    execute_values(
                        cursor,
                        """
                        INSERT INTO code_chunks
                        (chunk_id, file_id, chunk_type, name, class_name, symbol_qualified_name,
                         start_line, end_line, content, chunk_hash, docstring, signature,
                         imports, concepts, embedding_id, search_vector)
                        VALUES %s
                        ON CONFLICT (chunk_id) DO NOTHING
                        """,
                        rows,
                        template=(
                            "(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,"
                            " to_tsvector('english', %s))"
                        ),
                    )

            pg.commit()
            return True, str(file_id), None
        except Exception as e:
            pg.rollback()
            return False, None, str(e)
        finally:
            if cursor is not None:
                cursor.close()

    def _find_repo_root(self, path: Path) -> Optional[Path]:
        """Find the nearest git repo root for a path."""
        for parent in [path] + list(path.parents):
            if (parent / ".git").exists():
                return parent
        return None

    def _infer_repo_metadata(self, code_path: Path, repo_name: Optional[str]) -> Dict[str, Any]:
        """Infer repository metadata from a code file path."""
        repo_root = self._find_repo_root(code_path)
        repo_url = None
        branch = "main"
        commit_sha = "unknown"
        resolved_name = repo_name or (repo_root.name if repo_root else code_path.parent.name)

        if repo_root:
            try:
                import subprocess
                result = subprocess.run(
                    ["git", "-C", str(repo_root), "remote", "get-url", "origin"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    repo_url = result.stdout.strip()
                    match = re.search(r'github\.com[:/](.+?)(?:\.git)?$', repo_url)
                    if match:
                        resolved_name = match.group(1)

                result = subprocess.run(
                    ["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    branch = result.stdout.strip()

                result = subprocess.run(
                    ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    commit_sha = result.stdout.strip()
            except Exception as e:
                logger.debug(f"Git metadata extraction failed: {e}")

        return {
            "repo_name": resolved_name,
            "repo_root": str(repo_root or code_path.parent),
            "repo_url": repo_url,
            "default_branch": branch,
            "commit_sha": commit_sha,
        }

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
        """Extract concepts from text using local LLM.

        Uses Ollama with fast model (qwen3:4b) and heavy fallback (deepseek-r1:8b)
        for robust concept extraction. Concepts are normalized to snake_case.
        """
        concepts = self.extractor.extract_concepts(text)
        # Normalize: handle both dict format {name: ...} and string format
        result = []
        for c in concepts:
            if not c:
                continue
            # Handle dict format from LocalEntityExtractor
            if isinstance(c, dict):
                name = c.get('name', c.get('canonical', ''))
            else:
                name = str(c)
            if name:
                result.append(name.lower().strip())
        return result

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
        - Postgres: documents + passages (citation-eligible)
        - ChromaDB: Passage vectors (BGE-M3)
        - Neo4j: Paper node + concept links
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

        metadata = self._extract_pdf_metadata(pdf_path)
        title = metadata.get("title", Path(pdf_path).stem)
        authors = metadata.get("authors", [])
        year = metadata.get("year", 0)
        doi = metadata.get("doi")
        pmid = metadata.get("pmid")
        arxiv_id = metadata.get("arxiv_id")
        file_hash = metadata.get("file_hash", self._file_hash(pdf_path))

        # Deterministic document identity
        doc_id = compute_doc_id(
            title=title,
            authors=authors,
            year=year,
            doi=doi,
            pmid=pmid,
            arxiv_id=arxiv_id,
        )

        # Extract passages with page-local provenance
        try:
            passages = self._pdf_parser.extract_with_provenance(pdf_path, doc_id)
        except Exception as e:
            errors.append(f"Enhanced parsing failed: {e}")
            finalize("failed")
            return IngestResult(
                artifact_id="",
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

        if not passages:
            errors.append("No passages extracted")
            finalize("failed")
            return IngestResult(
                artifact_id=file_hash,
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

        # Extract concepts from title + first passage for speed
        concept_seed = f"{title}\n\n{passages[0].passage_text}"
        concepts = self.extract_concepts(concept_seed)

        # 1. Postgres (System of Record) - REQUIRED
        pg = self._get_postgres()
        caps = self._get_pg_capabilities()
        if not pg or not (caps.get("documents") and caps.get("passages")):
            errors.append("Postgres documents/passages tables unavailable - aborting ingest")
            finalize("failed")
            return IngestResult(
                artifact_id=file_hash,
                title=title,
                artifact_type="paper",
                chunks_added=0,
                concepts_linked=concepts,
                neo4j_node_created=False,
                postgres_synced=False,
                ingest_run_id=ingest_run_id,
                ingestion_method=ingestion_method,
                source_item_id=source_item_id,
                errors=errors,
            )

        postgres_synced = False
        try:
            upsert_document(
                doc_id=doc_id,
                title=title,
                authors=authors,
                year=year,
                doi=doi,
                pmid=pmid,
                arxiv_id=arxiv_id,
                parser_version=passages[0].parser_version if passages else None,
                db_conn=pg,
            )

            if caps.get("doc_aliases") and file_hash:
                cursor = pg.cursor()
                cursor.execute(
                    """
                    INSERT INTO doc_aliases (doc_id, alias_type, alias_value)
                    VALUES (%s, 'file_hash', %s)
                    ON CONFLICT (alias_type, alias_value) DO NOTHING
                    """,
                    (str(doc_id), file_hash),
                )
                pg.commit()
                cursor.close()
        except Exception as e:
            errors.append(f"Postgres document upsert failed: {e}")
            finalize("failed")
            return IngestResult(
                artifact_id=file_hash,
                title=title,
                artifact_type="paper",
                chunks_added=0,
                concepts_linked=concepts,
                neo4j_node_created=False,
                postgres_synced=False,
                ingest_run_id=ingest_run_id,
                ingestion_method=ingestion_method,
                source_item_id=source_item_id,
                errors=errors,
            )

        passages_ok, passages_error = self._sync_postgres_passages(passages)
        if not passages_ok:
            errors.append(f"Postgres passages insert failed: {passages_error}")
            finalize("failed")
            return IngestResult(
                artifact_id=file_hash,
                title=title,
                artifact_type="paper",
                chunks_added=0,
                concepts_linked=concepts,
                neo4j_node_created=False,
                postgres_synced=False,
                ingest_run_id=ingest_run_id,
                ingestion_method=ingestion_method,
                source_item_id=source_item_id,
                errors=errors,
            )

        postgres_synced = True

        # Optional: update artifacts table for legacy compatibility (no legacy chunks)
        pg_synced, _, pg_error = self._sync_postgres_artifact(
            artifact_type="paper",
            title=title,
            file_path=pdf_path,
            file_hash=file_hash,
            chunks=[],
            chunk_ids=[],
            ingest_run_id=ingest_run_id,
            source_item_id=source_item_id,
            ingestion_method=ingestion_method,
            doc_id=str(doc_id),
            insert_chunks=False,
        )
        if not pg_synced and pg_error:
            logger.warning(f"Postgres artifact sync failed: {pg_error}")

        # 2. ChromaDB (vector index)
        try:
            coll = self._get_papers_collection()
            texts = [p.passage_text for p in passages]
            embeddings = self._encode_texts(texts)

            metadatas = []
            for p in passages:
                meta = {
                    "doc_id": str(doc_id),
                    "title": title,
                    "doi": doi or "",
                    "pmid": pmid or "",
                    "year": year or 0,
                    "page_num": p.page_num,
                    "section": p.section,
                    "concepts": ",".join(concepts),
                    "parser_version": p.parser_version or "",
                    "source_model": "bge_m3_v1" if "bge-m3" in EMBEDDING_MODEL.lower() else EMBEDDING_MODEL,
                }
                if source_item_id is not None:
                    meta["source_item_id"] = source_item_id
                metadatas.append(meta)

            ids = [f"p_{p.passage_id}" for p in passages]
            coll.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
            logger.info(f"Added {len(passages)} passages to ChromaDB")
        except Exception as e:
            errors.append(f"ChromaDB error: {e}")

        # 3. Neo4j (graph)
        neo4j_created = False
        try:
            driver = self._get_neo4j()
            with driver.session() as session:
                session.run(
                    """
                    MERGE (p:Paper {doc_id: $doc_id})
                    SET p.title = $title,
                        p.year = $year,
                        p.doi = $doi,
                        p.path = $path,
                        p.ingested_at = datetime(),
                        p.ingest_run_id = $run_id,
                        p.ingestion_method = $ingestion_method,
                        p.source_item_id = $source_item_id
                    """,
                    doc_id=str(doc_id),
                    title=title,
                    year=year,
                    doi=doi,
                    path=pdf_path,
                    run_id=ingest_run_id,
                    ingestion_method=ingestion_method,
                    source_item_id=source_item_id,
                )

                for concept in concepts:
                    session.run(
                        """
                        MATCH (p:Paper {doc_id: $doc_id})
                        MERGE (c:CONCEPT {name: $concept})
                        MERGE (p)-[r:MENTIONS]->(c)
                        ON CREATE SET r.source_model = 'ingest_v1', r.created_at = datetime()
                        ON MATCH SET r.source_model = 'ingest_v1', r.updated_at = datetime()
                        """,
                        doc_id=str(doc_id),
                        concept=concept,
                    )

                neo4j_created = True
                logger.info(f"Created Neo4j node with {len(concepts)} concept links")
        except Exception as e:
            errors.append(f"Neo4j error: {e}")

        status = "committed" if not errors else "failed"
        finalize(status, chunks_added=len(passages), concepts_count=len(concepts))

        return IngestResult(
            artifact_id=str(doc_id),
            title=title,
            artifact_type="paper",
            chunks_added=len(passages),
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
        - Code files/chunks with line-level provenance
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

        repo_meta = self._infer_repo_metadata(Path(code_path), repo_name)
        repo_name = repo_meta["repo_name"]
        repo_root = Path(repo_meta["repo_root"])
        rel_path = Path(code_path).name
        try:
            rel_path = str(Path(code_path).relative_to(repo_root))
        except ValueError:
            pass
        language = Path(code_path).suffix.lstrip('.') or "unknown"
        title = f"{repo_name}/{Path(code_path).name}"

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

        for chunk in chunks:
            chunk_hash = hashlib.sha256(chunk.text.encode()).hexdigest()[:16]
            chunk_id = uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"code:{file_hash}:{chunk_hash}:{chunk.start_line}:{chunk.end_line}"
            )
            chunk.chunk_hash = chunk_hash
            chunk.chunk_id = str(chunk_id)

        # Extract concepts from imports and content
        concepts = self._extract_code_concepts(content)

        # 1. Postgres (System of Record) - REQUIRED
        try:
            stat = Path(code_path).stat()
            file_size = stat.st_size
            modified_time = datetime.fromtimestamp(stat.st_mtime)
        except Exception:
            file_size = len(content.encode())
            modified_time = None

        loc = content.count('\n') + 1

        pg_synced, _, pg_error = self._sync_postgres_code_file(
            repo_name=repo_name,
            repo_url=repo_meta.get("repo_url"),
            repo_root=str(repo_root),
            default_branch=repo_meta.get("default_branch", "main"),
            commit_sha=repo_meta.get("commit_sha", "unknown"),
            rel_path=rel_path,
            language=language,
            file_hash=file_hash,
            file_size=file_size,
            loc=loc,
            modified_time=modified_time,
            chunks=chunks,
            concepts=concepts,
        )
        if not pg_synced:
            errors.append(f"Postgres error: {pg_error}")
            finalize("failed")
            return IngestResult(
                artifact_id=file_hash,
                title=title,
                artifact_type="code",
                chunks_added=0,
                concepts_linked=concepts,
                neo4j_node_created=False,
                postgres_synced=False,
                ingest_run_id=ingest_run_id,
                ingestion_method=ingestion_method,
                source_item_id=source_item_id,
                errors=errors,
            )

        postgres_synced = True

        # Optional: update artifacts table for legacy compatibility (no legacy chunks)
        legacy_ok, _, legacy_error = self._sync_postgres_artifact(
            artifact_type="code",
            title=title,
            file_path=code_path,
            file_hash=file_hash,
            chunks=[],
            chunk_ids=[],
            ingest_run_id=ingest_run_id,
            source_item_id=source_item_id,
            ingestion_method=ingestion_method,
            insert_chunks=False,
        )
        if not legacy_ok and legacy_error:
            logger.warning(f"Postgres artifact sync failed: {legacy_error}")

        # 2. ChromaDB (vector index)
        try:
            coll = self._get_code_collection()
            texts = [c.text for c in chunks]
            embeddings = self._encode_texts(texts)

            org = repo_name.split('/')[0] if '/' in repo_name else repo_name
            metadatas = []
            for chunk in chunks:
                meta = {
                    "repo_name": repo_name,
                    "org": org,
                    "file_path": rel_path,
                    "name": chunk.name,
                    "chunk_type": chunk.chunk_type,
                    "language": language,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "concepts": ",".join(concepts),
                    "source_model": "bge_m3_v1" if "bge-m3" in EMBEDDING_MODEL.lower() else EMBEDDING_MODEL,
                }
                if source_item_id is not None:
                    meta["source_item_id"] = source_item_id
                metadatas.append(meta)

            ids = [f"c_{chunk.chunk_id}" for chunk in chunks]
            coll.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
            logger.info(f"Added {len(chunks)} code chunks to ChromaDB")
        except Exception as e:
            errors.append(f"ChromaDB error: {e}")

        # 3. Neo4j (graph)
        neo4j_created = False
        try:
            driver = self._get_neo4j()
            with driver.session() as session:
                session.run(
                    """
                    MERGE (c:Code {file_hash: $hash})
                    SET c.title = $title,
                        c.path = $path,
                        c.repo = $repo_name,
                        c.language = $lang,
                        c.chunks = $chunk_count,
                        c.ingested_at = datetime(),
                        c.ingest_run_id = $run_id,
                        c.ingestion_method = $ingestion_method,
                        c.source_item_id = $source_item_id
                    """,
                    hash=file_hash,
                    title=title,
                    path=code_path,
                    repo_name=repo_name,
                    lang=language,
                    chunk_count=len(chunks),
                    run_id=ingest_run_id,
                    ingestion_method=ingestion_method,
                    source_item_id=source_item_id,
                )

                for concept in concepts:
                    session.run(
                        """
                        MATCH (code:Code {file_hash: $hash})
                        MERGE (c:CONCEPT {name: $concept})
                        MERGE (code)-[:USES]->(c)
                        """,
                        hash=file_hash,
                        concept=concept,
                    )

                neo4j_created = True
        except Exception as e:
            errors.append(f"Neo4j error: {e}")

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

    def _chunk_code(self, content: str, extension: str) -> List[CodeChunk]:
        """Chunk code by functions/classes or fallback heuristics."""
        chunks: List[CodeChunk] = []

        if extension in ['.py']:
            import ast

            class ChunkVisitor(ast.NodeVisitor):
                def __init__(self, lines: List[str]):
                    self.lines = lines
                    self.chunks: List[CodeChunk] = []
                    self._class_stack: List[str] = []

                def _add_chunk(self, node, chunk_type: str, name: str, class_name: Optional[str]):
                    start = max(node.lineno - 1, 0)
                    end = node.end_lineno if hasattr(node, 'end_lineno') else start + 50
                    text = '\n'.join(self.lines[start:end])
                    if len(text) > 50:
                        self.chunks.append(CodeChunk(
                            text=text,
                            chunk_type=chunk_type,
                            name=name,
                            class_name=class_name,
                            start_line=start + 1,
                            end_line=end,
                        ))

                def visit_ClassDef(self, node):
                    self._add_chunk(node, "class", node.name, None)
                    self._class_stack.append(node.name)
                    self.generic_visit(node)
                    self._class_stack.pop()

                def visit_FunctionDef(self, node):
                    class_name = self._class_stack[-1] if self._class_stack else None
                    chunk_type = "method" if class_name else "function"
                    self._add_chunk(node, chunk_type, node.name, class_name)

                def visit_AsyncFunctionDef(self, node):
                    class_name = self._class_stack[-1] if self._class_stack else None
                    chunk_type = "method" if class_name else "function"
                    self._add_chunk(node, chunk_type, node.name, class_name)

            try:
                tree = ast.parse(content)
                lines = content.split('\n')
                visitor = ChunkVisitor(lines)
                visitor.visit(tree)
                chunks = visitor.chunks
            except SyntaxError:
                chunks = []

        if not chunks:
            # Fallback: approximate chunking
            chunk_texts = self.chunk_text(content)
            start_line = 1
            for chunk_text in chunk_texts:
                line_count = chunk_text.count('\n') + 1
                end_line = start_line + line_count - 1
                chunks.append(CodeChunk(
                    text=chunk_text,
                    chunk_type="module",
                    name="module",
                    class_name=None,
                    start_line=start_line,
                    end_line=end_line,
                ))
                start_line = end_line + 1

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

        # 1. Add to ChromaDB (code collection)
        try:
            coll = self._get_code_collection()
            embeddings = self._encode_texts(chunks)
            org = repo_name.split('/')[0] if '/' in repo_name else repo_name

            metadatas = []
            for i in range(len(chunks)):
                meta = {
                    "repo_name": repo_name,
                    "org": org,
                    "file_path": repo_path,
                    "name": f"{repo_name}_repo_summary",
                    "chunk_type": "repo_flattened",
                    "language": "mixed",
                    "concepts": ",".join(concepts),
                    "source_model": "bge_m3_v1" if "bge-m3" in EMBEDDING_MODEL.lower() else EMBEDDING_MODEL,
                }
                if source_item_id is not None:
                    meta["source_item_id"] = source_item_id
                metadatas.append(meta)

            coll.upsert(
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

        # 3. Add to Postgres (legacy artifact, no chunks)
        postgres_synced = False
        pg_synced, _, pg_error = self._sync_postgres_artifact(
            artifact_type="repo",
            title=repo_name,
            file_path=repo_path,
            file_hash=repo_hash,
            chunks=[],
            chunk_ids=[],
            ingest_run_id=ingest_run_id,
            source_item_id=source_item_id,
            ingestion_method=ingestion_method,
            insert_chunks=False,
        )
        if pg_synced:
            postgres_synced = True
            logger.info(f"Synced {repo_name} to Postgres")
        elif pg_error:
            logger.warning(f"Postgres artifact sync failed: {pg_error}")

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
    Postgres-first ingestion with best-effort secondary indexing.

    Postgres is the system of record. Vector/graph indexes are updated
    after Postgres writes and can be rebuilt if they fail.
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
        """Ingest PDF with Postgres-first ordering."""
        logger.info(f"Postgres-first ingest: {pdf_path}")
        return super().ingest_pdf(
            pdf_path=pdf_path,
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
