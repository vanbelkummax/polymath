#!/usr/bin/env python3
"""
Knowledge Base Derived Tables

Manages derived tables for:
- Chunk-level concept extraction results
- Migration job tracking and checkpointing

These tables are derived from the core artifacts/chunks/passages tables
and can be rebuilt without data loss.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


def ensure_derived_tables(conn) -> None:
    """Create derived tables if they don't exist.

    These tables store processed/derived data that can be regenerated
    from the source artifacts/chunks/passages tables.

    Args:
        conn: psycopg2 connection object
    """
    cursor = conn.cursor()

    try:
        # Table 1: chunk_concepts
        # Stores LLM-extracted concepts at the chunk level
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunk_concepts (
            chunk_id TEXT NOT NULL,
            concept_name TEXT NOT NULL,
            concept_type TEXT NULL,
            aliases JSONB NULL,
            confidence REAL NULL,
            extractor_model TEXT NOT NULL,
            extractor_version TEXT NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (chunk_id, concept_name, extractor_version)
        )
        """)

        # Index for fast lookups by concept
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunk_concepts_name
        ON chunk_concepts(concept_name)
        """)

        # Index for filtering by version
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunk_concepts_version
        ON chunk_concepts(extractor_version)
        """)

        # Index for confidence-based queries
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunk_concepts_confidence
        ON chunk_concepts(confidence)
        WHERE confidence >= 0.7
        """)

        # Table 2: passage_concepts
        # Same structure for passage-based chunks (papers)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS passage_concepts (
            passage_id UUID NOT NULL,
            concept_name TEXT NOT NULL,
            concept_type TEXT NULL,
            aliases JSONB NULL,
            confidence REAL NULL,
            extractor_model TEXT NOT NULL,
            extractor_version TEXT NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (passage_id, concept_name, extractor_version)
        )
        """)

        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_passage_concepts_name
        ON passage_concepts(concept_name)
        """)

        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_passage_concepts_version
        ON passage_concepts(extractor_version)
        """)

        # Table 3: kb_migrations
        # Tracks migration job progress for resumability
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS kb_migrations (
            job_name TEXT PRIMARY KEY,
            cursor_position TEXT NULL,
            cursor_type TEXT NULL,
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            status TEXT NOT NULL,
            notes TEXT NULL,
            items_processed INTEGER DEFAULT 0,
            items_failed INTEGER DEFAULT 0,
            started_at TIMESTAMPTZ DEFAULT NOW(),
            finished_at TIMESTAMPTZ NULL
        )
        """)

        conn.commit()
        logger.info("Derived tables ensured successfully")

    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to create derived tables: {e}")
        raise
    finally:
        cursor.close()


def upsert_chunk_concept(
    conn,
    chunk_id: str,
    concept_name: str,
    concept_type: str,
    aliases: List[str],
    confidence: float,
    extractor_model: str,
    extractor_version: str
) -> None:
    """Insert or update a chunk concept.

    Args:
        conn: Database connection
        chunk_id: Chunk identifier
        concept_name: Normalized concept name (snake_case)
        concept_type: Concept type (method, domain, etc.)
        aliases: List of alternative names
        confidence: Extraction confidence 0.0-1.0
        extractor_model: Model name used
        extractor_version: Extractor version tag
    """
    import json

    cursor = conn.cursor()
    try:
        cursor.execute("""
        INSERT INTO chunk_concepts (
            chunk_id, concept_name, concept_type, aliases, confidence,
            extractor_model, extractor_version
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (chunk_id, concept_name, extractor_version)
        DO UPDATE SET
            concept_type = EXCLUDED.concept_type,
            aliases = EXCLUDED.aliases,
            confidence = EXCLUDED.confidence,
            extractor_model = EXCLUDED.extractor_model,
            created_at = NOW()
        """, (
            chunk_id, concept_name, concept_type, json.dumps(aliases),
            confidence, extractor_model, extractor_version
        ))
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to upsert chunk concept: {e}")
        raise
    finally:
        cursor.close()


def upsert_passage_concept(
    conn,
    passage_id: str,
    concept_name: str,
    concept_type: str,
    aliases: List[str],
    confidence: float,
    extractor_model: str,
    extractor_version: str,
    evidence: Optional[Dict[str, Any]] = None
) -> None:
    """Insert or update a passage concept with optional evidence.

    Args:
        conn: Database connection
        passage_id: Passage UUID
        concept_name: Normalized concept name (canonical, snake_case)
        concept_type: Concept type (method, domain, etc.)
        aliases: List of alternative names
        confidence: Extraction confidence 0.0-1.0
        extractor_model: Model name used
        extractor_version: Extractor version tag
        evidence: Optional evidence dict with standardized contract:
            {
                "surface": str|null,        # Exact substring from passage (or null)
                "context": str|null,        # Context containing surface (or null)
                "support": str,             # "literal"|"normalized"|"inferred"|"none"
                "quality": dict,            # Text quality metrics from text_quality()
                "source_text": str          # "raw"|"soft_normalized"
            }

            Support types:
            - "literal": surface+context are exact substrings of raw passage text
            - "normalized": matches only after normalize_for_match() transform
            - "inferred": LLM high-confidence extraction without text evidence
            - "none": concept extracted but no verifiable text evidence

            CRITICAL: Only "literal" support is audit-grade citable.
    """
    import json

    cursor = conn.cursor()
    try:
        cursor.execute("""
        INSERT INTO passage_concepts (
            passage_id, concept_name, concept_type, aliases, confidence,
            extractor_model, extractor_version, evidence
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (passage_id, concept_name, extractor_version)
        DO UPDATE SET
            concept_type = EXCLUDED.concept_type,
            aliases = EXCLUDED.aliases,
            confidence = EXCLUDED.confidence,
            extractor_model = EXCLUDED.extractor_model,
            evidence = EXCLUDED.evidence,
            created_at = NOW()
        """, (
            passage_id, concept_name, concept_type, json.dumps(aliases),
            confidence, extractor_model, extractor_version,
            json.dumps(evidence) if evidence else None
        ))
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to upsert passage concept: {e}")
        raise
    finally:
        cursor.close()


def update_migration_checkpoint(
    conn,
    job_name: str,
    cursor_position: Optional[str],
    status: str,
    items_processed: int = 0,
    items_failed: int = 0,
    notes: Optional[str] = None,
    cursor_type: str = "chunk_id"
) -> None:
    """Update migration job checkpoint.

    Args:
        conn: Database connection
        job_name: Unique job identifier
        cursor_position: Last processed item ID
        status: Job status (running, completed, failed)
        items_processed: Total items processed
        items_failed: Total items failed
        notes: Optional notes
        cursor_type: Type of cursor (chunk_id, passage_id, artifact_id)
    """
    cursor = conn.cursor()
    try:
        cursor.execute("""
        INSERT INTO kb_migrations (
            job_name, cursor_position, cursor_type, status, notes,
            items_processed, items_failed, updated_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
        ON CONFLICT (job_name)
        DO UPDATE SET
            cursor_position = EXCLUDED.cursor_position,
            cursor_type = EXCLUDED.cursor_type,
            status = EXCLUDED.status,
            notes = EXCLUDED.notes,
            items_processed = kb_migrations.items_processed + EXCLUDED.items_processed,
            items_failed = kb_migrations.items_failed + EXCLUDED.items_failed,
            updated_at = NOW(),
            finished_at = CASE
                WHEN EXCLUDED.status IN ('completed', 'failed') THEN NOW()
                ELSE NULL
            END
        """, (job_name, cursor_position, cursor_type, status, notes, items_processed, items_failed))
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to update migration checkpoint: {e}")
        raise
    finally:
        cursor.close()


def get_migration_checkpoint(conn, job_name: str) -> Optional[Dict[str, Any]]:
    """Get migration job checkpoint.

    Args:
        conn: Database connection
        job_name: Job identifier

    Returns:
        Dict with job metadata, or None if not found
    """
    cursor = conn.cursor()
    try:
        cursor.execute("""
        SELECT job_name, cursor_position, cursor_type, status, notes,
               items_processed, items_failed, started_at, updated_at, finished_at
        FROM kb_migrations
        WHERE job_name = %s
        """, (job_name,))

        row = cursor.fetchone()
        if not row:
            return None

        return {
            "job_name": row[0],
            "cursor_position": row[1],
            "cursor_type": row[2],
            "status": row[3],
            "notes": row[4],
            "items_processed": row[5],
            "items_failed": row[6],
            "started_at": row[7],
            "updated_at": row[8],
            "finished_at": row[9]
        }
    finally:
        cursor.close()


def get_chunk_concepts(
    conn,
    chunk_id: str,
    extractor_version: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Get all concepts for a chunk.

    Args:
        conn: Database connection
        chunk_id: Chunk identifier
        extractor_version: Optional version filter

    Returns:
        List of concept dicts
    """
    cursor = conn.cursor()
    try:
        if extractor_version:
            cursor.execute("""
            SELECT concept_name, concept_type, aliases, confidence,
                   extractor_model, extractor_version, created_at
            FROM chunk_concepts
            WHERE chunk_id = %s AND extractor_version = %s
            ORDER BY confidence DESC
            """, (chunk_id, extractor_version))
        else:
            cursor.execute("""
            SELECT concept_name, concept_type, aliases, confidence,
                   extractor_model, extractor_version, created_at
            FROM chunk_concepts
            WHERE chunk_id = %s
            ORDER BY confidence DESC, created_at DESC
            """, (chunk_id,))

        rows = cursor.fetchall()
        return [
            {
                "name": row[0],
                "type": row[1],
                "aliases": row[2],
                "confidence": row[3],
                "extractor_model": row[4],
                "extractor_version": row[5],
                "created_at": row[6]
            }
            for row in rows
        ]
    finally:
        cursor.close()


def get_passage_concepts(
    conn,
    passage_id: str,
    extractor_version: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Get all concepts for a passage.

    Args:
        conn: Database connection
        passage_id: Passage UUID
        extractor_version: Optional version filter

    Returns:
        List of concept dicts
    """
    cursor = conn.cursor()
    try:
        if extractor_version:
            cursor.execute("""
            SELECT concept_name, concept_type, aliases, confidence,
                   extractor_model, extractor_version, created_at
            FROM passage_concepts
            WHERE passage_id = %s AND extractor_version = %s
            ORDER BY confidence DESC
            """, (passage_id, extractor_version))
        else:
            cursor.execute("""
            SELECT concept_name, concept_type, aliases, confidence,
                   extractor_model, extractor_version, created_at
            FROM passage_concepts
            WHERE passage_id = %s
            ORDER BY confidence DESC, created_at DESC
            """, (passage_id,))

        rows = cursor.fetchall()
        return [
            {
                "name": row[0],
                "type": row[1],
                "aliases": row[2],
                "confidence": row[3],
                "extractor_model": row[4],
                "extractor_version": row[5],
                "created_at": row[6]
            }
            for row in rows
        ]
    finally:
        cursor.close()


if __name__ == "__main__":
    # Test derived tables creation
    import psycopg2

    conn = psycopg2.connect("dbname=polymath user=polymath host=/var/run/postgresql")
    try:
        ensure_derived_tables(conn)
        print("✓ Derived tables created successfully")

        # Test checkpoint
        update_migration_checkpoint(
            conn, "test_job", "chunk_123", "running",
            items_processed=100, notes="Test migration"
        )
        checkpoint = get_migration_checkpoint(conn, "test_job")
        print(f"✓ Checkpoint: {checkpoint}")

    finally:
        conn.close()
