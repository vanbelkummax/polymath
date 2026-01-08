#!/usr/bin/env python3
"""
Backfill chunk_concepts table with LLM-extracted concepts.

Iterates through all chunks/passages and extracts typed concepts using
local Ollama models, storing results in derived tables.
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.kb_derived import (
    ensure_derived_tables,
    upsert_chunk_concept,
    upsert_passage_concept,
    update_migration_checkpoint,
    get_migration_checkpoint
)
from lib.local_extractor import LocalEntityExtractor
from lib.db import get_db_connection

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def backfill_chunks(
    conn,
    extractor: LocalEntityExtractor,
    extractor_version: str,
    batch_size: int = 16,
    limit: Optional[int] = None,
    resume: bool = True,
    dry_run: bool = False
):
    """Backfill concepts for code chunks.

    Args:
        conn: Database connection
        extractor: LocalEntityExtractor instance
        extractor_version: Version tag
        batch_size: Chunks to process before checkpoint
        limit: Max chunks to process (None = all)
        resume: Resume from last checkpoint
        dry_run: Don't write to database
    """
    job_name = f"backfill_chunk_concepts_{extractor_version}"

    # Get checkpoint if resuming
    start_after = None
    if resume:
        checkpoint = get_migration_checkpoint(conn, job_name)
        if checkpoint and checkpoint["status"] != "completed":
            start_after = checkpoint["cursor_position"]
            logger.info(f"Resuming from chunk_id={start_after}")

    # Query chunks
    cursor = conn.cursor()
    if start_after:
        query = """
        SELECT id, content FROM chunks
        WHERE id > %s AND length(content) >= 50
        ORDER BY id
        """
        params = (start_after,)
    else:
        query = """
        SELECT id, content FROM chunks
        WHERE length(content) >= 50
        ORDER BY id
        """
        params = None

    if limit:
        query += f" LIMIT {limit}"

    cursor.execute(query, params)

    processed = 0
    failed = 0
    batch = []

    for row in cursor:
        chunk_id, content = row
        batch.append((chunk_id, content))

        if len(batch) >= batch_size:
            # Process batch
            for cid, text in batch:
                try:
                    concepts = extractor.extract_concepts(text)

                    if not dry_run:
                        for concept in concepts:
                            upsert_chunk_concept(
                                conn, cid,
                                concept["name"],
                                concept["type"],
                                concept["aliases"],
                                concept["confidence"],
                                extractor.fast_model,
                                extractor_version
                            )

                    processed += 1
                    if processed % 100 == 0:
                        logger.info(f"Processed {processed} chunks, {len(concepts)} concepts from last chunk")

                except Exception as e:
                    logger.error(f"Failed to process chunk {cid}: {e}")
                    failed += 1

            # Checkpoint
            if not dry_run:
                update_migration_checkpoint(
                    conn, job_name, batch[-1][0], "running",
                    items_processed=len(batch), items_failed=0,
                    cursor_type="chunk_id"
                )

            batch = []

    # Process remaining
    for cid, text in batch:
        try:
            concepts = extractor.extract_concepts(text)

            if not dry_run:
                for concept in concepts:
                    upsert_chunk_concept(
                        conn, cid,
                        concept["name"],
                        concept["type"],
                        concept["aliases"],
                        concept["confidence"],
                        extractor.fast_model,
                        extractor_version
                    )

            processed += 1

        except Exception as e:
            logger.error(f"Failed to process chunk {cid}: {e}")
            failed += 1

    # Final checkpoint
    if not dry_run:
        update_migration_checkpoint(
            conn, job_name, None, "completed",
            items_processed=0, items_failed=failed,
            notes=f"Completed: {processed} processed, {failed} failed"
        )

    cursor.close()
    logger.info(f"Backfill complete: {processed} processed, {failed} failed")


def backfill_passages(
    conn,
    extractor: LocalEntityExtractor,
    extractor_version: str,
    batch_size: int = 16,
    limit: Optional[int] = None,
    resume: bool = True,
    dry_run: bool = False
):
    """Backfill concepts for paper passages.

    Args:
        conn: Database connection
        extractor: LocalEntityExtractor instance
        extractor_version: Version tag
        batch_size: Passages to process before checkpoint
        limit: Max passages to process (None = all)
        resume: Resume from last checkpoint
        dry_run: Don't write to database
    """
    job_name = f"backfill_passage_concepts_{extractor_version}"

    # Get checkpoint if resuming
    start_after = None
    if resume:
        checkpoint = get_migration_checkpoint(conn, job_name)
        if checkpoint and checkpoint["status"] != "completed":
            start_after = checkpoint["cursor_position"]
            logger.info(f"Resuming from passage_id={start_after}")

    # Query passages
    cursor = conn.cursor()
    if start_after:
        query = """
        SELECT passage_id::text, passage_text FROM passages
        WHERE passage_id::text > %s AND length(passage_text) >= 50
        ORDER BY passage_id
        """
        params = (start_after,)
    else:
        query = """
        SELECT passage_id::text, passage_text FROM passages
        WHERE length(passage_text) >= 50
        ORDER BY passage_id
        """
        params = None

    if limit:
        query += f" LIMIT {limit}"

    cursor.execute(query, params)

    processed = 0
    failed = 0
    batch = []

    for row in cursor:
        passage_id, text = row
        batch.append((passage_id, text))

        if len(batch) >= batch_size:
            # Process batch
            for pid, text in batch:
                try:
                    # Quality-gated evidence extraction with support typing
                    # Returns concepts with embedded evidence dict:
                    # {canonical, type, aliases, confidence, evidence: {...}}
                    concepts = extractor.extract_concepts_with_evidence(text)

                    if not dry_run:
                        for concept in concepts:
                            # Evidence dict now includes standardized contract:
                            # {surface, context, support, quality, source_text}
                            # Support types: "literal"|"normalized"|"inferred"|"none"
                            evidence = concept.get("evidence")

                            upsert_passage_concept(
                                conn, pid,
                                concept["canonical"],
                                concept["type"],
                                concept["aliases"],
                                concept["confidence"],
                                extractor.fast_model,
                                extractor_version,
                                evidence=evidence  # Pass standardized evidence dict
                            )

                    processed += 1
                    if processed % 100 == 0:
                        logger.info(f"Processed {processed} passages")

                except Exception as e:
                    logger.error(f"Failed to process passage {pid}: {e}")
                    failed += 1

            # Checkpoint
            if not dry_run:
                update_migration_checkpoint(
                    conn, job_name, batch[-1][0], "running",
                    items_processed=len(batch), items_failed=0,
                    cursor_type="passage_id"
                )

            batch = []

    # Process remaining
    for pid, text in batch:
        try:
            # Evidence-bound extraction for passages (audit-grade citations)
            concepts = extractor.extract_concepts_with_evidence(text)

            if not dry_run:
                for concept in concepts:
                    # Build evidence dict from surface form + snippet
                    evidence = {
                        "surface": concept.get("surface"),
                        "snippet": concept.get("snippet"),
                        "support": concept.get("support", "literal")
                    }

                    upsert_passage_concept(
                        conn, pid,
                        concept["canonical"],  # Use canonical as concept_name
                        concept["type"],
                        concept["aliases"],
                        concept["confidence"],
                        extractor.fast_model,
                        extractor_version,
                        evidence=evidence  # Add evidence binding
                    )

            processed += 1

        except Exception as e:
            logger.error(f"Failed to process passage {pid}: {e}")
            failed += 1

    # Final checkpoint
    if not dry_run:
        update_migration_checkpoint(
            conn, job_name, None, "completed",
            items_processed=0, items_failed=failed,
            notes=f"Completed: {processed} processed, {failed} failed"
        )

    cursor.close()
    logger.info(f"Backfill complete: {processed} processed, {failed} failed")


def main():
    parser = argparse.ArgumentParser(description="Backfill chunk concepts with LLM extraction")
    parser.add_argument("--pg-dsn", default="dbname=polymath user=polymath host=/var/run/postgresql")
    parser.add_argument("--model-fast", help="Fast LLM model")
    parser.add_argument("--model-heavy", help="Heavy LLM model")
    parser.add_argument("--extractor-version", default="llm_v2")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, help="Max items to process")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--target", choices=["chunks", "passages", "both"], default="both")

    args = parser.parse_args()

    # Connect to database
    import psycopg2
    conn = psycopg2.connect(args.pg_dsn)

    try:
        # Ensure derived tables exist
        ensure_derived_tables(conn)

        # Create extractor
        extractor = LocalEntityExtractor(
            fast_model=args.model_fast,
            heavy_model=args.model_heavy,
            extractor_version=args.extractor_version
        )

        logger.info(f"Starting backfill with {extractor.fast_model}")

        if args.target in ["chunks", "both"]:
            logger.info("Backfilling code chunks...")
            backfill_chunks(
                conn, extractor, args.extractor_version,
                batch_size=args.batch_size,
                limit=args.limit,
                resume=args.resume,
                dry_run=args.dry_run
            )

        if args.target in ["passages", "both"]:
            logger.info("Backfilling paper passages...")
            backfill_passages(
                conn, extractor, args.extractor_version,
                batch_size=args.batch_size,
                limit=args.limit,
                resume=args.resume,
                dry_run=args.dry_run
            )

        logger.info("Backfill complete!")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
