#!/usr/bin/env python3
"""
Backfill passage_concepts using Gemini Batch API.

Features:
- Uses Gemini 2.0 Flash-Lite for cost efficiency
- Populates evidence JSONB field (currently always NULL)
- Idempotent: only processes passages without gemini_batch_v1
- Resumable via kb_migrations checkpoints
- Supports pilot mode (N=200) and full backfill

Usage:
    # Pilot run (200 passages)
    python scripts/backfill_passage_concepts_gemini_batch.py --pilot

    # Full backfill
    python scripts/backfill_passage_concepts_gemini_batch.py --full

    # Status check
    python scripts/backfill_passage_concepts_gemini_batch.py --status

    # Cost estimate
    python scripts/backfill_passage_concepts_gemini_batch.py --estimate
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.gemini_batch import (
    BatchRequest,
    run_sync_batch,
    parse_concept_response,
    build_evidence_jsonb,
    estimate_cost,
    DEFAULT_MODEL
)
from lib.kb_derived import (
    ensure_derived_tables,
    upsert_passage_concept,
    update_migration_checkpoint,
    get_migration_checkpoint
)
from lib.db import get_db_connection

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
EXTRACTOR_MODEL = DEFAULT_MODEL
EXTRACTOR_VERSION = "gemini_batch_v1"
JOB_NAME = f"backfill_passage_concepts_{EXTRACTOR_VERSION}"
BATCH_SIZE = 50  # Passages per API batch (for sync mode)
MAX_TEXT_CHARS = 2000  # Truncate passages longer than this
DEFAULT_DELAY = 0.3  # Seconds between API requests (was 1.0, reduced since no rate limits hit)


def fetch_remaining_passages(
    conn,
    limit: Optional[int] = None,
    after_id: Optional[str] = None,
    worker_id: Optional[int] = None,
    num_workers: Optional[int] = None
) -> List[Tuple[str, str]]:
    """Fetch passages that don't have gemini_batch_v1 concepts.

    Args:
        conn: Database connection
        limit: Max passages to fetch (None = all)
        after_id: Resume after this passage_id
        worker_id: Worker ID for sharding (0 to num_workers-1)
        num_workers: Total number of parallel workers

    Returns:
        List of (passage_id, passage_text) tuples
    """
    cursor = conn.cursor()

    query = """
    SELECT p.passage_id::text, p.passage_text
    FROM passages p
    WHERE NOT EXISTS (
        SELECT 1 FROM passage_concepts pc
        WHERE pc.passage_id = p.passage_id
        AND pc.extractor_version = %s
    )
    """
    params = [EXTRACTOR_VERSION]

    if after_id:
        query += " AND p.passage_id > %s::uuid"
        params.append(after_id)

    # Shard by worker using modulo on passage_id hash
    if worker_id is not None and num_workers is not None and num_workers > 1:
        query += f" AND MOD(('x' || SUBSTRING(p.passage_id::text, 1, 8))::bit(32)::int, {num_workers}) = {worker_id}"

    query += " ORDER BY p.passage_id"

    if limit:
        query += f" LIMIT {int(limit)}"

    cursor.execute(query, params)
    results = cursor.fetchall()
    cursor.close()

    return results


def count_remaining_passages(conn) -> int:
    """Count passages without gemini_batch_v1 concepts."""
    cursor = conn.cursor()
    cursor.execute("""
    SELECT COUNT(*)
    FROM passages p
    WHERE NOT EXISTS (
        SELECT 1 FROM passage_concepts pc
        WHERE pc.passage_id = p.passage_id
        AND pc.extractor_version = %s
    )
    """, (EXTRACTOR_VERSION,))
    count = cursor.fetchone()[0]
    cursor.close()
    return count


def process_batch(
    conn,
    passages: List[Tuple[str, str]],
    dry_run: bool = False,
    delay: float = DEFAULT_DELAY
) -> Dict[str, int]:
    """Process a batch of passages through Gemini API.

    Args:
        conn: Database connection
        passages: List of (passage_id, passage_text) tuples
        dry_run: If True, don't write to database
        delay: Seconds between API requests

    Returns:
        Dict with success_count, fail_count, concepts_inserted
    """
    # Build batch requests
    requests = [
        BatchRequest(custom_id=pid, text=text[:MAX_TEXT_CHARS])
        for pid, text in passages
    ]

    # Run through Gemini API (sync mode with rate limiting)
    results = run_sync_batch(
        requests,
        model=EXTRACTOR_MODEL,
        max_output_tokens=384,  # Reduced to prevent truncation
        delay_between_requests=delay,
        max_retries=5
    )

    stats = {"success": 0, "failed": 0, "concepts": 0}

    for result in results:
        passage_id = result["custom_id"]
        concepts = result.get("concepts", [])
        error = result.get("error")

        if error:
            logger.warning(f"Passage {passage_id} failed: {error}")
            stats["failed"] += 1
            continue

        if not concepts:
            logger.debug(f"Passage {passage_id} yielded no concepts")
            stats["failed"] += 1
            continue

        # Get original text for evidence building
        source_text = next((t for p, t in passages if p == passage_id), "")

        # Insert each concept
        for concept in concepts:
            evidence = build_evidence_jsonb(concept, source_text)

            if not dry_run:
                try:
                    upsert_passage_concept(
                        conn,
                        passage_id=passage_id,
                        concept_name=concept["name"],
                        concept_type=concept["type"],
                        aliases=concept.get("aliases", []),
                        confidence=concept.get("confidence", 0.7),
                        extractor_model=EXTRACTOR_MODEL,
                        extractor_version=EXTRACTOR_VERSION,
                        evidence=evidence
                    )
                    stats["concepts"] += 1
                except Exception as e:
                    logger.error(f"Failed to insert concept {concept['name']} for {passage_id}: {e}")

        stats["success"] += 1

    return stats


def run_pilot(conn, n: int = 200, dry_run: bool = False, delay: float = DEFAULT_DELAY) -> None:
    """Run pilot extraction on N passages.

    Args:
        conn: Database connection
        n: Number of passages to process
        dry_run: If True, don't write to database
        delay: Seconds between API requests
    """
    logger.info(f"Starting pilot run with {n} passages (dry_run={dry_run}, delay={delay}s)")

    # Fetch passages
    passages = fetch_remaining_passages(conn, limit=n)
    logger.info(f"Fetched {len(passages)} passages")

    if not passages:
        logger.info("No remaining passages to process")
        return

    # Process in batches
    total_stats = {"success": 0, "failed": 0, "concepts": 0}

    for i in range(0, len(passages), BATCH_SIZE):
        batch = passages[i:i + BATCH_SIZE]
        logger.info(f"Processing batch {i // BATCH_SIZE + 1}/{(len(passages) + BATCH_SIZE - 1) // BATCH_SIZE}")

        stats = process_batch(conn, batch, dry_run=dry_run, delay=delay)

        total_stats["success"] += stats["success"]
        total_stats["failed"] += stats["failed"]
        total_stats["concepts"] += stats["concepts"]

        # Update checkpoint after each batch
        if not dry_run and batch:
            last_id = batch[-1][0]
            update_migration_checkpoint(
                conn,
                job_name=JOB_NAME + "_pilot",
                cursor_position=last_id,
                status="running",
                items_processed=stats["success"],
                items_failed=stats["failed"],
                cursor_type="passage_id"
            )

    # Mark pilot complete
    if not dry_run:
        update_migration_checkpoint(
            conn,
            job_name=JOB_NAME + "_pilot",
            cursor_position=None,
            status="completed",
            items_processed=0,
            items_failed=0,
            notes=f"Pilot complete: {total_stats['success']} passages, {total_stats['concepts']} concepts"
        )

    logger.info(f"Pilot complete: {total_stats}")

    # Generate QC report
    generate_qc_report(conn, "pilot")


def run_full_backfill(
    conn,
    dry_run: bool = False,
    resume: bool = True,
    worker_id: Optional[int] = None,
    num_workers: int = 1,
    delay: float = DEFAULT_DELAY
) -> None:
    """Run full backfill of all remaining passages.

    Args:
        conn: Database connection
        dry_run: If True, don't write to database
        resume: If True, resume from last checkpoint
        worker_id: Worker ID for sharding (0 to num_workers-1)
        num_workers: Total number of parallel workers
        delay: Seconds between API requests
    """
    worker_suffix = f"_w{worker_id}" if worker_id is not None else ""
    job_name = JOB_NAME + worker_suffix

    # Get checkpoint if resuming
    after_id = None
    if resume:
        checkpoint = get_migration_checkpoint(conn, job_name)
        if checkpoint and checkpoint.get("status") == "running":
            after_id = checkpoint.get("cursor_position")
            logger.info(f"Resuming from checkpoint: {after_id}")

    # Count remaining
    remaining = count_remaining_passages(conn)
    worker_info = f" (worker {worker_id}/{num_workers})" if worker_id is not None else ""
    logger.info(f"Starting full backfill{worker_info}: {remaining} passages remaining, delay={delay}s")

    if remaining == 0:
        logger.info("No remaining passages to process")
        return

    # Initialize job if not resuming
    if not after_id:
        update_migration_checkpoint(
            conn,
            job_name=job_name,
            cursor_position=None,
            status="running",
            items_processed=0,
            items_failed=0,
            cursor_type="passage_id"
        )

    total_stats = {"success": 0, "failed": 0, "concepts": 0}
    batch_num = 0

    while True:
        # Fetch next batch with worker sharding
        passages = fetch_remaining_passages(
            conn,
            limit=BATCH_SIZE,
            after_id=after_id,
            worker_id=worker_id,
            num_workers=num_workers
        )

        if not passages:
            break

        batch_num += 1
        logger.info(f"Processing batch {batch_num} ({len(passages)} passages){worker_info}")

        stats = process_batch(conn, passages, dry_run=dry_run, delay=delay)

        total_stats["success"] += stats["success"]
        total_stats["failed"] += stats["failed"]
        total_stats["concepts"] += stats["concepts"]

        # Update checkpoint
        if not dry_run:
            last_id = passages[-1][0]
            update_migration_checkpoint(
                conn,
                job_name=job_name,
                cursor_position=last_id,
                status="running",
                items_processed=stats["success"],
                items_failed=stats["failed"],
                cursor_type="passage_id"
            )
            after_id = last_id

        # Log progress periodically
        if batch_num % 10 == 0:
            logger.info(f"Progress{worker_info}: {total_stats['success']} success, {total_stats['concepts']} concepts")

    # Mark complete
    if not dry_run:
        update_migration_checkpoint(
            conn,
            job_name=job_name,
            cursor_position=None,
            status="completed",
            items_processed=0,
            items_failed=0,
            notes=f"Complete: {total_stats['success']} passages, {total_stats['concepts']} concepts"
        )

    logger.info(f"Full backfill complete{worker_info}: {total_stats}")

    # Generate QC report (only for single worker or worker 0)
    if worker_id is None or worker_id == 0:
        generate_qc_report(conn, "full")


def generate_qc_report(conn, run_type: str) -> None:
    """Generate QC report for the backfill run.

    Args:
        conn: Database connection
        run_type: "pilot" or "full"
    """
    cursor = conn.cursor()

    # Evidence coverage
    cursor.execute("""
    SELECT
        COUNT(*) as total,
        COUNT(*) FILTER (WHERE evidence IS NOT NULL) as with_evidence,
        COUNT(*) FILTER (WHERE evidence IS NULL) as without_evidence
    FROM passage_concepts
    WHERE extractor_version = %s
    """, (EXTRACTOR_VERSION,))
    evidence_stats = cursor.fetchone()

    # Concepts per passage
    cursor.execute("""
    SELECT
        AVG(cnt) as avg_concepts,
        PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY cnt) as p10,
        PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY cnt) as p90
    FROM (
        SELECT passage_id, COUNT(*) as cnt
        FROM passage_concepts
        WHERE extractor_version = %s
        GROUP BY passage_id
    ) sub
    """, (EXTRACTOR_VERSION,))
    concept_stats = cursor.fetchone()

    # Type distribution
    cursor.execute("""
    SELECT concept_type, COUNT(*) as cnt
    FROM passage_concepts
    WHERE extractor_version = %s
    GROUP BY concept_type
    ORDER BY cnt DESC
    """, (EXTRACTOR_VERSION,))
    type_dist = cursor.fetchall()

    # Top concepts
    cursor.execute("""
    SELECT concept_name, COUNT(*) as cnt
    FROM passage_concepts
    WHERE extractor_version = %s
    GROUP BY concept_name
    ORDER BY cnt DESC
    LIMIT 20
    """, (EXTRACTOR_VERSION,))
    top_concepts = cursor.fetchall()

    # Potential garbage (very long names)
    cursor.execute("""
    SELECT concept_name, LENGTH(concept_name) as len
    FROM passage_concepts
    WHERE extractor_version = %s AND LENGTH(concept_name) > 60
    ORDER BY len DESC
    LIMIT 10
    """, (EXTRACTOR_VERSION,))
    long_names = cursor.fetchall()

    cursor.close()

    # Write report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path(__file__).parent.parent / "docs" / "runlogs" / f"gemini_batch_qc_{timestamp}.md"

    total = evidence_stats[0] if evidence_stats else 0
    with_ev = evidence_stats[1] if evidence_stats else 0
    ev_pct = (with_ev / total * 100) if total > 0 else 0

    report = f"""# Gemini Batch QC Report - {run_type.upper()}
**Generated**: {datetime.now().isoformat()}
**Extractor**: {EXTRACTOR_MODEL} ({EXTRACTOR_VERSION})

## Evidence Coverage
| Metric | Value |
|--------|-------|
| Total concepts | {total:,} |
| With evidence | {with_ev:,} ({ev_pct:.1f}%) |
| Without evidence | {evidence_stats[2] if evidence_stats else 0:,} |

## Concepts Per Passage
| Metric | Value |
|--------|-------|
| Average | {f"{concept_stats[0]:.2f}" if concept_stats and concept_stats[0] is not None else 'N/A'} |
| P10 | {f"{concept_stats[1]:.1f}" if concept_stats and concept_stats[1] is not None else 'N/A'} |
| P90 | {f"{concept_stats[2]:.1f}" if concept_stats and concept_stats[2] is not None else 'N/A'} |

## Type Distribution
| Type | Count |
|------|-------|
"""
    for ctype, cnt in type_dist:
        report += f"| {ctype} | {cnt:,} |\n"

    report += """
## Top 20 Concepts
| Concept | Count |
|---------|-------|
"""
    for name, cnt in top_concepts:
        report += f"| {name} | {cnt:,} |\n"

    if long_names:
        report += """
## Potential Garbage (name > 60 chars)
| Concept | Length |
|---------|--------|
"""
        for name, length in long_names:
            report += f"| {name[:50]}... | {length} |\n"

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    logger.info(f"QC report written to {report_path}")


def show_status(conn) -> None:
    """Show current backfill status."""
    cursor = conn.cursor()

    # Migration status
    cursor.execute("""
    SELECT job_name, status, items_processed, items_failed, updated_at
    FROM kb_migrations
    WHERE job_name LIKE %s
    ORDER BY updated_at DESC
    """, (f"%{EXTRACTOR_VERSION}%",))
    migrations = cursor.fetchall()

    # Counts
    cursor.execute("SELECT COUNT(*) FROM passages")
    total_passages = cursor.fetchone()[0]

    cursor.execute("""
    SELECT COUNT(DISTINCT passage_id) FROM passage_concepts WHERE extractor_version = %s
    """, (EXTRACTOR_VERSION,))
    with_concepts = cursor.fetchone()[0]

    remaining = total_passages - with_concepts

    cursor.close()

    print(f"\n=== Gemini Batch Backfill Status ===")
    print(f"Total passages: {total_passages:,}")
    print(f"With {EXTRACTOR_VERSION}: {with_concepts:,}")
    print(f"Remaining: {remaining:,}")
    print(f"Progress: {with_concepts / total_passages * 100:.2f}%")

    print(f"\n=== Migration Jobs ===")
    for job in migrations:
        print(f"  {job[0]}: {job[1]} ({job[2]:,} processed, {job[3]:,} failed)")


def show_cost_estimate(conn) -> None:
    """Show cost estimate for remaining passages."""
    remaining = count_remaining_passages(conn)

    # Sample passages to estimate avg length
    cursor = conn.cursor()
    cursor.execute("""
    SELECT AVG(LENGTH(passage_text)) FROM (
        SELECT passage_text FROM passages
        ORDER BY RANDOM() LIMIT 500
    ) sub
    """)
    avg_len = cursor.fetchone()[0] or 1500
    cursor.close()

    cost = estimate_cost(remaining, avg_input_chars=int(avg_len))

    print(f"\n=== Cost Estimate for {remaining:,} Passages ===")
    print(f"Model: {cost['model']}")
    print(f"Avg passage length: {int(avg_len):,} chars")
    print(f"Input tokens: {cost['input_tokens']:,} (${cost['input_cost_usd']:.2f})")
    print(f"Output tokens: {cost['output_tokens']:,} (${cost['output_cost_usd']:.2f})")
    print(f"TOTAL ESTIMATED COST: ${cost['total_cost_usd']:.2f}")

    # Write to file
    report_path = Path(__file__).parent.parent / "docs" / "runlogs" / "gemini_batch_cost_estimate.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(f"""# Gemini Batch Cost Estimate
**Generated**: {datetime.now().isoformat()}

## Parameters
- Remaining passages: {remaining:,}
- Avg passage length: {int(avg_len):,} chars
- Model: {cost['model']}

## Token Estimates
- Input: {cost['input_tokens']:,} tokens
- Output: {cost['output_tokens']:,} tokens

## Cost Breakdown
- Input cost: ${cost['input_cost_usd']:.4f}
- Output cost: ${cost['output_cost_usd']:.4f}
- **Total: ${cost['total_cost_usd']:.2f}**

## Notes
- Batch API pricing (50% discount vs real-time)
- Actual cost may vary based on output length
- Estimate assumes ~60% output token utilization
""")
    print(f"\nEstimate saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Backfill passage concepts with Gemini Batch API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run pilot (200 passages)
  python %(prog)s --pilot

  # Run full backfill with 4 parallel workers
  python %(prog)s --full --worker-id 0 --num-workers 4 &
  python %(prog)s --full --worker-id 1 --num-workers 4 &
  python %(prog)s --full --worker-id 2 --num-workers 4 &
  python %(prog)s --full --worker-id 3 --num-workers 4 &

  # Check status
  python %(prog)s --status
"""
    )
    parser.add_argument("--pilot", action="store_true", help="Run pilot (200 passages)")
    parser.add_argument("--pilot-n", type=int, default=200, help="Number of passages for pilot")
    parser.add_argument("--full", action="store_true", help="Run full backfill")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--estimate", action="store_true", help="Show cost estimate")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to database")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh (don't resume)")

    # Parallel worker options
    parser.add_argument("--worker-id", type=int, default=None,
                        help="Worker ID for parallel processing (0 to num_workers-1)")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Total number of parallel workers (default: 1)")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY,
                        help=f"Delay between API requests in seconds (default: {DEFAULT_DELAY})")

    args = parser.parse_args()

    # Validate worker args
    if args.worker_id is not None:
        if args.num_workers <= 1:
            print("ERROR: --worker-id requires --num-workers > 1")
            sys.exit(1)
        if args.worker_id < 0 or args.worker_id >= args.num_workers:
            print(f"ERROR: --worker-id must be 0 to {args.num_workers - 1}")
            sys.exit(1)

    # Ensure GEMINI_API_KEY is set for API operations
    if (args.pilot or args.full) and not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY environment variable not set")
        print("Get your API key from https://aistudio.google.com/apikey")
        sys.exit(1)

    # Connect to database
    conn = get_db_connection()

    try:
        # Ensure tables exist
        ensure_derived_tables(conn)

        if args.status:
            show_status(conn)
        elif args.estimate:
            show_cost_estimate(conn)
        elif args.pilot:
            run_pilot(conn, n=args.pilot_n, dry_run=args.dry_run, delay=args.delay)
        elif args.full:
            run_full_backfill(
                conn,
                dry_run=args.dry_run,
                resume=not args.no_resume,
                worker_id=args.worker_id,
                num_workers=args.num_workers,
                delay=args.delay
            )
        else:
            parser.print_help()

    finally:
        conn.close()


if __name__ == "__main__":
    main()
