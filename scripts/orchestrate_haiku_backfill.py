#!/usr/bin/env python3
"""
Orchestrate Haiku passage backfill - AUTONOMOUS VERSION

This script runs autonomously and calls Anthropic API directly for concept extraction.
Use this for long-running backfill operations.

For interactive use from Claude Code, use the helper functions directly
and spawn Task tool subagents.
"""

import sys
import os
import time
import anthropic
from pathlib import Path
from typing import List, Optional

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.backfill_passage_concepts_haiku import (
    fetch_passages_batch,
    build_extraction_prompt,
    parse_haiku_response,
    save_concepts_for_passage,
    checkpoint,
    get_checkpoint,
    get_stats
)
from lib.db import get_db_connection

# Initialize Anthropic client
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

BATCH_SIZE = 10
CHECKPOINT_INTERVAL = 100  # Save checkpoint every N passages
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


def extract_concepts_with_api(prompt: str) -> List[str]:
    """
    Extract concepts using Anthropic API directly.

    Args:
        prompt: Extraction prompt

    Returns:
        List of extracted concept names
    """
    try:
        message = client.messages.create(
            model="claude-haiku-20250103",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        # Extract text from response
        response_text = message.content[0].text

        # Parse concepts
        concepts = parse_haiku_response(response_text)
        return concepts

    except anthropic.RateLimitError as e:
        print(f"Rate limit hit: {e}")
        raise
    except Exception as e:
        print(f"API error: {e}")
        raise


def process_batch(conn, batch, stats):
    """Process a batch of passages."""
    batch_start = time.time()
    processed = 0
    failed = 0
    last_passage_id = None

    for passage_id, text, title, year in batch:
        last_passage_id = passage_id

        # Build prompt
        prompt = build_extraction_prompt(text, title, year)

        # Extract with retries
        concepts = None
        for attempt in range(MAX_RETRIES):
            try:
                concepts = extract_concepts_with_api(prompt)
                break
            except anthropic.RateLimitError:
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY * (attempt + 1)
                    print(f"Rate limit - waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"Failed after {MAX_RETRIES} retries")
                    failed += 1
                    continue
            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(1)
                else:
                    failed += 1
                    continue

        # Save concepts if extracted
        if concepts:
            saved = save_concepts_for_passage(conn, passage_id, concepts)
            print(f"✓ {passage_id[:8]}... - {saved} concepts - {title[:40]}")
            processed += 1
        else:
            print(f"✗ {passage_id[:8]}... - Failed")
            failed += 1

        # Checkpoint periodically
        if (stats['with_concepts'] + processed) % CHECKPOINT_INTERVAL == 0:
            checkpoint(
                conn,
                last_passage_id,
                processed=stats['with_concepts'] + processed,
                failed=stats['checkpoint']['items_failed'] + failed if stats['checkpoint'] else failed,
                status="running"
            )
            conn.commit()
            print(f"Checkpoint saved at {stats['with_concepts'] + processed}")

    # Final checkpoint for batch
    if last_passage_id:
        checkpoint(
            conn,
            last_passage_id,
            processed=stats['with_concepts'] + processed,
            failed=stats['checkpoint']['items_failed'] + failed if stats['checkpoint'] else failed,
            status="running"
        )
        conn.commit()

    batch_time = time.time() - batch_start
    rate = processed / batch_time if batch_time > 0 else 0
    print(f"\nBatch: {processed} processed, {failed} failed in {batch_time:.1f}s ({rate:.1f}/sec)")

    return processed, failed


def run_backfill(max_passages: Optional[int] = None):
    """
    Run autonomous backfill.

    Args:
        max_passages: Max passages to process (None = all)
    """
    conn = get_db_connection()

    # Initial stats
    stats = get_stats(conn)
    print("=== Starting Haiku Backfill ===")
    print(f"Total eligible: {stats['total_eligible']:,}")
    print(f"With concepts:  {stats['with_concepts']:,}")
    print(f"Remaining:      {stats['remaining']:,}")
    print(f"Progress:       {stats['percent_complete']}%\n")

    if stats['checkpoint']:
        print(f"Resuming from: {stats['checkpoint']['cursor_position']}")

    total_processed = 0
    total_failed = 0
    start_time = time.time()

    try:
        while True:
            # Check if done
            if max_passages and total_processed >= max_passages:
                print(f"\nReached max passages ({max_passages})")
                break

            # Fetch batch
            cursor = get_checkpoint(conn)
            batch = fetch_passages_batch(conn, start_after_id=cursor, batch_size=BATCH_SIZE)

            if not batch:
                print("\n✓ All passages processed!")
                checkpoint(
                    conn,
                    cursor,
                    processed=stats['with_concepts'] + total_processed,
                    failed=total_failed,
                    status="completed"
                )
                conn.commit()
                break

            # Process batch
            processed, failed = process_batch(conn, batch, stats)
            total_processed += processed
            total_failed += failed

            # Progress update
            elapsed = time.time() - start_time
            rate = total_processed / elapsed if elapsed > 0 else 0
            remaining = stats['remaining'] - total_processed
            eta_hours = (remaining / rate / 3600) if rate > 0 else 0

            print(f"\nSession: {total_processed} processed, {total_failed} failed")
            print(f"Rate: {rate:.1f} passages/sec")
            print(f"Remaining: {remaining:,} (~{eta_hours:.1f}h at current rate)\n")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        checkpoint(
            conn,
            get_checkpoint(conn),
            processed=stats['with_concepts'] + total_processed,
            failed=total_failed,
            status="paused"
        )
        conn.commit()

    finally:
        conn.close()

        # Final stats
        elapsed = time.time() - start_time
        print(f"\n=== Session Complete ===")
        print(f"Processed: {total_processed}")
        print(f"Failed: {total_failed}")
        print(f"Time: {elapsed/60:.1f} min")
        print(f"Rate: {total_processed/(elapsed/60):.1f} passages/min")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Orchestrate Haiku passage backfill")
    parser.add_argument("--max", type=int, help="Max passages to process")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size")

    args = parser.parse_args()

    if args.batch_size:
        BATCH_SIZE = args.batch_size

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        print("Set it with: export ANTHROPIC_API_KEY='your-key-here'")
        sys.exit(1)

    run_backfill(max_passages=args.max)
