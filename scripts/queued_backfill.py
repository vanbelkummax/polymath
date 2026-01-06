#!/usr/bin/env python3
"""
Queued Backfill - Waits for hybrid ingestion to complete, then runs backfill.
Monitors the hybrid_ingest process and starts backfill when it's done.
"""

import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

HYBRID_PID_FILE = "/home/user/work/polymax/hybrid_ingest.pid"
POLYMAX_DIR = "/home/user/work/polymax"


def is_process_running(pid: int) -> bool:
    """Check if a process is still running."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def wait_for_hybrid_completion():
    """Wait for hybrid ingestion to complete."""
    print(f"[{datetime.now():%H:%M}] Checking for hybrid ingestion process...")

    if not os.path.exists(HYBRID_PID_FILE):
        print("  No hybrid_ingest.pid found - proceeding immediately")
        return

    try:
        with open(HYBRID_PID_FILE) as f:
            pid = int(f.read().strip())
    except:
        print("  Could not read PID file - proceeding")
        return

    if not is_process_running(pid):
        print(f"  Hybrid ingestion (PID {pid}) already completed")
        return

    print(f"  Hybrid ingestion running (PID {pid}) - waiting for completion...")
    print("  Checking every 60 seconds...")

    check_count = 0
    while is_process_running(pid):
        check_count += 1
        if check_count % 5 == 0:  # Every 5 minutes
            print(f"  [{datetime.now():%H:%M}] Still waiting... (PID {pid} active)")
        time.sleep(60)

    print(f"  [{datetime.now():%H:%M}] Hybrid ingestion completed!")
    time.sleep(10)  # Brief pause to ensure clean DB state


def run_script(script_name: str, description: str) -> bool:
    """Run a Python script from the scripts directory."""
    print(f"\n[{datetime.now():%H:%M}] {description}...")

    script_path = Path(POLYMAX_DIR) / "scripts" / script_name

    if not script_path.exists():
        print(f"  ERROR: Script not found at {script_path}")
        return False

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=POLYMAX_DIR,
        capture_output=False
    )

    return result.returncode == 0


def run_backfill():
    """Run the metadata backfill script."""
    return run_script("backfill_metadata.py", "Starting metadata backfill")


def run_concept_linking():
    """Run the concept linking fix script."""
    return run_script("fix_concept_links.py", "Starting concept linking fix")


def main():
    print("=" * 60)
    print("QUEUED MAINTENANCE TASKS")
    print("=" * 60)
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M}")
    print()
    print("Tasks queued:")
    print("  1. Metadata backfill (~84K chunks)")
    print("  2. Concept linking fix (13+ links)")
    print()

    # Wait for hybrid ingestion
    wait_for_hybrid_completion()

    # Run backfill
    backfill_ok = run_backfill()

    # Run concept linking (regardless of backfill result)
    concept_ok = run_concept_linking()

    print()
    print("=" * 60)
    print("QUEUED TASKS SUMMARY")
    print("=" * 60)
    print(f"  Metadata backfill: {'✓ SUCCESS' if backfill_ok else '✗ FAILED'}")
    print(f"  Concept linking:   {'✓ SUCCESS' if concept_ok else '✗ FAILED'}")
    print(f"Completed: {datetime.now():%Y-%m-%d %H:%M}")


if __name__ == "__main__":
    main()
