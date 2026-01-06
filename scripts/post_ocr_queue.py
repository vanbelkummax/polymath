#!/usr/bin/env python3
"""
Post-OCR Queue - Runs after OCR completes
1. Ingests OCR results to ChromaDB
2. Runs metadata backfill
3. Runs concept linking
"""

import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

POLYMAX_DIR = "/home/user/work/polymax"
OCR_PID_FILE = f"{POLYMAX_DIR}/ocr_to_files.pid"


def is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def wait_for_ocr():
    print(f"[{datetime.now():%H:%M}] Checking for OCR process...")

    if not os.path.exists(OCR_PID_FILE):
        print("  No OCR PID file - proceeding")
        return

    try:
        pid = int(Path(OCR_PID_FILE).read_text().strip())
    except:
        print("  Could not read PID - proceeding")
        return

    if not is_running(pid):
        print(f"  OCR (PID {pid}) already done")
        return

    print(f"  OCR running (PID {pid}) - waiting...")
    while is_running(pid):
        time.sleep(60)
        if int(time.time()) % 300 == 0:  # Every 5 min
            print(f"  [{datetime.now():%H:%M}] Still waiting...")

    print(f"  [{datetime.now():%H:%M}] OCR completed!")
    time.sleep(5)


def run_script(name: str, desc: str) -> bool:
    print(f"\n[{datetime.now():%H:%M}] {desc}...")
    result = subprocess.run(
        [sys.executable, f"scripts/{name}"],
        cwd=POLYMAX_DIR
    )
    success = result.returncode == 0
    print(f"  {'✓ Done' if success else '✗ Failed'}")
    return success


def main():
    print("=" * 60)
    print("POST-OCR MAINTENANCE QUEUE")
    print("=" * 60)
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M}\n")

    # Wait for OCR
    wait_for_ocr()

    # Run tasks in sequence
    results = {}

    results['ingest'] = run_script("ingest_ocr_results.py", "Ingesting OCR results")
    results['backfill'] = run_script("backfill_metadata.py", "Backfilling metadata")
    results['concepts'] = run_script("fix_concept_links.py", "Fixing concept links")

    print("\n" + "=" * 60)
    print("QUEUE COMPLETE")
    print("=" * 60)
    for task, ok in results.items():
        print(f"  {task}: {'✓' if ok else '✗'}")
    print(f"Finished: {datetime.now():%Y-%m-%d %H:%M}")


if __name__ == "__main__":
    main()
