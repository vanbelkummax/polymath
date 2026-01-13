#!/usr/bin/env python3
"""
Batch code ingestion for Polymath System - handles recursive directory scanning.
Usage: python3 scripts/batch_code_ingest.py /path/to/repo --repo-name squidpy
"""
import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import logging

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.unified_ingest import TransactionalIngestor, IngestResult, BatchResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Patterns to skip
SKIP_PATTERNS = [
    '__pycache__', '.git', '.github', 'node_modules', 'venv', 'env',
    '.tox', '.mypy_cache', '.pytest_cache', 'build', 'dist', '.eggs',
    'migrations', 'test_data', 'fixtures'
]

# Extensions to index
CODE_EXTENSIONS = ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.hpp', '.go', '.rs', '.rb']


def should_skip(path: Path) -> bool:
    """Check if path should be skipped."""
    parts = path.parts
    for skip in SKIP_PATTERNS:
        if skip in parts:
            return True
    return False


def find_code_files(repo_path: Path, extensions: list = None) -> list:
    """Recursively find all code files."""
    extensions = extensions or CODE_EXTENSIONS
    files = []

    for ext in extensions:
        for f in repo_path.glob(f"**/*{ext}"):
            if not should_skip(f):
                files.append(f)

    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="Batch code ingestion")
    parser.add_argument("repo_path", help="Path to repository")
    parser.add_argument("--repo-name", help="Repository name (default: directory name)")
    parser.add_argument("--extensions", nargs="+", default=[".py"],
                       help="File extensions to index (default: .py)")
    parser.add_argument("--dry-run", action="store_true", help="Count files only")
    parser.add_argument("--limit", type=int, help="Limit number of files")
    parser.add_argument("--log-file", help="Log file path")
    args = parser.parse_args()

    repo_path = Path(args.repo_path).resolve()
    repo_name = args.repo_name or repo_path.name

    if not repo_path.is_dir():
        print(f"Error: {repo_path} is not a directory")
        return 1

    # Find files
    print(f"Scanning {repo_path} for code files...")
    files = find_code_files(repo_path, args.extensions)

    if args.limit:
        files = files[:args.limit]

    print(f"Found {len(files)} code files")

    if args.dry_run:
        for f in files[:10]:
            print(f"  {f.relative_to(repo_path)}")
        if len(files) > 10:
            print(f"  ... and {len(files) - 10} more")
        return 0

    if not files:
        print("No files to ingest")
        return 0

    # Initialize ingestor
    ingestor = TransactionalIngestor(use_ocr=False)
    ingest_run_id = ingestor._ensure_ingest_run(None)

    result = BatchResult(
        total_files=len(files),
        successful=0,
        failed=0,
        chunks_added=0,
        ingest_run_id=ingest_run_id
    )

    log_entries = []

    # Process files
    for i, code_file in enumerate(files, 1):
        relative_path = code_file.relative_to(repo_path)
        print(f"[{i}/{len(files)}] Ingesting: {relative_path}", end=" ")

        try:
            file_result = ingestor.ingest_code(
                str(code_file),
                repo_name=repo_name,
                ingest_run_id=ingest_run_id,
                ingestion_method="batch_code_ingest"
            )

            if file_result.postgres_synced:
                result.successful += 1
                result.chunks_added += file_result.chunks_added
                print(f"✓ ({file_result.chunks_added} chunks)")
                status = "success"
            else:
                result.failed += 1
                print(f"✗ ({file_result.errors})")
                status = "failed"
                result.errors[str(code_file)] = str(file_result.errors)

            log_entries.append({
                "file": str(relative_path),
                "status": status,
                "chunks": file_result.chunks_added,
                "timestamp": datetime.now().isoformat()
            })

        except Exception as e:
            result.failed += 1
            print(f"✗ ({e})")
            result.errors[str(code_file)] = str(e)
            log_entries.append({
                "file": str(relative_path),
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })

    # Write log if specified
    if args.log_file:
        with open(args.log_file, 'a') as f:
            for entry in log_entries:
                f.write(json.dumps(entry) + "\n")

    # Summary
    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE: {repo_name}")
    print(f"Total files: {result.total_files}")
    print(f"Successful: {result.successful}")
    print(f"Failed: {result.failed}")
    print(f"Chunks added: {result.chunks_added}")

    if result.errors:
        print(f"\nFirst 5 errors:")
        for path, err in list(result.errors.items())[:5]:
            print(f"  {Path(path).name}: {err[:50]}...")

    return 0


if __name__ == "__main__":
    exit(main())
