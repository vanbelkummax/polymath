#!/usr/bin/env python3
"""
Ingest Python code from key repositories into Polymath.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
from unified_ingest import UnifiedIngestor

# Repos to ingest
REPOS = [
    ("/home/user/sota-2um-st-prediction", "sota-2um"),
    ("/home/user/work/iSCALE", "iSCALE"),
    ("/home/user/work/polymax/lib", "polymax-lib"),
    ("/home/user/work/polymax/scripts", "polymax-scripts"),
    ("/home/user/work/polymax/mcp", "polymax-mcp"),
]

# Skip patterns
SKIP_PATTERNS = [
    "__pycache__",
    ".git",
    "chromadb",
    "node_modules",
    ".conda",
    "miniforge",
    "mamba",
]


def should_skip(path: str) -> bool:
    """Check if path should be skipped."""
    for pattern in SKIP_PATTERNS:
        if pattern in path:
            return True
    return False


def find_python_files(root: str) -> list:
    """Find all Python files in a directory."""
    files = []
    for path in Path(root).rglob("*.py"):
        if not should_skip(str(path)):
            files.append(str(path))
    return files


def main():
    ingestor = UnifiedIngestor()

    total_files = 0
    total_chunks = 0
    total_concepts = set()

    for repo_path, repo_name in REPOS:
        if not os.path.exists(repo_path):
            print(f"Skipping {repo_name}: path not found")
            continue

        files = find_python_files(repo_path)
        print(f"\n{'='*50}")
        print(f"Ingesting {repo_name}: {len(files)} Python files")
        print(f"{'='*50}")

        for i, file_path in enumerate(files):
            try:
                result = ingestor.ingest_code(
                    file_path,
                    repo_name=repo_name,
                    ingestion_method="repo_file_batch",
                )
                total_files += 1
                total_chunks += result.chunks_added
                total_concepts.update(result.concepts_linked)

                if result.chunks_added > 0:
                    print(f"  [{i+1}/{len(files)}] {Path(file_path).name}: {result.chunks_added} chunks")
                else:
                    print(f"  [{i+1}/{len(files)}] {Path(file_path).name}: skipped (no chunks)")

            except Exception as e:
                print(f"  [{i+1}/{len(files)}] {Path(file_path).name}: ERROR - {e}")

    print(f"\n{'='*50}")
    print(f"CODE INGESTION COMPLETE")
    print(f"Total files: {total_files}")
    print(f"Total chunks: {total_chunks}")
    print(f"Unique concepts: {len(total_concepts)}")
    print(f"Concepts: {', '.join(sorted(total_concepts)[:20])}")


if __name__ == "__main__":
    main()
