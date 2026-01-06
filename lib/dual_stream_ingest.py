#!/usr/bin/env python3
"""
Dual-Stream GitHub Repository Ingestion

Fixes the "README Bias" problem by ingesting repos in two streams:
1. Repo-level: Overview, structure, context (existing smart ingestion)
2. File-level: Key implementation files for code details

Priority files:
- model*.py, network*.py, architecture*.py, encoder*.py, decoder*.py
- Files in src/ or lib/ with 200-1000 lines (sweet spot)
- High class/function density files
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple
import logging

sys.path.insert(0, str(Path(__file__).parent))
from unified_ingest import UnifiedIngestor, IngestResult

logger = logging.getLogger(__name__)

# Patterns for key implementation files
KEY_FILE_PATTERNS = [
    "**/model*.py",
    "**/network*.py",
    "**/architecture*.py",
    "**/encoder*.py",
    "**/decoder*.py",
    "**/attention*.py",
    "**/transformer*.py",
    "**/embedding*.py",
    "**/*encoder*.py",      # Matches get_encoder.py, vit_encoder.py, etc.
    "**/*model*.py",        # Matches timm_model.py, base_model.py, etc.
    "**/*net*.py",          # Matches resnet.py, unet.py, etc.
    "src/**/*.py",
    "lib/**/*.py",
    "*/models/**/*.py",     # Common pattern: repo/models/
    "*/modeling/**/*.py",   # Common pattern: repo/modeling/
]

# Skip patterns (same as GitHubFlattener)
SKIP_PATTERNS = [
    "test_", "_test.py", "tests/", "examples/",
    "migrations/", "__pycache__/", "dist/", "build/",
]

# Size thresholds
MIN_LINES = 50      # Skip trivial files
MAX_LINES = 1000    # Skip huge files (likely generated/data)
SWEET_SPOT_MIN = 200  # Prefer files in sweet spot
SWEET_SPOT_MAX = 1000


def is_key_implementation_file(filepath: Path) -> bool:
    """
    Check if file is a key implementation file.

    Criteria:
    - Matches key patterns (model*.py, etc.)
    - Size in reasonable range
    - Not in skip patterns
    """
    path_str = str(filepath)

    # Check skip patterns
    if any(pattern in path_str for pattern in SKIP_PATTERNS):
        return False

    # Check if matches key patterns
    matched = False
    for pattern in KEY_FILE_PATTERNS:
        if filepath.match(pattern):
            matched = True
            break

    if not matched:
        return False

    # Check file size
    try:
        lines = sum(1 for _ in open(filepath, errors='ignore'))
        if lines < MIN_LINES or lines > MAX_LINES:
            return False
        return True
    except Exception:
        return False


def calculate_code_density(filepath: Path) -> float:
    """
    Calculate code density (class + function defs per 100 lines).

    Higher density = more implementation, less boilerplate.
    """
    try:
        content = filepath.read_text(errors='ignore')
        lines = content.count('\n') + 1

        # Count definitions
        num_classes = content.count('\nclass ')
        num_functions = content.count('\ndef ') + content.count('\nasync def ')

        density = (num_classes + num_functions) * 100 / lines
        return density
    except Exception:
        return 0.0


def find_key_implementation_files(
    repo_path: Path,
    max_files: int = 20
) -> List[Tuple[Path, float]]:
    """
    Find key implementation files in a repository.

    Returns:
        List of (filepath, priority_score) tuples, sorted by priority.
        Higher priority = more important to ingest.
    """
    candidates = []

    for pattern in KEY_FILE_PATTERNS:
        for filepath in repo_path.glob(pattern):
            if not filepath.is_file():
                continue

            if not is_key_implementation_file(filepath):
                continue

            # Calculate priority score
            try:
                lines = sum(1 for _ in open(filepath, errors='ignore'))
                density = calculate_code_density(filepath)

                # Priority factors:
                # 1. Code density (30%)
                # 2. Sweet spot size (40%)
                # 3. Pattern match priority (30%)

                density_score = min(density / 10, 1.0)  # Cap at 1.0

                # Sweet spot bonus
                if SWEET_SPOT_MIN <= lines <= SWEET_SPOT_MAX:
                    size_score = 1.0
                else:
                    size_score = 0.5

                # Pattern match priority (model.py > utils.py)
                pattern_score = 1.0
                if 'model' in filepath.stem or 'network' in filepath.stem:
                    pattern_score = 1.5
                elif 'util' in filepath.stem or 'helper' in filepath.stem:
                    pattern_score = 0.7

                priority = (
                    0.3 * density_score +
                    0.4 * size_score +
                    0.3 * pattern_score
                )

                candidates.append((filepath, priority))

            except Exception as e:
                logger.debug(f"Error analyzing {filepath}: {e}")
                continue

    # Sort by priority, limit
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:max_files]


def ingest_repo_dual_stream(
    repo_path: str,
    repo_name: str = None,
    max_key_files: int = 20
) -> Tuple[IngestResult, List[IngestResult]]:
    """
    Ingest repository using dual-stream approach.

    Returns:
        (repo_result, file_results)
        - repo_result: Result from repo-level ingestion
        - file_results: List of results from file-level ingestion
    """
    repo_path = Path(repo_path)
    repo_name = repo_name or repo_path.name

    ingestor = UnifiedIngestor()

    # Stream 1: Repo-level (overview, context)
    logger.info(f"Stream 1: Ingesting {repo_name} as flattened repo...")
    repo_result = ingestor.ingest_github_repo(
        str(repo_path),
        repo_name=repo_name,
        ingestion_method="dual_stream_repo",
    )

    # Stream 2: Key implementation files
    logger.info(f"Stream 2: Finding key implementation files in {repo_name}...")
    key_files = find_key_implementation_files(repo_path, max_files=max_key_files)

    logger.info(f"Found {len(key_files)} key implementation files")

    file_results = []
    for i, (filepath, priority) in enumerate(key_files):
        rel_path = filepath.relative_to(repo_path)
        logger.info(f"  [{i+1}/{len(key_files)}] {rel_path} (priority: {priority:.2f})")

        try:
            result = ingestor.ingest_code(
                str(filepath),
                repo_name=repo_name,
                ingestion_method="dual_stream_file",
            )
            file_results.append(result)
        except Exception as e:
            logger.error(f"    Error: {e}")

    # Summary
    total_chunks = repo_result.chunks_added + sum(r.chunks_added for r in file_results)
    logger.info(f"Dual-stream complete: {total_chunks} chunks")
    logger.info(f"  Repo-level: {repo_result.chunks_added} chunks")
    logger.info(f"  File-level: {sum(r.chunks_added for r in file_results)} chunks from {len(file_results)} files")

    return repo_result, file_results


def main():
    """Test dual-stream ingestion on UNI repo."""
    import argparse

    parser = argparse.ArgumentParser(description="Dual-stream repo ingestion")
    parser.add_argument("repo_path", help="Path to repository")
    parser.add_argument("--repo-name", help="Repository name")
    parser.add_argument("--max-files", type=int, default=20, help="Max key files to ingest")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    repo_result, file_results = ingest_repo_dual_stream(
        args.repo_path,
        repo_name=args.repo_name,
        max_key_files=args.max_files
    )

    print(f"\n{'='*60}")
    print("DUAL-STREAM INGESTION COMPLETE")
    print(f"{'='*60}")
    print(f"Repo: {repo_result.title}")
    print(f"  Repo-level chunks: {repo_result.chunks_added}")
    print(f"  File-level chunks: {sum(r.chunks_added for r in file_results)}")
    print(f"  Total chunks: {repo_result.chunks_added + sum(r.chunks_added for r in file_results)}")
    print(f"  Files ingested: {len(file_results)}")


if __name__ == "__main__":
    main()
