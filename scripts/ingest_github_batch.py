#!/usr/bin/env python3
"""
Batch ingest all GitHub repos from the data/github_repos directory.
Ingests Python, TypeScript, JavaScript, Rust, Go, and Markdown files.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
from unified_ingest import UnifiedIngestor

GITHUB_REPOS_DIR = "/home/user/work/polymax/data/github_repos"

# File extensions to ingest
CODE_EXTENSIONS = {".py", ".ts", ".tsx", ".js", ".jsx", ".rs", ".go"}
DOC_EXTENSIONS = {".md", ".rst", ".txt"}

# Skip patterns
SKIP_PATTERNS = [
    "__pycache__", ".git", "node_modules", ".venv", "venv",
    "dist", "build", ".next", "target", ".cargo",
    "vendor", ".pytest_cache", ".mypy_cache",
    "chromadb", "test_", "_test.py", "tests/fixtures",
]

# Priority repos (ingest more thoroughly)
PRIORITY_REPOS = {
    "claude-code", "skills", "agents.md", "papers-we-love",
    "openai-cookbook", "exo", "ai-hedge-fund", "opencode",
}


def should_skip(path: str) -> bool:
    """Check if path should be skipped."""
    path_lower = path.lower()
    return any(pattern in path_lower for pattern in SKIP_PATTERNS)


def find_files(root: str, extensions: set, max_files: int = 500) -> list:
    """Find files with given extensions in a directory."""
    files = []
    for ext in extensions:
        for path in Path(root).rglob(f"*{ext}"):
            if not should_skip(str(path)):
                files.append(str(path))
                if len(files) >= max_files:
                    return files
    return files


def main():
    ingestor = UnifiedIngestor()

    repos = sorted([d for d in Path(GITHUB_REPOS_DIR).iterdir() if d.is_dir()])
    print(f"Found {len(repos)} repos to process")

    total_files = 0
    total_chunks = 0
    results = []

    for repo_dir in repos:
        repo_name = repo_dir.name
        is_priority = repo_name in PRIORITY_REPOS
        max_files = 500 if is_priority else 100

        print(f"\n{'='*60}")
        print(f"Processing: {repo_name} {'[PRIORITY]' if is_priority else ''}")
        print(f"{'='*60}")

        # Find code files
        code_files = find_files(str(repo_dir), CODE_EXTENSIONS, max_files)
        doc_files = find_files(str(repo_dir), DOC_EXTENSIONS, max_files // 2)

        all_files = code_files + doc_files

        if not all_files:
            print(f"  No ingestable files found, skipping")
            continue

        print(f"  Found {len(code_files)} code + {len(doc_files)} doc files")

        repo_chunks = 0
        repo_concepts = set()

        for filepath in all_files:
            try:
                ext = Path(filepath).suffix

                if ext in CODE_EXTENSIONS:
                    result = ingestor.ingest_code(
                        filepath,
                        repo_name=repo_name,
                        ingestion_method="github_batch_file",
                    )
                else:
                    # For markdown/text, use ingest_code with text handling
                    result = ingestor.ingest_code(
                        filepath,
                        repo_name=repo_name,
                        ingestion_method="github_batch_doc",
                    )

                if result:
                    repo_chunks += result.chunks_added
                    repo_concepts.update(result.concepts_linked)

            except Exception as e:
                print(f"  Error ingesting {filepath}: {e}")

        total_files += len(all_files)
        total_chunks += repo_chunks

        results.append({
            "repo": repo_name,
            "files": len(all_files),
            "chunks": repo_chunks,
            "concepts": list(repo_concepts)[:10]
        })

        print(f"  Ingested: {len(all_files)} files, {repo_chunks} chunks")
        if repo_concepts:
            print(f"  Concepts: {', '.join(list(repo_concepts)[:5])}")

    # Summary
    print(f"\n{'='*60}")
    print(f"INGESTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total repos: {len(repos)}")
    print(f"Total files: {total_files}")
    print(f"Total chunks: {total_chunks}")

    # Write report
    report_path = f"/home/user/work/polymax/reports/github_ingest_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, "w") as f:
        f.write(f"# GitHub Repos Ingestion Report\n\n")
        f.write(f"Date: {datetime.now().isoformat()}\n\n")
        f.write(f"## Summary\n")
        f.write(f"- Repos processed: {len(repos)}\n")
        f.write(f"- Files ingested: {total_files}\n")
        f.write(f"- Chunks created: {total_chunks}\n\n")
        f.write(f"## Per-Repo Details\n\n")
        for r in results:
            f.write(f"### {r['repo']}\n")
            f.write(f"- Files: {r['files']}\n")
            f.write(f"- Chunks: {r['chunks']}\n")
            if r['concepts']:
                f.write(f"- Concepts: {', '.join(r['concepts'])}\n")
            f.write("\n")

    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()
