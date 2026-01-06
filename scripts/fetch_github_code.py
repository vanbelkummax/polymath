#!/usr/bin/env python3
"""
GitHub Code Fetcher for Polymath
Fetches code from GitHub repos without full cloning.
Uses GitHub API to get specific files.
"""

import os
import re
import json
import base64
import requests
from pathlib import Path
from typing import List, Dict, Optional
import chromadb
from sentence_transformers import SentenceTransformer

# Config
CHROMADB_PATH = "/home/user/work/polymax/chromadb/polymath_v2"
COLLECTION_NAME = "code_corpus"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN") or os.environ.get("PAT_2")

# File patterns to fetch
FETCH_PATTERNS = [
    "README.md",
    "*.py",
    "*.ipynb",
    "models/*.py",
    "src/*.py",
    "lib/*.py",
]

# Skip patterns
SKIP_PATTERNS = [
    "test_*.py",
    "*_test.py",
    "setup.py",
    "conftest.py",
]


def get_github_headers():
    """Get headers for GitHub API."""
    headers = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    return headers


def fetch_repo_tree(owner: str, repo: str, branch: str = "main") -> List[Dict]:
    """Fetch repository file tree."""
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    response = requests.get(url, headers=get_github_headers())

    if response.status_code != 200:
        # Try master branch
        url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/master?recursive=1"
        response = requests.get(url, headers=get_github_headers())

    if response.status_code != 200:
        print(f"Error fetching tree: {response.status_code}")
        return []

    data = response.json()
    return data.get("tree", [])


def should_fetch_file(path: str) -> bool:
    """Check if file should be fetched based on patterns."""
    filename = os.path.basename(path)
    ext = os.path.splitext(filename)[1]

    # Skip patterns
    for pattern in SKIP_PATTERNS:
        if pattern.startswith("*"):
            if filename.endswith(pattern[1:]):
                return False
        elif filename == pattern:
            return False

    # Check extension
    if ext in [".py", ".ipynb", ".md", ".yaml", ".yml"]:
        return True

    return False


def fetch_file_content(owner: str, repo: str, path: str) -> Optional[str]:
    """Fetch content of a single file."""
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    response = requests.get(url, headers=get_github_headers())

    if response.status_code != 200:
        return None

    data = response.json()
    if data.get("encoding") == "base64":
        content = base64.b64decode(data["content"]).decode("utf-8", errors="ignore")
        return content

    return None


def chunk_python_code(content: str, filepath: str, repo_name: str) -> List[Dict]:
    """Chunk Python code by functions/classes."""
    import ast
    import hashlib

    chunks = []
    try:
        tree = ast.parse(content)
        lines = content.split('\n')

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start_line = node.lineno - 1
                end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 20
                chunk_text = '\n'.join(lines[start_line:end_line])

                docstring = ast.get_docstring(node) or ""
                chunk_type = "class" if isinstance(node, ast.ClassDef) else "function"

                chunks.append({
                    'id': hashlib.md5(f"{filepath}:{node.name}:{start_line}".encode()).hexdigest(),
                    'text': f"{chunk_type}: {node.name}\n{docstring}\n\n{chunk_text}",
                    'metadata': {
                        'repo': repo_name,
                        'file': filepath,
                        'type': chunk_type,
                        'name': node.name,
                        'source': 'code',
                        'origin': 'github_remote'
                    }
                })
    except SyntaxError:
        pass

    return chunks


def chunk_markdown(content: str, filepath: str, repo_name: str) -> List[Dict]:
    """Chunk markdown by sections."""
    import hashlib

    chunks = []
    sections = content.split('\n## ')

    for i, section in enumerate(sections):
        if len(section.strip()) < 50:
            continue

        lines = section.split('\n')
        title = lines[0].replace('#', '').strip()[:100]

        chunks.append({
            'id': hashlib.md5(f"{filepath}:{title}:{i}".encode()).hexdigest(),
            'text': f"markdown: {title}\n\n{section[:5000]}",
            'metadata': {
                'repo': repo_name,
                'file': filepath,
                'type': 'markdown',
                'name': title,
                'source': 'code',
                'origin': 'github_remote'
            }
        })

    return chunks


def fetch_and_ingest_repo(owner: str, repo: str, collection) -> int:
    """Fetch and ingest a GitHub repository."""
    print(f"\n📦 Fetching: {owner}/{repo}")

    tree = fetch_repo_tree(owner, repo)
    if not tree:
        print(f"   ⚠️ Could not fetch repository tree")
        return 0

    files_to_fetch = [f for f in tree if f["type"] == "blob" and should_fetch_file(f["path"])]
    print(f"   Found {len(files_to_fetch)} files to process")

    all_chunks = []
    for file_info in files_to_fetch[:50]:  # Limit to 50 files per repo
        path = file_info["path"]
        content = fetch_file_content(owner, repo, path)

        if not content:
            continue

        if path.endswith(".py"):
            chunks = chunk_python_code(content, path, f"{owner}/{repo}")
        elif path.endswith(".md"):
            chunks = chunk_markdown(content, path, f"{owner}/{repo}")
        else:
            continue

        all_chunks.extend(chunks)

    if all_chunks:
        ids = [c['id'] for c in all_chunks]
        documents = [c['text'] for c in all_chunks]
        metadatas = [c['metadata'] for c in all_chunks]

        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        print(f"   ✓ Ingested {len(all_chunks)} chunks")

    return len(all_chunks)


def extract_github_urls_from_papers(paper_collection) -> List[str]:
    """Extract GitHub URLs mentioned in papers."""
    results = paper_collection.query(
        query_texts=["github repository code available implementation"],
        n_results=100,
        include=['documents']
    )

    urls = set()
    pattern = r'github\.com/([\w-]+)/([\w-]+)'

    for doc in results['documents'][0]:
        matches = re.findall(pattern, doc)
        for owner, repo in matches:
            urls.add(f"{owner}/{repo}")

    return list(urls)


def main():
    import sys

    print("=" * 60)
    print("GITHUB CODE FETCHER FOR POLYMATH")
    print("=" * 60)

    if not GITHUB_TOKEN:
        print("⚠️  No GITHUB_TOKEN or PAT_2 found. Rate limits will apply.")
        print("   Set GITHUB_TOKEN or PAT_2 environment variable for higher limits.")

    # Connect to ChromaDB
    client = chromadb.PersistentClient(path=CHROMADB_PATH)
    code_collection = client.get_or_create_collection(name=COLLECTION_NAME)

    print(f"\nExisting code chunks: {code_collection.count():,}")

    # Mode: specific repos or auto-discover from papers
    if len(sys.argv) > 1:
        # Specific repos provided
        repos = sys.argv[1:]
    else:
        # Auto-discover from papers
        print("\nDiscovering GitHub repos mentioned in papers...")
        paper_collection = client.get_collection('polymath_corpus')
        repos = extract_github_urls_from_papers(paper_collection)
        print(f"Found {len(repos)} unique repos")

    total = 0
    for repo_str in repos[:20]:  # Limit to 20 repos
        if '/' in repo_str:
            owner, repo = repo_str.split('/', 1)
            repo = repo.split('/')[0]  # Handle trailing paths
            count = fetch_and_ingest_repo(owner, repo, code_collection)
            total += count

    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total chunks ingested: {total:,}")
    print(f"Collection total: {code_collection.count():,}")


if __name__ == "__main__":
    main()
