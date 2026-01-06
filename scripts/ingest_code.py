#!/usr/bin/env python3
"""
Code Ingestion for Polymath - AST-aware chunking
Ingests Python files, notebooks, and markdown from repos.
Lightweight: skips weights, data, venv.
"""

import os
import ast
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional
import chromadb
from sentence_transformers import SentenceTransformer

# Config
CHROMADB_PATH = "/home/user/work/polymax/chromadb/polymath_v2"
COLLECTION_NAME = "code_corpus"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# File patterns to include
INCLUDE_EXTENSIONS = {'.py', '.ipynb', '.md', '.yaml', '.yml', '.json'}

# Directories to skip
SKIP_DIRS = {
    '__pycache__', '.git', 'venv', 'env', '.venv',
    'node_modules', '.eggs', '*.egg-info', 'dist', 'build',
    'checkpoints', 'weights', 'models', 'data', 'outputs',
    '.ipynb_checkpoints', 'wandb', 'lightning_logs'
}

# Max file size (skip large files like model configs with weights)
MAX_FILE_SIZE = 500_000  # 500KB


def should_skip_dir(dirname: str) -> bool:
    """Check if directory should be skipped."""
    return any(dirname == skip or dirname.endswith(skip.replace('*', ''))
               for skip in SKIP_DIRS)


def extract_python_chunks(filepath: Path, content: str) -> List[Dict]:
    """Extract functions and classes from Python file using AST."""
    chunks = []
    try:
        tree = ast.parse(content)
        lines = content.split('\n')

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # Get the source code for this node
                start_line = node.lineno - 1
                end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 20
                chunk_lines = lines[start_line:end_line]
                chunk_text = '\n'.join(chunk_lines)

                # Get docstring if present
                docstring = ast.get_docstring(node) or ""

                # Create chunk
                chunk_type = "class" if isinstance(node, ast.ClassDef) else "function"
                chunks.append({
                    'text': chunk_text,
                    'type': chunk_type,
                    'name': node.name,
                    'docstring': docstring,
                    'start_line': start_line + 1,
                    'end_line': end_line,
                    'file': str(filepath)
                })

        # If no functions/classes found, chunk the whole file
        if not chunks and len(content) > 100:
            chunks.append({
                'text': content[:10000],  # First 10k chars
                'type': 'module',
                'name': filepath.stem,
                'docstring': '',
                'start_line': 1,
                'end_line': len(lines),
                'file': str(filepath)
            })

    except SyntaxError:
        # If AST parsing fails, treat as raw text
        chunks.append({
            'text': content[:10000],
            'type': 'raw',
            'name': filepath.stem,
            'docstring': '',
            'start_line': 1,
            'end_line': content.count('\n'),
            'file': str(filepath)
        })

    return chunks


def extract_notebook_chunks(filepath: Path, content: str) -> List[Dict]:
    """Extract code cells from Jupyter notebook."""
    chunks = []
    try:
        nb = json.loads(content)
        cells = nb.get('cells', [])

        for i, cell in enumerate(cells):
            cell_type = cell.get('cell_type', 'code')
            source = ''.join(cell.get('source', []))

            if len(source.strip()) < 50:
                continue

            chunks.append({
                'text': source[:8000],
                'type': f'notebook_{cell_type}',
                'name': f'{filepath.stem}_cell_{i}',
                'docstring': '',
                'start_line': i,
                'end_line': i,
                'file': str(filepath)
            })
    except json.JSONDecodeError:
        pass

    return chunks


def extract_markdown_chunks(filepath: Path, content: str) -> List[Dict]:
    """Extract sections from markdown files."""
    chunks = []
    sections = content.split('\n## ')

    for i, section in enumerate(sections):
        if len(section.strip()) < 50:
            continue

        # Get section title
        lines = section.split('\n')
        title = lines[0].replace('#', '').strip() if lines else filepath.stem

        chunks.append({
            'text': section[:8000],
            'type': 'markdown',
            'name': title[:100],
            'docstring': '',
            'start_line': i,
            'end_line': i,
            'file': str(filepath)
        })

    return chunks


def process_file(filepath: Path) -> List[Dict]:
    """Process a single file and return chunks."""
    try:
        if not filepath.exists() or filepath.is_symlink():
            return []
        if filepath.stat().st_size > MAX_FILE_SIZE:
            return []
    except (OSError, FileNotFoundError):
        return []

    try:
        content = filepath.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return []

    if len(content.strip()) < 100:
        return []

    ext = filepath.suffix.lower()

    if ext == '.py':
        return extract_python_chunks(filepath, content)
    elif ext == '.ipynb':
        return extract_notebook_chunks(filepath, content)
    elif ext == '.md':
        return extract_markdown_chunks(filepath, content)
    elif ext in {'.yaml', '.yml', '.json'}:
        # Config files - just chunk as-is
        return [{
            'text': content[:8000],
            'type': 'config',
            'name': filepath.name,
            'docstring': '',
            'start_line': 1,
            'end_line': content.count('\n'),
            'file': str(filepath)
        }]

    return []


def walk_repo(repo_path: Path) -> List[Dict]:
    """Walk a repository and extract all code chunks."""
    all_chunks = []
    repo_name = repo_path.name

    for root, dirs, files in os.walk(repo_path):
        # Filter out directories to skip
        dirs[:] = [d for d in dirs if not should_skip_dir(d)]

        for filename in files:
            filepath = Path(root) / filename
            if filepath.suffix.lower() not in INCLUDE_EXTENSIONS:
                continue

            chunks = process_file(filepath)
            for chunk in chunks:
                chunk['repo'] = repo_name
                # Create unique ID
                chunk['id'] = hashlib.md5(
                    f"{chunk['file']}:{chunk['name']}:{chunk['start_line']}".encode()
                ).hexdigest()
                all_chunks.append(chunk)

    return all_chunks


def ingest_to_chromadb(chunks: List[Dict], collection):
    """Ingest chunks into ChromaDB."""
    if not chunks:
        return 0

    # Prepare for ChromaDB
    ids = [c['id'] for c in chunks]
    documents = [f"{c['type']}: {c['name']}\n{c['docstring']}\n\n{c['text']}" for c in chunks]
    metadatas = [{
        'repo': c['repo'],
        'file': c['file'],
        'type': c['type'],
        'name': c['name'],
        'start_line': c['start_line'],
        'source': 'code'
    } for c in chunks]

    # Batch upsert
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i+batch_size]
        batch_docs = documents[i:i+batch_size]
        batch_meta = metadatas[i:i+batch_size]

        collection.upsert(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_meta
        )

    return len(chunks)


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ingest_code.py <repo_path> [repo_path2] ...")
        print("\nExample:")
        print("  python ingest_code.py /home/user/hist2st-benchmark /home/user/work/iSCALE")
        sys.exit(1)

    repo_paths = [Path(p) for p in sys.argv[1:]]

    print("=" * 60)
    print("CODE INGESTION FOR POLYMATH")
    print("=" * 60)

    # Connect to ChromaDB
    print("\nConnecting to ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMADB_PATH)

    # Get or create code collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Code chunks from repositories"}
    )

    existing = collection.count()
    print(f"Existing code chunks: {existing:,}")

    total_ingested = 0

    for repo_path in repo_paths:
        if not repo_path.exists():
            print(f"\n⚠️  Repo not found: {repo_path}")
            continue

        print(f"\n📁 Processing: {repo_path.name}")
        chunks = walk_repo(repo_path)
        print(f"   Found {len(chunks)} chunks")

        if chunks:
            ingested = ingest_to_chromadb(chunks, collection)
            total_ingested += ingested
            print(f"   ✓ Ingested {ingested} chunks")

    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total new chunks: {total_ingested:,}")
    print(f"Collection total: {collection.count():,}")


if __name__ == "__main__":
    main()
