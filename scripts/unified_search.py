#!/usr/bin/env python3
"""
Unified Search for Polymath - Query both code and papers (BGE-M3 default)
Returns clearly labeled results from each source.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib.config import CHROMADB_PATH, PAPERS_COLLECTION, CODE_COLLECTION, EMBEDDING_MODEL

# Embedding model - must match what was used during ingestion
PAPER_MODEL = EMBEDDING_MODEL
CODE_MODEL = EMBEDDING_MODEL

# Cache models
_models = {}

# Icons for result types
ICONS = {
    'paper': '📄',
    'code': '💻',
    'function': '🔧',
    'class': '📦',
    'notebook_code': '📓',
    'notebook_markdown': '📝',
    'markdown': '📋',
    'config': '⚙️',
    'module': '📁',
    'raw': '📜',
}


def get_icon(source: str, type_: str) -> str:
    """Get appropriate icon for result type."""
    if source == 'paper':
        return ICONS['paper']
    return ICONS.get(type_, ICONS['code'])


def get_model(model_name: str) -> SentenceTransformer:
    """Get or load embedding model."""
    if model_name not in _models:
        _models[model_name] = SentenceTransformer(model_name)
    return _models[model_name]


def search_papers(client, query: str, n: int = 5) -> List[Dict]:
    """Search paper collection with correct embeddings."""
    try:
        model = get_model(PAPER_MODEL)
        embedding = model.encode([query])[0].tolist()

        coll = client.get_collection(PAPERS_COLLECTION)
        results = coll.query(
            query_embeddings=[embedding],
            n_results=n,
            include=['documents', 'metadatas', 'distances']
        )

        output = []
        for i, (doc, meta, dist) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            title = (meta.get('title') or meta.get('doc_title') or meta.get('paper_title') or 'Unknown')[:80]
            output.append({
                'source': 'paper',
                'type': 'paper_chunk',
                'title': title,
                'preview': doc[:200].replace('\n', ' '),
                'metadata': meta,
                'distance': dist,
                'icon': ICONS['paper']
            })
        return output
    except Exception as e:
        print(f"Paper search error: {e}")
        return []


def search_code(client, query: str, n: int = 5) -> List[Dict]:
    """Search code collection with correct embeddings."""
    try:
        model = get_model(CODE_MODEL)
        embedding = model.encode([query])[0].tolist()

        coll = client.get_collection(CODE_COLLECTION)
        results = coll.query(
            query_embeddings=[embedding],
            n_results=n,
            include=['documents', 'metadatas', 'distances']
        )

        output = []
        for i, (doc, meta, dist) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            type_ = meta.get('chunk_type') or meta.get('type', 'code')
            name = meta.get('name') or meta.get('symbol') or 'unknown'
            repo = meta.get('repo_name') or meta.get('repo', '')
            file_ = (meta.get('file_path') or meta.get('file') or '').split('/')[-1]

            output.append({
                'source': 'code',
                'type': type_,
                'title': f"{type_}: {name}",
                'location': f"{repo}/{file_}",
                'preview': doc[:200].replace('\n', ' '),
                'metadata': meta,
                'distance': dist,
                'icon': get_icon('code', type_)
            })
        return output
    except Exception as e:
        print(f"Code search error: {e}")
        return []


def unified_search(query: str, n_each: int = 5, source: str = 'all') -> Tuple[List[Dict], List[Dict]]:
    """
    Search both collections and return results.

    Args:
        query: Search query
        n_each: Number of results from each source
        source: 'all', 'code', or 'papers'

    Returns:
        Tuple of (paper_results, code_results)
    """
    client = chromadb.PersistentClient(path=str(CHROMADB_PATH))

    papers = []
    code = []

    if source in ['all', 'papers']:
        papers = search_papers(client, query, n_each)

    if source in ['all', 'code']:
        code = search_code(client, query, n_each)

    return papers, code


def format_results(papers: List[Dict], code: List[Dict], verbose: bool = False) -> str:
    """Format results for display."""
    lines = []

    if papers:
        lines.append("\n" + "=" * 60)
        lines.append("📄 PAPER RESULTS")
        lines.append("=" * 60)
        for i, r in enumerate(papers, 1):
            lines.append(f"\n{i}. {r['title']}")
            if verbose:
                lines.append(f"   Distance: {r['distance']:.4f}")
            lines.append(f"   {r['preview']}...")

    if code:
        lines.append("\n" + "=" * 60)
        lines.append("💻 CODE RESULTS")
        lines.append("=" * 60)
        for i, r in enumerate(code, 1):
            lines.append(f"\n{i}. {r['icon']} {r['title']}")
            lines.append(f"   Location: {r['location']}")
            if verbose:
                lines.append(f"   Distance: {r['distance']:.4f}")
            # Show first line of code preview
            preview = r['preview'].split('\n')[0][:100]
            lines.append(f"   {preview}...")

    if not papers and not code:
        lines.append("No results found.")

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Unified search across papers and code')
    parser.add_argument('query', nargs='+', help='Search query')
    parser.add_argument('-n', type=int, default=5, help='Results per source (default: 5)')
    parser.add_argument('--code', action='store_true', help='Search code only')
    parser.add_argument('--papers', action='store_true', help='Search papers only')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show distances')

    args = parser.parse_args()
    query = ' '.join(args.query)

    # Determine source
    if args.code:
        source = 'code'
    elif args.papers:
        source = 'papers'
    else:
        source = 'all'

    print(f"\n🔍 Searching: \"{query}\"")
    print(f"   Source: {source}")

    papers, code = unified_search(query, args.n, source)
    print(format_results(papers, code, args.verbose))


if __name__ == "__main__":
    main()
