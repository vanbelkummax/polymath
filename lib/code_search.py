#!/usr/bin/env python3
"""
Hybrid Code Search

Combines:
- Postgres FTS (lexical, exact matches)
- Optional embeddings (semantic similarity)

Usage:
    from lib.code_search import search_code
    results = search_code("attention pooling", repo="mahmoodlab/*", limit=10)
"""
import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import psycopg2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CodeResult:
    """A code search result with citation-grade provenance."""
    chunk_id: str
    repo_name: str
    file_path: str
    commit_sha: str
    start_line: int
    end_line: int
    chunk_type: str
    name: str
    class_name: Optional[str]
    symbol_qualified_name: str
    content: str
    docstring: Optional[str]
    signature: Optional[str]
    concepts: List[str]
    score: float
    match_type: str  # 'fts', 'embedding', 'hybrid'

    def permalink(self) -> str:
        """Generate GitHub permalink."""
        # Convert git URL to web URL
        if not self.commit_sha or self.commit_sha == 'unknown':
            return f"{self.repo_name}/{self.file_path}#L{self.start_line}-L{self.end_line}"
        return f"https://github.com/{self.repo_name}/blob/{self.commit_sha}/{self.file_path}#L{self.start_line}-L{self.end_line}"

    def short_ref(self) -> str:
        """Short reference string."""
        return f"{self.repo_name}:{self.file_path}:{self.start_line}"


def get_db_connection():
    return psycopg2.connect(
        dbname='polymath',
        user='polymath',
        host='/var/run/postgresql'
    )


def search_code(
    query: str,
    repo: Optional[str] = None,
    language: Optional[str] = None,
    chunk_type: Optional[str] = None,
    concepts: Optional[List[str]] = None,
    limit: int = 20,
    fts_weight: float = 0.6,
    use_embeddings: bool = False
) -> List[CodeResult]:
    """
    Hybrid code search.

    Args:
        query: Search query
        repo: Filter by repo name (supports wildcards: 'mahmoodlab/*')
        language: Filter by language ('python', 'r', etc.)
        chunk_type: Filter by type ('function', 'method', 'class')
        concepts: Filter by auto-tagged concepts
        limit: Max results
        fts_weight: Weight for FTS vs embedding scores (0-1)
        use_embeddings: Whether to also search embeddings (slower)

    Returns:
        List of CodeResult objects with provenance
    """
    conn = get_db_connection()
    cur = conn.cursor()

    results = []
    seen_chunks = set()

    # === FTS Search ===
    fts_results = _fts_search(cur, query, repo, language, chunk_type, concepts, limit * 2)
    for rank, row in enumerate(fts_results):
        if row['chunk_id'] not in seen_chunks:
            seen_chunks.add(row['chunk_id'])
            row['score'] = (1.0 - rank / len(fts_results)) * fts_weight
            row['match_type'] = 'fts'
            results.append(row)

    # === Embedding Search (optional) ===
    if use_embeddings:
        emb_results = _embedding_search(query, repo, limit * 2)
        for rank, row in enumerate(emb_results):
            if row['chunk_id'] not in seen_chunks:
                seen_chunks.add(row['chunk_id'])
                row['score'] = (1.0 - rank / len(emb_results)) * (1 - fts_weight)
                row['match_type'] = 'embedding'
                results.append(row)
            else:
                # Boost score for items found by both
                for r in results:
                    if r['chunk_id'] == row['chunk_id']:
                        r['score'] += (1.0 - rank / len(emb_results)) * (1 - fts_weight)
                        r['match_type'] = 'hybrid'

    # Sort by score and limit
    results.sort(key=lambda x: x['score'], reverse=True)
    results = results[:limit]

    conn.close()

    # Convert to CodeResult objects
    return [_dict_to_result(r) for r in results]


def _fts_search(
    cur,
    query: str,
    repo: Optional[str],
    language: Optional[str],
    chunk_type: Optional[str],
    concepts: Optional[List[str]],
    limit: int
) -> List[Dict]:
    """Full-text search using Postgres tsvector."""
    # Build query
    # Convert to tsquery format
    terms = query.split()
    tsquery = ' & '.join(terms)

    sql = """
        SELECT
            c.chunk_id::text,
            f.repo_name,
            f.file_path,
            f.head_commit_sha,
            c.start_line,
            c.end_line,
            c.chunk_type,
            c.name,
            c.class_name,
            c.symbol_qualified_name,
            c.content,
            c.docstring,
            c.signature,
            c.concepts,
            ts_rank_cd(c.search_vector, plainto_tsquery('english', %s)) as rank
        FROM code_chunks c
        JOIN code_files f ON c.file_id = f.file_id
        WHERE c.search_vector @@ plainto_tsquery('english', %s)
    """
    params = [query, query]

    if repo:
        if '*' in repo:
            # Wildcard match
            pattern = repo.replace('*', '%')
            sql += " AND f.repo_name LIKE %s"
            params.append(pattern)
        else:
            sql += " AND f.repo_name = %s"
            params.append(repo)

    if language:
        sql += " AND f.language = %s"
        params.append(language)

    if chunk_type:
        sql += " AND c.chunk_type = %s"
        params.append(chunk_type)

    if concepts:
        sql += " AND c.concepts && %s"
        params.append(concepts)

    sql += " ORDER BY rank DESC LIMIT %s"
    params.append(limit)

    cur.execute(sql, tuple(params))
    columns = [desc[0] for desc in cur.description]
    return [dict(zip(columns, row)) for row in cur.fetchall()]


def _embedding_search(query: str, repo: Optional[str], limit: int) -> List[Dict]:
    """Semantic search using ChromaDB embeddings."""
    # TODO: Implement when embeddings are added
    # For now, return empty
    return []


def _dict_to_result(d: Dict) -> CodeResult:
    """Convert dict to CodeResult."""
    return CodeResult(
        chunk_id=d['chunk_id'],
        repo_name=d['repo_name'],
        file_path=d['file_path'],
        commit_sha=d.get('head_commit_sha', 'unknown'),
        start_line=d['start_line'],
        end_line=d['end_line'],
        chunk_type=d['chunk_type'],
        name=d['name'],
        class_name=d.get('class_name'),
        symbol_qualified_name=d['symbol_qualified_name'],
        content=d['content'],
        docstring=d.get('docstring'),
        signature=d.get('signature'),
        concepts=d.get('concepts', []),
        score=d.get('score', 0),
        match_type=d.get('match_type', 'fts')
    )


def find_by_symbol(symbol: str, exact: bool = False) -> List[CodeResult]:
    """Find code by symbol name."""
    conn = get_db_connection()
    cur = conn.cursor()

    if exact:
        sql = """
            SELECT
                c.chunk_id::text, f.repo_name, f.file_path, f.head_commit_sha,
                c.start_line, c.end_line, c.chunk_type, c.name, c.class_name,
                c.symbol_qualified_name, c.content, c.docstring, c.signature, c.concepts
            FROM code_chunks c
            JOIN code_files f ON c.file_id = f.file_id
            WHERE c.name = %s OR c.symbol_qualified_name = %s
            LIMIT 50
        """
        cur.execute(sql, (symbol, symbol))
    else:
        sql = """
            SELECT
                c.chunk_id::text, f.repo_name, f.file_path, f.head_commit_sha,
                c.start_line, c.end_line, c.chunk_type, c.name, c.class_name,
                c.symbol_qualified_name, c.content, c.docstring, c.signature, c.concepts
            FROM code_chunks c
            JOIN code_files f ON c.file_id = f.file_id
            WHERE c.name ILIKE %s OR c.symbol_qualified_name ILIKE %s
            LIMIT 50
        """
        cur.execute(sql, (f'%{symbol}%', f'%{symbol}%'))

    columns = [desc[0] for desc in cur.description]
    results = [dict(zip(columns, row)) for row in cur.fetchall()]
    conn.close()

    return [_dict_to_result({**r, 'score': 1.0, 'match_type': 'symbol'}) for r in results]


def find_by_concept(concept: str, limit: int = 20) -> List[CodeResult]:
    """Find code tagged with a concept."""
    conn = get_db_connection()
    cur = conn.cursor()

    sql = """
        SELECT
            c.chunk_id::text, f.repo_name, f.file_path, f.head_commit_sha,
            c.start_line, c.end_line, c.chunk_type, c.name, c.class_name,
            c.symbol_qualified_name, c.content, c.docstring, c.signature, c.concepts
        FROM code_chunks c
        JOIN code_files f ON c.file_id = f.file_id
        WHERE %s = ANY(c.concepts)
        LIMIT %s
    """
    cur.execute(sql, (concept, limit))

    columns = [desc[0] for desc in cur.description]
    results = [dict(zip(columns, row)) for row in cur.fetchall()]
    conn.close()

    return [_dict_to_result({**r, 'score': 1.0, 'match_type': 'concept'}) for r in results]


def get_repo_summary(repo_name: str) -> Dict[str, Any]:
    """Get summary stats for a repo."""
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            f.repo_name,
            f.repo_url,
            f.default_branch,
            MAX(f.head_commit_sha) as commit_sha,
            COUNT(DISTINCT f.file_id) as file_count,
            COUNT(c.chunk_id) as chunk_count,
            array_agg(DISTINCT f.language) FILTER (WHERE f.language IS NOT NULL) as languages,
            array_agg(DISTINCT unnest) FILTER (WHERE unnest IS NOT NULL) as concepts
        FROM code_files f
        LEFT JOIN code_chunks c ON c.file_id = f.file_id
        LEFT JOIN LATERAL unnest(c.concepts) ON true
        WHERE f.repo_name LIKE %s
        GROUP BY f.repo_name, f.repo_url, f.default_branch
    """, (f'{repo_name}%',))

    row = cur.fetchone()
    conn.close()

    if row:
        return {
            'repo_name': row[0],
            'url': row[1],
            'branch': row[2],
            'commit': row[3],
            'files': row[4],
            'chunks': row[5],
            'languages': row[6] or [],
            'concepts': (row[7] or [])[:20]  # Top 20
        }
    return {}


# CLI interface
if __name__ == '__main__':
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python3 -m lib.code_search 'query' [--repo mahmoodlab/*] [--limit 10]")
        sys.exit(1)

    query = sys.argv[1]
    repo = None
    limit = 10

    for i, arg in enumerate(sys.argv):
        if arg == '--repo' and i + 1 < len(sys.argv):
            repo = sys.argv[i + 1]
        if arg == '--limit' and i + 1 < len(sys.argv):
            limit = int(sys.argv[i + 1])

    results = search_code(query, repo=repo, limit=limit)

    print(f"\n=== {len(results)} results for '{query}' ===\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r.chunk_type}] {r.name}")
        print(f"   {r.short_ref()}")
        print(f"   {r.signature or '(no signature)'}")
        if r.concepts:
            print(f"   Concepts: {', '.join(r.concepts[:5])}")
        print()
