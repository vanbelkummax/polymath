#!/usr/bin/env python3
"""
Polymath Tools for Codex CLI

Simple function wrappers that Codex can call directly.
No MCP required - just import and use.

Usage:
    from codex_tools import search_papers, search_code, ingest_pdf
"""

import sys
from pathlib import Path

# Ensure lib is importable
sys.path.insert(0, str(Path(__file__).parent))

from typing import List, Dict, Optional


def search_papers(
    query: str,
    n: int = 10,
    year_min: int = None,
    year_max: int = None,
    concepts: List[str] = None,
    rerank: bool = True
) -> List[Dict]:
    """
    Search indexed papers with semantic + lexical hybrid search.

    Args:
        query: Search query
        n: Number of results
        year_min: Filter papers from this year
        year_max: Filter papers until this year
        concepts: Filter by concept tags
        rerank: Use cross-encoder reranking

    Returns:
        List of paper results with title, year, snippet, score
    """
    from lib.hybrid_search_v2 import HybridSearcherV2
    hs = HybridSearcherV2()
    return hs.search_papers(
        query, n=n, year_min=year_min, year_max=year_max,
        concepts=concepts, rerank=rerank
    )


def search_code(
    query: str,
    n: int = 10,
    org_filter: str = None,
    repo_filter: str = None,
    language: str = None,
    chunk_type: str = None
) -> List[Dict]:
    """
    Search indexed code repositories.

    Args:
        query: Search query
        n: Number of results
        org_filter: Filter by GitHub org (e.g., "mahmoodlab")
        repo_filter: Filter by repo name
        language: Filter by language (python, r, javascript, etc.)
        chunk_type: Filter by type (function, class, module)

    Returns:
        List of code results with repo, file, snippet, score
    """
    from lib.hybrid_search_v2 import HybridSearcherV2
    hs = HybridSearcherV2()
    return hs.search_code(
        query, n=n, org_filter=org_filter, repo_filter=repo_filter,
        language=language, chunk_type=chunk_type
    )


def hybrid_search(query: str, n: int = 20, rerank: bool = True) -> List[Dict]:
    """
    Search across papers + code + graph with RRF fusion.

    Args:
        query: Search query
        n: Number of results
        rerank: Use cross-encoder reranking

    Returns:
        Combined results from all sources
    """
    from lib.hybrid_search_v2 import HybridSearcherV2
    hs = HybridSearcherV2()
    return hs.hybrid_search(query, n=n, rerank=rerank)


def graph_query(cypher: str) -> List[Dict]:
    """
    Run a Cypher query against the Neo4j knowledge graph.

    Args:
        cypher: Cypher query string

    Returns:
        Query results as list of dicts

    Example:
        graph_query("MATCH (c:CONCEPT) RETURN c.name, c.count ORDER BY c.count DESC LIMIT 10")
    """
    import os
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", os.environ.get("NEO4J_PASSWORD", "polymathic2026"))
    )

    with driver.session() as session:
        result = session.run(cypher)
        return [dict(record) for record in result]


def ingest_pdf(path: str) -> Dict:
    """
    Ingest a PDF into the knowledge base.

    Args:
        path: Path to PDF file

    Returns:
        Ingestion result with doc_id, chunks_added, concepts
    """
    from lib.unified_ingest import TransactionalIngestor
    ingestor = TransactionalIngestor()
    result = ingestor.ingest_pdf(path)
    return {
        "doc_id": str(result.doc_id) if result.doc_id else None,
        "chunks_added": result.chunks_added,
        "concepts": result.concepts_linked,
        "errors": result.errors
    }


def ingest_code(path: str, repo_name: str = None) -> Dict:
    """
    Ingest a code file into the knowledge base.

    Args:
        path: Path to code file
        repo_name: Optional repository name

    Returns:
        Ingestion result with chunks_added, concepts
    """
    from lib.unified_ingest import UnifiedIngestor
    ingestor = UnifiedIngestor()
    result = ingestor.ingest_code(path, repo_name=repo_name)
    return {
        "chunks_added": result.chunks_added if result else 0,
        "concepts": result.concepts_linked if result else [],
    }


def find_evidence(claim: str, top_k: int = 30) -> List[Dict]:
    """
    Find evidence spans that support a claim using NLI.

    Args:
        claim: The claim to find evidence for
        top_k: Number of candidate passages to consider

    Returns:
        List of evidence spans with entailment scores
    """
    from lib.evidence_extractor import EvidenceExtractor
    extractor = EvidenceExtractor()
    spans = extractor.extract_spans_for_claim(claim, top_k_passages=top_k)
    return [
        {
            "text": s.span_text,
            "page": s.page_num,
            "entailment": s.entailment_score,
            "doc_id": str(s.doc_id)
        }
        for s in spans
    ]


def get_stats() -> Dict:
    """Get current system statistics."""
    from lib.db import Database
    from lib.config import POSTGRES_DSN

    db = Database(dsn=POSTGRES_DSN)
    stats = {}

    for table in ["documents", "passages", "code_files", "code_chunks", "concepts"]:
        try:
            row = db.fetch_one(f"SELECT count(*) as count FROM {table}")
            stats[table] = row["count"] if row else 0
        except:
            stats[table] = "error"

    db.close()
    return stats


def discover_openalex(query: str, max_results: int = 20) -> List[Dict]:
    """
    Discover papers from OpenAlex (250M+ works).

    Args:
        query: Search query
        max_results: Maximum results

    Returns:
        List of papers with title, authors, year, doi, citations
    """
    from lib.sentry.sources.openalex import OpenAlexSource
    oa = OpenAlexSource()
    return oa.discover(query, max_results=max_results)


# CLI interface
if __name__ == "__main__":
    import json
    import argparse

    parser = argparse.ArgumentParser(description="Polymath Tools for Codex")
    parser.add_argument("command", choices=[
        "search", "code", "hybrid", "graph", "stats", "discover"
    ])
    parser.add_argument("query", nargs="?", default="")
    parser.add_argument("-n", type=int, default=10)
    parser.add_argument("--year-min", type=int)
    parser.add_argument("--org", type=str)

    args = parser.parse_args()

    if args.command == "search":
        results = search_papers(args.query, n=args.n, year_min=args.year_min)
    elif args.command == "code":
        results = search_code(args.query, n=args.n, org_filter=args.org)
    elif args.command == "hybrid":
        results = hybrid_search(args.query, n=args.n)
    elif args.command == "graph":
        results = graph_query(args.query)
    elif args.command == "stats":
        results = get_stats()
    elif args.command == "discover":
        results = discover_openalex(args.query, max_results=args.n)

    print(json.dumps(results, indent=2, default=str))
