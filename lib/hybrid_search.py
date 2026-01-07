#!/usr/bin/env python3
"""
Hybrid Search: Combines Postgres FTS + ChromaDB vectors + Neo4j graph.
Implements Reciprocal Rank Fusion (RRF) for result merging.
"""

import os
import logging
from collections import OrderedDict
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import asyncio

# Config
CHROMA_PATH = os.environ.get("CHROMA_PATH", "/home/user/work/polymax/chromadb/polymath_v2")
POSTGRES_URL = os.environ.get("POSTGRES_URL", "dbname=polymath user=polymath host=/var/run/postgresql")
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
EMBEDDING_MODEL = "all-mpnet-base-v2"  # 768-dim, matches ChromaDB collection

logger = logging.getLogger(__name__)

# Import monitoring decorator
try:
    import sys
    sys.path.insert(0, '/home/user/work/polymax')
    from monitoring.metrics_collector import track_query
    MONITORING_ENABLED = True
except ImportError:
    # Monitoring not available, create no-op decorator
    def track_query(operation):
        def decorator(func):
            return func
        return decorator
    MONITORING_ENABLED = False

# Rosetta Stone query expansion
try:
    import sys
    sys.path.insert(0, '/home/user/work/polymax/lib')
    from rosetta_query_expander import expand_query_with_llm
    ROSETTA_AVAILABLE = True
except Exception:
    ROSETTA_AVAILABLE = False
    expand_query_with_llm = None


@dataclass
class SearchResult:
    """Unified search result from any source."""
    id: str
    title: str
    content: str
    source: str           # 'vector', 'lexical', 'graph'
    score: float
    metadata: Dict


def reciprocal_rank_fusion(
    result_lists: List[List[SearchResult]],
    k: int = 60
) -> List[SearchResult]:
    """
    Reciprocal Rank Fusion (RRF) to merge ranked lists.
    RRF score = sum(1 / (k + rank_i)) across all lists where item appears.
    """
    scores = {}
    results_by_id = {}
    
    for results in result_lists:
        for rank, result in enumerate(results, 1):
            if result.id not in scores:
                scores[result.id] = 0
                results_by_id[result.id] = result
            scores[result.id] += 1 / (k + rank)
    
    # Update scores and sort
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    
    merged = []
    for rid in sorted_ids:
        result = results_by_id[rid]
        result.score = scores[rid]
        merged.append(result)
    
    return merged


class HybridSearcher:
    """
    Multi-index searcher combining:
    - ChromaDB: Dense vector similarity
    - Postgres: BM25/FTS exact token match
    - Neo4j: Graph-based concept expansion
    """
    
    def __init__(self):
        self._chroma = None
        self._pg = None
        self._neo4j = None
        self._embedder = None
        self._embedding_cache = OrderedDict()
        self._embedding_cache_size = int(os.environ.get("EMBED_CACHE_SIZE", "256"))

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(EMBEDDING_MODEL)
        return self._embedder

    def _encode_query(self, query: str):
        if self._embedding_cache_size <= 0:
            return self._get_embedder().encode([query]).tolist()

        cached = self._embedding_cache.get(query)
        if cached is not None:
            self._embedding_cache.move_to_end(query)
            return cached

        embedding = self._get_embedder().encode([query]).tolist()
        self._embedding_cache[query] = embedding
        if len(self._embedding_cache) > self._embedding_cache_size:
            self._embedding_cache.popitem(last=False)
        return embedding
    
    def _get_chroma(self):
        if self._chroma is None:
            import chromadb
            client = chromadb.PersistentClient(path=CHROMA_PATH)
            self._chroma = client.get_collection("polymath_corpus")
        return self._chroma
    
    def _get_postgres(self):
        if self._pg is None:
            try:
                import psycopg2
                # Use explicit params to avoid env var format issues
                self._pg = psycopg2.connect(
                    dbname='polymath',
                    user='polymath',
                    host='/var/run/postgresql'
                )
            except Exception as e:
                logger.debug(f"Postgres connection failed: {e}")
                pass  # Postgres not configured yet
        return self._pg
    
    def _get_neo4j(self):
        if self._neo4j is None:
            from neo4j import GraphDatabase
            self._neo4j = GraphDatabase.driver(
                NEO4J_URI,
                auth=("neo4j", os.environ.get("NEO4J_PASSWORD", "polymathic2026"))
            )
        return self._neo4j

    def rosetta_expand(self, query: str, max_terms: int = 6) -> List[str]:
        """Expand query with Rosetta Stone if available."""
        if not ROSETTA_AVAILABLE or not expand_query_with_llm:
            return []
        try:
            terms = expand_query_with_llm(query)
        except Exception:
            return []
        return terms[:max_terms] if terms else []
    
    @track_query('vector_search')
    def vector_search(self, query: str, n: int = 20) -> List[SearchResult]:
        """Dense vector search via ChromaDB."""
        coll = self._get_chroma()
        query_embedding = self._encode_query(query)
        results = coll.query(query_embeddings=query_embedding, n_results=n)

        search_results = []
        for i, doc_id in enumerate(results['ids'][0]):
            search_results.append(SearchResult(
                id=doc_id,
                title=results['metadatas'][0][i].get('title', 'Unknown'),
                content=results['documents'][0][i][:500],
                source='vector',
                score=1 - results['distances'][0][i] if results.get('distances') else 0.9 - i*0.01,
                metadata=results['metadatas'][0][i]
            ))
        return search_results
    
    @track_query('lexical_search')
    def lexical_search(self, query: str, n: int = 20) -> List[SearchResult]:
        """BM25/FTS search via Postgres."""
        pg = self._get_postgres()
        if pg is None:
            return []  # Postgres not available

        cursor = pg.cursor()
        cursor.execute("""
            SELECT c.id, COALESCE(a.title, 'Unknown'), c.content,
                   ts_rank(c.content_tsv, plainto_tsquery('english', %s)) as score
            FROM chunks c
            LEFT JOIN artifacts a ON c.artifact_id = a.id
            WHERE c.content_tsv @@ plainto_tsquery('english', %s)
            ORDER BY score DESC
            LIMIT %s
        """, (query, query, n))

        results = []
        for row in cursor.fetchall():
            results.append(SearchResult(
                id=row[0],
                title=row[1],
                content=row[2][:500],
                source='lexical',
                score=float(row[3]),
                metadata={}
            ))
        return results
    
    def graph_expand(self, search_query: str, n: int = 10) -> List[str]:
        """Find related concepts via Neo4j for query expansion."""
        driver = self._get_neo4j()

        # Find concepts matching query terms
        with driver.session() as session:
            result = session.run("""
                MATCH (c:CONCEPT)
                WHERE toLower(c.name) CONTAINS toLower($search_term)
                OPTIONAL MATCH (c)-[:RELATES_TO]-(related:CONCEPT)
                RETURN c.name as concept, collect(related.name) as related
                LIMIT 5
            """, search_term=search_query)
            
            expansions = []
            for record in result:
                expansions.append(record['concept'])
                expansions.extend(record['related'][:3])
            
            return list(set(expansions))[:n]
    
    @track_query('hybrid_search')
    def hybrid_search(
        self,
        query: str,
        n_results: int = 10,
        use_graph_expansion: bool = True,
        use_rosetta: bool = False,
        vector_weight: float = 0.6,
        lexical_weight: float = 0.4
    ) -> List[SearchResult]:
        """
        Execute hybrid search across all indexes.

        Strategy:
        1. Run vector + lexical in parallel
        2. Optionally expand query via graph
        3. Fuse results with RRF
        """
        result_lists = []
        if not use_rosetta:
            use_rosetta = os.environ.get("ROSETTA_ENABLED", "0") == "1"

        # Core searches
        vector_results = self.vector_search(query, n=n_results * 2)
        result_lists.append(vector_results)

        lexical_results = self.lexical_search(query, n=n_results * 2)
        if lexical_results:
            result_lists.append(lexical_results)

        # Graph expansion
        if use_graph_expansion:
            expansions = self.graph_expand(query)
            if expansions:
                # Search expanded terms
                expanded_query = " ".join(expansions[:3])
                expanded_results = self.vector_search(expanded_query, n=n_results)
                if expanded_results:
                    # Weight expanded results lower
                    for r in expanded_results:
                        r.score *= 0.5
                    result_lists.append(expanded_results)

        # Rosetta Stone expansion
        if use_rosetta:
            rosetta_terms = self.rosetta_expand(query)
            if rosetta_terms:
                expanded_query = f"{query} " + " ".join(rosetta_terms)
                expanded_results = self.vector_search(expanded_query, n=n_results)
                if expanded_results:
                    for r in expanded_results:
                        r.score *= 0.4
                    result_lists.append(expanded_results)

        # Fuse with RRF
        merged = reciprocal_rank_fusion(result_lists)

        return merged[:n_results]


# CLI interface
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python hybrid_search.py 'your query'")
        sys.exit(1)
    
    query = " ".join(sys.argv[1:])
    searcher = HybridSearcher()
    
    print(f"Hybrid search: '{query}'")
    print("=" * 60)
    
    results = searcher.hybrid_search(query)
    for i, r in enumerate(results, 1):
        print(f"\n{i}. [{r.source}] {r.title}")
        print(f"   Score: {r.score:.4f}")
        print(f"   {r.content[:200]}...")
