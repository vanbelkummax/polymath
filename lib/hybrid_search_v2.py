#!/usr/bin/env python3
"""
Hybrid Search v2 - Two-Collection Architecture

Supports:
- polymath_bge_m3: Scientific papers with year, concepts, section filtering
- polymath_code_bge_m3: Code chunks with repo, language, chunk_type filtering

Usage:
    from lib.hybrid_search_v2 import HybridSearcherV2, SearchResult

    hs = HybridSearcherV2()

    # Search papers only
    results = hs.search_papers("transformer attention mechanism", n=10)

    # Search code only with filters
    results = hs.search_code(
        "attention pooling",
        n=10,
        repo_filter="mahmoodlab",  # Filter by org or repo
        language="python"
    )

    # Search both and merge (default)
    results = hs.hybrid_search("spatial transcriptomics", n=20)

    # Search with concept filter
    results = hs.search_papers("neural network", concepts=["transformer", "attention"])
"""

import os
import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Dict, Optional, Literal

# Config - import from centralized config
from lib.config import (
    CHROMADB_PATH as CHROMA_PATH, POSTGRES_DSN, NEO4J_URI,
    EMBEDDING_MODEL, RERANKER_MODEL, EMBEDDING_DIM,
    CHROMA_COLLECTION_NAME, PAPERS_COLLECTION, CODE_COLLECTION,
    PAPERS_COLLECTION_LEGACY, CODE_COLLECTION_LEGACY
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _extract_title_from_content(content: str, fallback: str = "Untitled") -> str:
    """
    Extract a meaningful title from content when metadata title is missing or numeric.

    Strategies:
    1. Look for non-numeric markdown header (# Title)
    2. Take first meaningful sentence (< 150 chars, substantive content)
    3. Return truncated first line as fallback
    """
    if not content:
        return fallback

    lines = content.strip().split('\n')

    # Strategy 1: Markdown header (skip if it's just a number)
    for line in lines[:5]:
        line = line.strip()
        if line.startswith('# '):
            title = line[2:].strip()
            # Skip numeric-only headers
            if title and not title.isdigit() and len(title) < 200:
                return title

    # Strategy 2: First meaningful line (skip numeric headers and short fragments)
    for line in lines:
        line = line.strip()
        # Skip empty, numeric-only, or too-short lines
        if not line or line.isdigit() or len(line) < 15:
            continue
        # Skip markdown headers that are numbers
        if line.startswith('#') and line.lstrip('#').strip().isdigit():
            continue
        # Skip lines that look like metadata or URLs
        skip_patterns = ['abstract', 'keywords:', 'doi:', 'copyright', 'www.', 'http',
                        'figure', 'table', '©', 'page ', 'et al']
        if any(skip in line.lower() for skip in skip_patterns):
            continue
        # Good candidate - extract meaningful portion
        if len(line) > 150:
            # Try to find a sentence boundary
            for sep in ['. ', '? ', '! ']:
                if sep in line[:150]:
                    return line[:line.index(sep)+1]
            return line[:147] + "..."
        return line

    # Strategy 3: Truncate whatever we have
    for line in lines:
        line = line.strip()
        if line and not line.isdigit() and not line.startswith('#'):
            if len(line) > 80:
                return line[:77] + "..."
            return line

    return fallback


def _is_placeholder_title(title: str) -> bool:
    """Check if title is a numeric placeholder."""
    if not title:
        return True
    return title.strip().isdigit()


@dataclass
class SearchResult:
    """Unified search result."""
    id: str
    title: str
    content: str
    source: str           # 'papers', 'code', 'lexical', 'graph'
    score: float
    metadata: Dict

    def __repr__(self):
        return f"SearchResult({self.source}, {self.score:.3f}, {self.title[:50]}...)"


def reciprocal_rank_fusion(
    result_lists: List[List[SearchResult]],
    k: int = 60
) -> List[SearchResult]:
    """RRF to merge ranked lists."""
    scores = {}
    results_by_id = {}

    for results in result_lists:
        for rank, result in enumerate(results, 1):
            if result.id not in scores:
                scores[result.id] = 0
                results_by_id[result.id] = result
            scores[result.id] += 1 / (k + rank)

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    merged = []
    for rid in sorted_ids:
        result = results_by_id[rid]
        result.score = scores[rid]
        merged.append(result)

    return merged


class HybridSearcherV2:
    """
    Multi-index searcher with two-collection architecture.

    Collections:
    - polymath_bge_m3: Papers with year, concepts, doi, pmid
    - polymath_code_bge_m3: Code with repo, language, chunk_type, concepts

    Features:
    - Cross-encoder reranking for improved relevance
    - RRF fusion of multiple retrieval methods
    - Title extraction from content for numeric placeholders
    """

    def __init__(self, use_reranker: bool = True):
        self._papers_coll = None
        self._code_coll = None
        self._pg = None
        self._neo4j = None
        self._embedder = None
        self._reranker = None
        self._use_reranker = use_reranker
        self._embed_cache = OrderedDict()
        self._cache_size = 256

    def _get_embedder(self):
        """Lazy-load embedding model with BGE-M3 support.

        BGE-M3 is a 2024 SOTA model with 1024-dim embeddings that supports
        dense, sparse, and multi-vector retrieval. For best performance,
        we use FlagEmbedding when available, falling back to sentence-transformers.
        """
        if self._embedder is None:
            model_name = EMBEDDING_MODEL

            # BGE-M3 requires specific initialization for optimal performance
            if 'bge-m3' in model_name.lower():
                try:
                    from FlagEmbedding import BGEM3FlagModel
                    self._embedder = BGEM3FlagModel(
                        model_name,
                        use_fp16=True,  # GPU optimization
                        device='cuda'
                    )
                    self._use_bge_m3 = True
                    logger.info(f"Loaded BGE-M3 model (FlagEmbedding): {model_name}")
                except ImportError:
                    # Fallback to sentence-transformers
                    from sentence_transformers import SentenceTransformer
                    self._embedder = SentenceTransformer(model_name, device='cuda')
                    self._use_bge_m3 = False
                    logger.warning("FlagEmbedding not installed, using sentence-transformers fallback")
            else:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(model_name, device='cuda')
                self._use_bge_m3 = False
                logger.info(f"Loaded embedding model: {model_name}")

        return self._embedder

    def _get_reranker(self):
        """Lazy-load cross-encoder reranker.

        Uses FlagEmbedding's BGE reranker for best performance with BGE-M3.
        Falls back to sentence-transformers CrossEncoder if unavailable.
        """
        if self._reranker is None and self._use_reranker:
            try:
                # Try FlagEmbedding reranker first (best with BGE-M3)
                if 'bge-reranker' in RERANKER_MODEL.lower():
                    try:
                        from FlagEmbedding import FlagReranker
                        self._reranker = FlagReranker(
                            RERANKER_MODEL,
                            use_fp16=True,
                            device='cuda'
                        )
                        self._use_flag_reranker = True
                        logger.info(f"Loaded FlagEmbedding reranker: {RERANKER_MODEL}")
                    except ImportError:
                        # Fall back to sentence-transformers
                        from sentence_transformers import CrossEncoder
                        self._reranker = CrossEncoder(RERANKER_MODEL, device='cuda')
                        self._use_flag_reranker = False
                        logger.warning("FlagEmbedding not available, using sentence-transformers reranker")
                else:
                    from sentence_transformers import CrossEncoder
                    self._reranker = CrossEncoder(RERANKER_MODEL, device='cuda')
                    self._use_flag_reranker = False
                    logger.info(f"Loaded CrossEncoder reranker: {RERANKER_MODEL}")
            except Exception as e:
                logger.warning(f"Failed to load reranker: {e}")
                self._use_reranker = False
        return self._reranker

    def rerank(
        self,
        query: str,
        results: List['SearchResult'],
        top_k: int = None
    ) -> List['SearchResult']:
        """
        Rerank results using cross-encoder for improved relevance.

        Supports both FlagReranker (BGE) and sentence-transformers CrossEncoder.

        Args:
            query: Original search query
            results: List of SearchResult objects to rerank
            top_k: Number of results to return (default: all)

        Returns:
            Reranked list of SearchResults with updated scores
        """
        if not results:
            return results

        reranker = self._get_reranker()
        if reranker is None:
            return results

        # Prepare passages for reranking
        passages = [r.content[:512] for r in results]  # Truncate for speed

        try:
            # Get reranker scores - API differs between FlagReranker and CrossEncoder
            if getattr(self, '_use_flag_reranker', False):
                # FlagReranker uses compute_score
                scores = reranker.compute_score([[query, p] for p in passages])
                if not isinstance(scores, list):
                    scores = [scores]  # Handle single result
            else:
                # CrossEncoder uses predict with pairs
                pairs = [(query, p) for p in passages]
                scores = reranker.predict(pairs, show_progress_bar=False)

            # Update results with new scores
            for i, result in enumerate(results):
                result.score = float(scores[i])

            # Sort by new scores
            results.sort(key=lambda x: x.score, reverse=True)

            if top_k:
                results = results[:top_k]

            logger.debug(f"Reranked {len(results)} results")
            return results

        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
            return results

    def _encode(self, query: str):
        """Encode text to embedding vector.

        Handles both BGE-M3 (FlagEmbedding) and standard sentence-transformers.
        """
        if query in self._embed_cache:
            return self._embed_cache[query]

        embedder = self._get_embedder()

        if getattr(self, '_use_bge_m3', False):
            # BGE-M3 returns dict with 'dense_vecs'
            result = embedder.encode([query], return_dense=True)
            emb = result['dense_vecs'][0].tolist()
        else:
            # Standard sentence-transformers
            emb = embedder.encode([query]).tolist()

        self._embed_cache[query] = emb
        if len(self._embed_cache) > self._cache_size:
            self._embed_cache.popitem(last=False)
        return emb

    def _get_papers_collection(self):
        if self._papers_coll is None:
            import chromadb
            client = chromadb.PersistentClient(path=CHROMA_PATH)
            try:
                # Use ONLY the BGE-M3 collection (1024-dim) - NO FALLBACK
                # Fallback is dangerous: querying 768-dim legacy with 1024-dim vectors crashes
                self._papers_coll = client.get_collection(PAPERS_COLLECTION)
                logger.debug(f"Using papers collection: {PAPERS_COLLECTION}")
            except Exception:
                logger.warning(
                    f"⚠️  Papers collection '{PAPERS_COLLECTION}' not found.\n"
                    f"   Run: python scripts/migrate_knowledge_base.py --apply"
                )
                return None
        return self._papers_coll

    def _get_code_collection(self):
        if self._code_coll is None:
            import chromadb
            client = chromadb.PersistentClient(path=CHROMA_PATH)
            try:
                # Use ONLY the BGE-M3 collection (1024-dim) - NO FALLBACK
                # Fallback is dangerous: querying 768-dim legacy with 1024-dim vectors crashes
                self._code_coll = client.get_collection(CODE_COLLECTION)
                logger.debug(f"Using code collection: {CODE_COLLECTION}")
            except Exception:
                logger.warning(
                    f"⚠️  Code collection '{CODE_COLLECTION}' not found.\n"
                    f"   Run: python scripts/migrate_knowledge_base.py --apply"
                )
                return None
        return self._code_coll

    def _get_postgres(self):
        if self._pg is None:
            try:
                import psycopg2
                self._pg = psycopg2.connect(POSTGRES_DSN)
            except Exception:
                pass
        return self._pg

    def _get_neo4j(self):
        if self._neo4j is None:
            from neo4j import GraphDatabase
            password = os.environ.get("NEO4J_PASSWORD")
            if not password:
                logger.warning("NEO4J_PASSWORD not set - graph search disabled")
                return None
            self._neo4j = GraphDatabase.driver(
                NEO4J_URI,
                auth=("neo4j", password)
            )
        return self._neo4j

    def search_papers(
        self,
        query: str,
        n: int = 20,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        concepts: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """Search papers collection with optional filters."""
        coll = self._get_papers_collection()
        if coll is None:
            return []

        emb = self._encode(query)

        # Build where clause
        where = {}
        where_doc = None

        if year_min and year_max:
            where["$and"] = [
                {"year": {"$gte": year_min}},
                {"year": {"$lte": year_max}}
            ]
        elif year_min:
            where["year"] = {"$gte": year_min}
        elif year_max:
            where["year"] = {"$lte": year_max}

        # Concept filtering via metadata
        if concepts:
            # Use $contains on the concepts string
            concept_filters = [{"concepts": {"$contains": c}} for c in concepts]
            if len(concept_filters) == 1:
                if where:
                    where = {"$and": [where, concept_filters[0]]}
                else:
                    where = concept_filters[0]
            else:
                if where:
                    where = {"$and": [where] + concept_filters}
                else:
                    where = {"$and": concept_filters}

        try:
            results = coll.query(
                query_embeddings=emb,
                n_results=n,
                where=where if where else None,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            logger.error(f"Papers search error: {e}")
            return []

        search_results = []
        for i, doc_id in enumerate(results['ids'][0]):
            meta = results['metadatas'][0][i]
            dist = results['distances'][0][i] if results.get('distances') else 0.5
            content = results['documents'][0][i][:500]

            # Fix placeholder titles by extracting from content
            raw_title = meta.get('title', '')
            if _is_placeholder_title(raw_title):
                title = _extract_title_from_content(content, fallback=f"Paper {doc_id[:8]}")
            else:
                title = raw_title

            search_results.append(SearchResult(
                id=doc_id,
                title=title,
                content=content,
                source='papers',
                score=1 - dist,  # Convert distance to similarity
                metadata=meta
            ))
        return search_results

    def search_code(
        self,
        query: str,
        n: int = 20,
        repo_filter: Optional[str] = None,
        org_filter: Optional[str] = None,
        language: Optional[str] = None,
        chunk_type: Optional[str] = None,
        concepts: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """Search code collection with optional filters."""
        coll = self._get_code_collection()
        if coll is None:
            return []

        emb = self._encode(query)

        # Build where clause
        filters = []

        if repo_filter:
            filters.append({"repo_name": {"$contains": repo_filter}})
        if org_filter:
            filters.append({"org": {"$eq": org_filter}})
        if language:
            filters.append({"language": {"$eq": language}})
        if chunk_type:
            filters.append({"chunk_type": {"$eq": chunk_type}})
        if concepts:
            for c in concepts:
                filters.append({"concepts": {"$contains": c}})

        where = None
        if len(filters) == 1:
            where = filters[0]
        elif len(filters) > 1:
            where = {"$and": filters}

        try:
            results = coll.query(
                query_embeddings=emb,
                n_results=n,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            logger.error(f"Code search error: {e}")
            return []

        search_results = []
        for i, doc_id in enumerate(results['ids'][0]):
            meta = results['metadatas'][0][i]
            dist = results['distances'][0][i] if results.get('distances') else 0.5
            search_results.append(SearchResult(
                id=doc_id,
                title=f"{meta.get('repo_name', '?')}/{meta.get('name', '?')}",
                content=results['documents'][0][i][:500],
                source='code',
                score=1 - dist,
                metadata=meta
            ))
        return search_results

    def lexical_search_passages(self, query: str, n: int = 20) -> List[SearchResult]:
        """Full-text search on passages table."""
        pg = self._get_postgres()
        if pg is None:
            return []

        cursor = pg.cursor()
        try:
            cursor.execute("""
                SELECT p.passage_id::text, d.title, p.passage_text,
                       ts_rank(to_tsvector('english', p.passage_text), plainto_tsquery('english', %s)) as score
                FROM passages p
                JOIN documents d ON p.doc_id = d.doc_id
                WHERE to_tsvector('english', p.passage_text) @@ plainto_tsquery('english', %s)
                ORDER BY score DESC
                LIMIT %s
            """, (query, query, n))

            results = []
            for row in cursor.fetchall():
                results.append(SearchResult(
                    id=row[0],
                    title=row[1] or 'Unknown',
                    content=row[2][:500],
                    source='lexical',
                    score=float(row[3]),
                    metadata={'source': 'passages'}
                ))
            return results
        except Exception as e:
            logger.error(f"Lexical search error: {e}")
            return []

    def lexical_search_code(self, query: str, n: int = 20) -> List[SearchResult]:
        """Full-text search on code_chunks table."""
        pg = self._get_postgres()
        if pg is None:
            return []

        cursor = pg.cursor()
        try:
            cursor.execute("""
                SELECT cc.chunk_id::text, cf.repo_name || '/' || cc.name, cc.content,
                       ts_rank(cc.search_vector, plainto_tsquery('english', %s)) as score
                FROM code_chunks cc
                JOIN code_files cf ON cc.file_id = cf.file_id
                WHERE cc.search_vector @@ plainto_tsquery('english', %s)
                ORDER BY score DESC
                LIMIT %s
            """, (query, query, n))

            results = []
            for row in cursor.fetchall():
                results.append(SearchResult(
                    id=row[0],
                    title=row[1] or 'Unknown',
                    content=row[2][:500],
                    source='lexical_code',
                    score=float(row[3]),
                    metadata={'source': 'code_chunks'}
                ))
            return results
        except Exception as e:
            logger.error(f"Code lexical search error: {e}")
            return []

    def hybrid_search(
        self,
        query: str,
        n: int = 20,
        search_papers: bool = True,
        search_code: bool = True,
        use_lexical: bool = True,
        rerank: bool = True,
        rerank_top_k: int = 50,
        **kwargs
    ) -> List[SearchResult]:
        """
        Full hybrid search across all sources.

        Uses RRF to merge:
        - Vector search on papers (if search_papers)
        - Vector search on code (if search_code)
        - Lexical search on passages (if use_lexical)
        - Lexical search on code_chunks (if use_lexical)

        Optionally reranks top candidates with cross-encoder for better relevance.

        Args:
            query: Search query
            n: Number of results to return
            search_papers: Include paper search
            search_code: Include code search
            use_lexical: Include lexical (BM25) search
            rerank: Apply cross-encoder reranking (default: True)
            rerank_top_k: Number of candidates to rerank before returning n
        """
        result_lists = []

        # Fetch more candidates if reranking
        fetch_n = max(n, rerank_top_k) if rerank and self._use_reranker else n

        if search_papers:
            paper_results = self.search_papers(query, n=fetch_n, **{k: v for k, v in kwargs.items() if k in ['year_min', 'year_max', 'concepts']})
            if paper_results:
                result_lists.append(paper_results)

        if search_code:
            code_results = self.search_code(query, n=fetch_n, **{k: v for k, v in kwargs.items() if k in ['repo_filter', 'org_filter', 'language', 'chunk_type', 'concepts']})
            if code_results:
                result_lists.append(code_results)

        if use_lexical:
            if search_papers:
                lex_papers = self.lexical_search_passages(query, n=fetch_n)
                if lex_papers:
                    result_lists.append(lex_papers)
            if search_code:
                lex_code = self.lexical_search_code(query, n=fetch_n)
                if lex_code:
                    result_lists.append(lex_code)

        if not result_lists:
            return []

        # Merge with RRF
        merged = reciprocal_rank_fusion(result_lists)[:rerank_top_k if rerank else n]

        # Optional cross-encoder reranking
        if rerank and self._use_reranker and len(merged) > 1:
            merged = self.rerank(query, merged, top_k=n)
        else:
            merged = merged[:n]

        return merged

    def query_passage_concepts(
        self,
        concept: str,
        n: int = 20,
        min_confidence: float = 0.7
    ) -> List[SearchResult]:
        """SQL search for passages by exact concept match."""
        pg = self._get_postgres()
        if pg is None:
            return []

        cursor = pg.cursor()
        try:
            cursor.execute("""
                SELECT DISTINCT ON (pc.passage_id)
                    pc.passage_id::text,
                    d.title,
                    LEFT(p.passage_text, 500),
                    pc.confidence,
                    pc.evidence->>'source_text' as evidence_text
                FROM passage_concepts pc
                JOIN passages p ON p.passage_id = pc.passage_id
                JOIN documents d ON p.doc_id = d.doc_id
                WHERE pc.concept_name ILIKE %s
                  AND pc.confidence >= %s
                ORDER BY pc.passage_id, pc.confidence DESC
                LIMIT %s
            """, (f"%{concept}%", min_confidence, n))

            results = []
            for row in cursor.fetchall():
                results.append(SearchResult(
                    id=row[0],
                    title=row[1] or 'Unknown',
                    content=row[2],
                    source='sql_concept',
                    score=float(row[3]) if row[3] else 0.8,
                    metadata={
                        'concept': concept,
                        'confidence': row[3],
                        'evidence': row[4]
                    }
                ))
            return results
        except Exception as e:
            logger.error(f"Concept SQL search error: {e}")
            return []

    def graph_expand_concept(
        self,
        concept: str,
        n_related: int = 10
    ) -> List[str]:
        """Get related concepts from Neo4j graph."""
        driver = self._get_neo4j()
        if driver is None:
            return []

        try:
            with driver.session() as session:
                result = session.run("""
                    MATCH (c:CONCEPT)-[r:RELATES_TO|CO_OCCURS]-(related:CONCEPT)
                    WHERE toLower(c.name) CONTAINS toLower($concept)
                    RETURN DISTINCT related.name as name, r.weight as weight
                    ORDER BY r.weight DESC
                    LIMIT $limit
                """, concept=concept, limit=n_related)

                return [record['name'] for record in result]
        except Exception as e:
            logger.debug(f"Graph expansion not available: {e}")
            return []

    def atlas_search(
        self,
        query: str,
        n: int = 20,
        min_confidence: float = 0.7,
        include_graph: bool = True
    ) -> Dict:
        """
        Tri-modal search with explainability trace.

        Combines:
        1. SQL concept search - Direct passage_concepts query
        2. Vector search - Semantic embedding similarity
        3. Graph expansion - Related concepts from Neo4j

        Returns dict with results and explain trace showing each channel's contribution.

        Args:
            query: Search query (can be concept name or natural language)
            n: Number of results to return
            min_confidence: Minimum confidence for SQL concept matches
            include_graph: Whether to expand via graph (may be slow)

        Returns:
            {
                "results": List[SearchResult],
                "explain": {
                    "sql": {"matched_concepts": [], "passages": [], "count": int},
                    "vector": {"top_neighbors": [], "scores": [], "count": int},
                    "graph": {"expanded_concepts": [], "paths_found": int}
                }
            }
        """
        explain = {
            "sql": {"matched_concepts": [], "passages": [], "count": 0},
            "vector": {"top_neighbors": [], "scores": [], "count": 0},
            "graph": {"expanded_concepts": [], "paths_found": 0}
        }

        result_lists = []

        # 1. SQL Concept Search - Extract concept terms from query
        query_terms = query.lower().replace('-', '_').split()
        sql_results = []

        for term in query_terms:
            if len(term) > 3:  # Skip short words
                term_results = self.query_passage_concepts(term, n=n, min_confidence=min_confidence)
                if term_results:
                    sql_results.extend(term_results)
                    explain["sql"]["matched_concepts"].append(term)

        # Deduplicate by passage_id
        seen_ids = set()
        unique_sql = []
        for r in sql_results:
            if r.id not in seen_ids:
                seen_ids.add(r.id)
                unique_sql.append(r)

        if unique_sql:
            result_lists.append(unique_sql[:n])
            explain["sql"]["count"] = len(unique_sql)
            explain["sql"]["passages"] = [r.id for r in unique_sql[:5]]

        # 2. Vector Search - Semantic similarity
        vector_results = self.search_papers(query, n=n)
        if vector_results:
            result_lists.append(vector_results)
            explain["vector"]["count"] = len(vector_results)
            explain["vector"]["top_neighbors"] = [r.title[:50] for r in vector_results[:5]]
            explain["vector"]["scores"] = [round(r.score, 3) for r in vector_results[:5]]

        # 3. Graph Expansion - Related concepts
        if include_graph:
            expanded_concepts = []
            for term in query_terms:
                if len(term) > 3:
                    related = self.graph_expand_concept(term, n_related=5)
                    expanded_concepts.extend(related)

            explain["graph"]["expanded_concepts"] = list(set(expanded_concepts))[:10]

            # Search for expanded concepts
            graph_results = []
            for exp_concept in expanded_concepts[:3]:  # Limit to top 3 expansions
                exp_results = self.query_passage_concepts(exp_concept, n=5, min_confidence=min_confidence)
                graph_results.extend(exp_results)

            # Deduplicate
            for r in graph_results:
                if r.id not in seen_ids:
                    seen_ids.add(r.id)
                    unique_sql.append(r)

            if graph_results:
                result_lists.append(graph_results[:n])
                explain["graph"]["paths_found"] = len(graph_results)

        # 4. Merge with RRF
        if not result_lists:
            return {"results": [], "explain": explain}

        merged = reciprocal_rank_fusion(result_lists)[:n]

        return {
            "results": merged,
            "explain": explain
        }


# Convenience functions
def search(query: str, n: int = 20, **kwargs) -> List[SearchResult]:
    """Quick search across all collections."""
    hs = HybridSearcherV2()
    return hs.hybrid_search(query, n=n, **kwargs)


def search_papers(query: str, n: int = 20, **kwargs) -> List[SearchResult]:
    """Quick search papers only."""
    hs = HybridSearcherV2()
    return hs.search_papers(query, n=n, **kwargs)


def search_code(query: str, n: int = 20, **kwargs) -> List[SearchResult]:
    """Quick search code only."""
    hs = HybridSearcherV2()
    return hs.search_code(query, n=n, **kwargs)


if __name__ == "__main__":
    # Test
    hs = HybridSearcherV2()
    print("Testing hybrid search v2...")

    print("\n=== Papers Search ===")
    results = hs.search_papers("transformer attention mechanism", n=5)
    for r in results:
        print(f"  [{r.score:.3f}] {r.title[:60]}")

    print("\n=== Code Search (mahmoodlab) ===")
    results = hs.search_code("attention pooling", n=5, org_filter="mahmoodlab")
    for r in results:
        print(f"  [{r.score:.3f}] {r.title}")

    print("\n=== Hybrid Search ===")
    results = hs.hybrid_search("spatial transcriptomics gene expression", n=5)
    for r in results:
        print(f"  [{r.score:.3f}] [{r.source}] {r.title[:50]}")
