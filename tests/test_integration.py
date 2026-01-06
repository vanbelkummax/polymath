#!/usr/bin/env python3
"""
Integration Tests for Polymath 11.0

Tests core functionality across:
- Database connections (Postgres, ChromaDB, Neo4j)
- Hybrid search (vector, lexical, reranking)
- MCP v11 tools (discovery, reasoning)
- Literature Sentry sources
- Evidence extraction

Run with: pytest tests/test_integration.py -v
"""

import pytest
import os
import sys
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))


# =============================================================================
# Database Connection Tests
# =============================================================================

class TestDatabaseConnections:
    """Test database connectivity and basic operations."""

    def test_postgres_connection(self):
        """Test Postgres connection and basic query."""
        import psycopg2
        conn = psycopg2.connect(
            dbname='polymath',
            user='polymath',
            host='/var/run/postgresql'
        )
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents")
        count = cursor.fetchone()[0]
        conn.close()

        assert count > 0, "No documents in database"
        print(f"Postgres: {count} documents")

    def test_chromadb_connection(self):
        """Test ChromaDB connection and collections."""
        import chromadb
        client = chromadb.PersistentClient(path="/home/user/work/polymax/chromadb")
        collections = client.list_collections()

        collection_names = [c.name for c in collections]
        assert "polymath_papers" in collection_names, "polymath_papers collection not found"
        assert "polymath_code" in collection_names, "polymath_code collection not found"

        papers = client.get_collection("polymath_papers")
        code = client.get_collection("polymath_code")
        print(f"ChromaDB: {papers.count()} papers, {code.count()} code chunks")

        assert papers.count() > 0, "No papers in ChromaDB"
        assert code.count() > 0, "No code in ChromaDB"

    def test_neo4j_connection(self):
        """Test Neo4j connection (requires NEO4J_PASSWORD env var)."""
        password = os.environ.get("NEO4J_PASSWORD")
        if not password:
            pytest.skip("NEO4J_PASSWORD not set")

        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", password)
        )

        with driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) as cnt")
            count = result.single()["cnt"]

        driver.close()
        print(f"Neo4j: {count} nodes")
        assert count > 0, "No nodes in Neo4j"


# =============================================================================
# Hybrid Search Tests
# =============================================================================

class TestHybridSearch:
    """Test hybrid search functionality."""

    @pytest.fixture
    def searcher(self):
        from hybrid_search_v2 import HybridSearcherV2
        return HybridSearcherV2(use_reranker=False)  # Faster without reranker

    def test_paper_search(self, searcher):
        """Test paper search returns results."""
        results = searcher.search_papers("machine learning", n=5)
        assert len(results) > 0, "No paper search results"
        assert all(r.source == "papers" for r in results)
        print(f"Paper search: {len(results)} results")

    def test_code_search(self, searcher):
        """Test code search returns results."""
        results = searcher.search_code("attention", n=5)
        assert len(results) > 0, "No code search results"
        assert all(r.source == "code" for r in results)
        print(f"Code search: {len(results)} results")

    def test_hybrid_search(self, searcher):
        """Test hybrid search merges multiple sources."""
        results = searcher.hybrid_search("neural network", n=10, rerank=False)
        assert len(results) > 0, "No hybrid search results"

        sources = set(r.source for r in results)
        print(f"Hybrid search: {len(results)} results from {sources}")

    def test_title_extraction(self, searcher):
        """Test that numeric titles are replaced with extracted content."""
        results = searcher.search_papers("cancer", n=10)

        # Check that no results have purely numeric titles
        numeric_titles = [r for r in results if r.title.strip().isdigit()]
        assert len(numeric_titles) == 0, f"Found {len(numeric_titles)} numeric titles"

    def test_search_with_year_filter(self, searcher):
        """Test year filtering works."""
        results = searcher.search_papers("transformer", n=10, year_min=2023)

        for r in results:
            year = r.metadata.get("year")
            if year and year > 0:
                assert year >= 2023, f"Found paper from {year}, expected >= 2023"


# =============================================================================
# MCP v11 Tool Tests
# =============================================================================

class TestMCPv11Tools:
    """Test MCP v11 tool implementations."""

    @pytest.fixture
    def discovery_tools(self):
        sys.path.insert(0, str(Path(__file__).parent.parent / "mcp" / "polymath_v11"))
        from tools.discovery import DiscoveryTools
        return DiscoveryTools()

    @pytest.fixture
    def reasoning_tools(self):
        sys.path.insert(0, str(Path(__file__).parent.parent / "mcp" / "polymath_v11"))
        from tools.reasoning import ReasoningTools
        return ReasoningTools()

    @pytest.mark.asyncio
    async def test_deep_hunt(self, discovery_tools):
        """Test deep_hunt tool returns papers and code."""
        result = await discovery_tools.deep_hunt("spatial transcriptomics", max_results=5)

        assert "papers" in result
        assert "code" in result
        assert result["total_papers"] > 0 or result["total_code"] > 0
        print(f"deep_hunt: {result['total_papers']} papers, {result['total_code']} code")

    @pytest.mark.asyncio
    async def test_find_gaps(self, discovery_tools):
        """Test find_gaps tool identifies research gaps."""
        result = await discovery_tools.find_gaps("colorectal cancer")

        assert "temporal_analysis" in result
        assert "concept_analysis" in result
        assert "suggested_gaps" in result
        print(f"find_gaps: {len(result['suggested_gaps'])} gaps found")

    @pytest.mark.asyncio
    async def test_collection_health(self, discovery_tools):
        """Test collection_health tool checks database status."""
        result = await discovery_tools.collection_health()

        assert "status" in result
        assert "stats" in result
        assert result["stats"]["total_artifacts"] > 0
        print(f"collection_health: {result['status']}, {result['stats']['total_artifacts']} artifacts")

    @pytest.mark.asyncio
    async def test_generate_hypothesis(self, reasoning_tools):
        """Test hypothesis generation."""
        result = await reasoning_tools.generate_hypothesis("spatial biology", num_hypotheses=3)

        assert "hypotheses" in result
        assert len(result["hypotheses"]) > 0
        print(f"generate_hypothesis: {len(result['hypotheses'])} hypotheses")

    @pytest.mark.asyncio
    async def test_serendipity(self, reasoning_tools):
        """Test serendipity engine finds unexpected connections."""
        result = await reasoning_tools.serendipity()

        assert "serendipity_bridge" in result
        assert "concept_a" in result["serendipity_bridge"]
        assert "concept_b" in result["serendipity_bridge"]
        print(f"serendipity: {result['serendipity_bridge']['concept_a']} <-> {result['serendipity_bridge']['concept_b']}")


# =============================================================================
# Literature Sentry Tests
# =============================================================================

class TestLiteratureSentry:
    """Test Literature Sentry source connectors."""

    def test_arxiv_source(self):
        """Test arXiv source returns papers."""
        from sentry.sources.arxiv import ArxivSource
        source = ArxivSource()

        results = source.discover("transformer attention", max_results=3)
        source.close()

        assert len(results) > 0, "No arXiv results"
        assert all(r["source"] == "arxiv" for r in results)
        print(f"arXiv: {len(results)} results")

    def test_biorxiv_source(self):
        """Test bioRxiv source returns papers."""
        from sentry.sources.biorxiv import BioRxivSource
        source = BioRxivSource()

        results = source.get_recent(days=3, max_results=3)
        source.close()

        assert len(results) > 0, "No bioRxiv results"
        assert all(r["source"] == "biorxiv" for r in results)
        print(f"bioRxiv: {len(results)} results")

    def test_europepmc_source(self):
        """Test Europe PMC source returns papers."""
        from sentry.sources.europepmc import EuropePMCSource
        source = EuropePMCSource()

        results = source.discover("spatial transcriptomics", max_results=3)
        source.close()

        assert len(results) > 0, "No Europe PMC results"
        print(f"Europe PMC: {len(results)} results")

    def test_scoring(self):
        """Test quality scoring."""
        from sentry.scoring import Scorer

        scorer = Scorer()

        # Test high-quality item (using trusted author name)
        high_quality = {
            "title": "Deep learning for spatial transcriptomics",
            "source": "arxiv",
            "citations": 100,
            "authors": ["Faisal Mahmood"],  # Must match TRUSTED_AUTHORS entry
        }
        scored = scorer.score(high_quality, "spatial transcriptomics")
        assert scored.is_trusted_lab, "Should be recognized as trusted lab"
        assert scored.priority_score >= 0.95, "Trusted lab should get 0.95 score"

        # Test low-quality item
        low_quality = {
            "title": "Random unrelated paper",
            "source": "unknown",
            "citations": 0,
        }
        scored_low = scorer.score(low_quality, "spatial transcriptomics")
        assert scored_low.priority_score < scored.priority_score


# =============================================================================
# Evidence System Tests
# =============================================================================

class TestEvidenceSystem:
    """Test evidence extraction and citation system."""

    def test_evidence_extractor_init(self):
        """Test evidence extractor initializes correctly."""
        from evidence_extractor import EvidenceExtractor

        extractor = EvidenceExtractor()
        assert extractor.nli_pipeline is not None
        assert extractor.nlp is not None

    def test_nli_scoring(self):
        """Test NLI scoring works correctly."""
        from evidence_extractor import EvidenceExtractor

        extractor = EvidenceExtractor()

        # Test entailment
        pairs = [{"text": "Graph neural networks are used for node classification.", "text_pair": "GNNs classify nodes."}]
        results = extractor.nli_pipeline(pairs)

        assert len(results) > 0
        # Should detect some level of entailment
        scores = extractor._parse_nli_scores(results[0])
        assert "entailment" in scores
        print(f"NLI entailment score: {scores['entailment']:.3f}")


# =============================================================================
# Utility Tests
# =============================================================================

class TestUtilities:
    """Test utility functions."""

    def test_title_extraction_from_content(self):
        """Test title extraction helper function."""
        from hybrid_search_v2 import _extract_title_from_content, _is_placeholder_title

        # Test numeric placeholder detection
        assert _is_placeholder_title("12345") == True
        assert _is_placeholder_title("Machine Learning") == False

        # Test title extraction
        content = "# 12345\n\nMachine learning is transforming biology."
        title = _extract_title_from_content(content)
        assert "12345" not in title
        assert len(title) > 10

    def test_rrf_fusion(self):
        """Test reciprocal rank fusion."""
        from hybrid_search_v2 import reciprocal_rank_fusion, SearchResult

        list1 = [
            SearchResult("a", "A", "content", "source1", 1.0, {}),
            SearchResult("b", "B", "content", "source1", 0.9, {}),
        ]
        list2 = [
            SearchResult("b", "B", "content", "source2", 1.0, {}),
            SearchResult("c", "C", "content", "source2", 0.9, {}),
        ]

        merged = reciprocal_rank_fusion([list1, list2])

        # "b" appears in both lists, should rank highest
        assert merged[0].id == "b"


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
