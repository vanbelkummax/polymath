#!/usr/bin/env python3
"""
Test Semantic Scholar API integration.

Tests:
1. API key loading from environment
2. Rate limiting enforcement (1 req/sec)
3. Paper search functionality
4. Result parsing
"""

import os
import sys
import time
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.sentry.sources.semanticscholar import SemanticScholarSource


@pytest.fixture
def ss_source():
    """Create SemanticScholarSource instance."""
    # Load API key from environment
    api_key = os.environ.get("SEMANTICSCHOLAR_API_KEY")
    if not api_key:
        pytest.skip("SEMANTICSCHOLAR_API_KEY not set")

    return SemanticScholarSource(api_key=api_key)


def test_api_key_loading(ss_source):
    """Test that API key is loaded from environment."""
    assert ss_source.api_key is not None
    assert len(ss_source.api_key) > 0
    print(f"API key loaded: {ss_source.api_key[:10]}...")


def test_rate_limiting(ss_source):
    """Test that rate limiting enforces 1 req/sec."""
    # Make 3 requests and measure timing
    start = time.time()

    for i in range(3):
        ss_source._rate_limit()

    elapsed = time.time() - start

    # Should take at least 2 seconds (3 requests at 1 req/sec means 2 seconds wait)
    assert elapsed >= 2.0, f"Rate limiting not enforced: {elapsed:.2f}s < 2.0s"
    print(f"✅ Rate limiting enforced: {elapsed:.2f}s for 3 requests")


def test_search_basic(ss_source):
    """Test basic search functionality."""
    results = ss_source.discover(
        query="spatial transcriptomics",
        max_results=5
    )

    assert isinstance(results, list)
    assert len(results) > 0, "No results returned"
    assert len(results) <= 5, "Too many results returned"

    # Check first result structure
    paper = results[0]
    assert "title" in paper
    assert "source" in paper
    assert paper["source"] == "semanticscholar"

    print(f"\n✅ Found {len(results)} papers")
    print(f"First paper: {paper['title']}")
    print(f"  Authors: {', '.join(paper.get('authors', [])[:3])}")
    print(f"  Year: {paper.get('year')}")
    print(f"  Citations: {paper.get('citation_count')}")


def test_search_with_filters(ss_source):
    """Test search with year and citation filters."""
    results = ss_source.discover(
        query="optimal transport",
        year_min=2020,
        min_citations=50,
        max_results=10
    )

    assert isinstance(results, list)

    if len(results) > 0:
        paper = results[0]
        # Check filters applied
        if paper.get("year"):
            assert paper["year"] >= 2020, f"Year filter failed: {paper['year']}"
        if paper.get("citation_count"):
            assert paper["citation_count"] >= 50, f"Citation filter failed: {paper['citation_count']}"

        print(f"\n✅ Filtered search found {len(results)} papers")
        print(f"Top cited: {paper['title']} ({paper.get('citation_count')} citations)")


def test_get_paper_by_doi(ss_source):
    """Test retrieving paper by DOI."""
    # Use a known DOI (Nature 2021 Visium HD paper)
    doi = "10.1038/s41592-022-01409-2"

    paper = ss_source.get_paper_by_id(doi, id_type="DOI")

    assert paper is not None
    assert "title" in paper
    assert "spatial" in paper["title"].lower() or "visium" in paper["title"].lower()

    print(f"\n✅ Retrieved paper by DOI:")
    print(f"  Title: {paper['title']}")
    print(f"  Year: {paper.get('year')}")


def test_highly_cited_papers(ss_source):
    """Test retrieving highly cited papers in a field."""
    results = ss_source.get_highly_cited(
        field="Computer Science",
        year=2023,
        min_citations=100,
        max_results=5
    )

    assert isinstance(results, list)

    if len(results) > 0:
        print(f"\n✅ Found {len(results)} highly cited CS papers from 2023")
        for i, paper in enumerate(results[:3], 1):
            print(f"  {i}. {paper['title']} ({paper.get('citation_count')} citations)")


def test_open_access_filter(ss_source):
    """Test open access filtering."""
    results = ss_source.discover(
        query="machine learning",
        open_access_only=True,
        max_results=10
    )

    assert isinstance(results, list)

    if len(results) > 0:
        # Check that results have OA indicators
        oa_count = sum(1 for p in results if p.get("is_open_access"))
        print(f"\n✅ OA filter: {oa_count}/{len(results)} papers marked as open access")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
