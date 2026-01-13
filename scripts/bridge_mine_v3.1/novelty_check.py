#!/usr/bin/env python3
"""
Novelty Check for Bridge Mine v3.1

Checks whether a cross-domain bridge is:
- Mode A (EXISTING): Explicit co-mention in literature → useful for lit review, not novel
- Mode B (NOVEL_CANDIDATE): No direct "{method} + spatial transcriptomics" hits → actual discovery

Uses PubMed and Semantic Scholar to verify novelty.
"""

import json
import os
import sys
import time
import re
import urllib.parse
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

sys.path.insert(0, '/home/user/polymath-repo')

import requests


class NoveltyMode(str, Enum):
    EXISTING = "EXISTING"                 # Found in literature (Mode A)
    NOVEL_CANDIDATE = "NOVEL_CANDIDATE"   # Not found (Mode B)
    NEEDS_REVIEW = "NEEDS_REVIEW"         # Ambiguous results
    CHECK_FAILED = "CHECK_FAILED"         # API error


@dataclass
class NoveltyResult:
    """Result of novelty check."""
    mode: NoveltyMode
    pubmed_hits: int
    s2_hits: int
    query_used: str
    evidence_urls: List[str]  # URLs to found papers (if EXISTING)
    confidence: float
    rationale: str


# PubMed E-utilities
PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_SUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

# Semantic Scholar
S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


def normalize_concept_for_search(concept: str) -> str:
    """Convert concept name to search-friendly form."""
    # Replace underscores with spaces
    term = concept.replace('_', ' ')
    # Remove special characters
    term = re.sub(r'[^\w\s-]', '', term)
    return term.strip()


def build_combined_query(
    source_concept: str,
    target_domain: str = "spatial transcriptomics"
) -> str:
    """
    Build a query checking for co-mention of source method and target domain.
    """
    source = normalize_concept_for_search(source_concept)
    target = normalize_concept_for_search(target_domain)

    # Quote exact phrases
    return f'"{source}" AND "{target}"'


def search_pubmed(query: str, max_results: int = 5) -> Tuple[int, List[dict]]:
    """
    Search PubMed for papers matching query.

    Returns:
        (total_count, list of paper metadata dicts)
    """
    try:
        # Search
        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'json',
            'usehistory': 'n'
        }

        response = requests.get(PUBMED_SEARCH_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        result = data.get('esearchresult', {})
        total = int(result.get('count', 0))
        ids = result.get('idlist', [])

        if not ids:
            return total, []

        # Get summaries
        summary_params = {
            'db': 'pubmed',
            'id': ','.join(ids),
            'retmode': 'json'
        }

        time.sleep(0.3)  # Rate limiting
        summary_response = requests.get(PUBMED_SUMMARY_URL, params=summary_params, timeout=10)
        summary_response.raise_for_status()
        summary_data = summary_response.json()

        papers = []
        for pmid in ids:
            paper_data = summary_data.get('result', {}).get(pmid, {})
            if paper_data:
                papers.append({
                    'pmid': pmid,
                    'title': paper_data.get('title', ''),
                    'year': paper_data.get('pubdate', '')[:4],
                    'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                })

        return total, papers

    except Exception as e:
        print(f"    PubMed search error: {e}")
        return -1, []


def search_semantic_scholar(query: str, max_results: int = 5) -> Tuple[int, List[dict]]:
    """
    Search Semantic Scholar for papers matching query.

    Returns:
        (total_count, list of paper metadata dicts)
    """
    try:
        params = {
            'query': query,
            'limit': max_results,
            'fields': 'title,year,url,paperId'
        }

        response = requests.get(S2_SEARCH_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        total = data.get('total', 0)
        papers = []

        for paper in data.get('data', []):
            papers.append({
                'paper_id': paper.get('paperId', ''),
                'title': paper.get('title', ''),
                'year': str(paper.get('year', '')),
                'url': paper.get('url', f"https://www.semanticscholar.org/paper/{paper.get('paperId', '')}")
            })

        return total, papers

    except Exception as e:
        print(f"    Semantic Scholar search error: {e}")
        return -1, []


def check_novelty(
    source_concept: str,
    target_concept: str,
    target_domain: str = "spatial transcriptomics",
    min_hits_for_existing: int = 3
) -> NoveltyResult:
    """
    Check if a source→target bridge is novel or already exists in literature.

    Args:
        source_concept: The source method/technique
        target_concept: The target problem/application
        target_domain: The target domain (default: spatial transcriptomics)
        min_hits_for_existing: Minimum combined hits to mark as EXISTING

    Returns:
        NoveltyResult with mode and evidence
    """
    # Build search query
    query = build_combined_query(source_concept, target_domain)

    # Search PubMed
    pubmed_total, pubmed_papers = search_pubmed(query)
    time.sleep(0.5)  # Rate limiting between APIs

    # Search Semantic Scholar
    s2_total, s2_papers = search_semantic_scholar(query)

    # Combine results
    all_urls = []
    for p in pubmed_papers:
        all_urls.append(p.get('url', ''))
    for p in s2_papers:
        all_urls.append(p.get('url', ''))
    all_urls = list(set(filter(None, all_urls)))[:5]  # Dedupe, limit

    # Handle API errors
    if pubmed_total < 0 and s2_total < 0:
        return NoveltyResult(
            mode=NoveltyMode.CHECK_FAILED,
            pubmed_hits=0,
            s2_hits=0,
            query_used=query,
            evidence_urls=[],
            confidence=0.0,
            rationale="Both PubMed and Semantic Scholar APIs failed"
        )

    # Use whichever worked, or average
    pubmed_count = max(0, pubmed_total)
    s2_count = max(0, s2_total)
    combined_hits = pubmed_count + s2_count

    # Classify
    if combined_hits >= min_hits_for_existing:
        return NoveltyResult(
            mode=NoveltyMode.EXISTING,
            pubmed_hits=pubmed_count,
            s2_hits=s2_count,
            query_used=query,
            evidence_urls=all_urls,
            confidence=min(1.0, combined_hits / 10),
            rationale=f"Found {combined_hits} papers with explicit co-mention - not novel"
        )

    elif combined_hits == 0:
        return NoveltyResult(
            mode=NoveltyMode.NOVEL_CANDIDATE,
            pubmed_hits=0,
            s2_hits=0,
            query_used=query,
            evidence_urls=[],
            confidence=0.9,
            rationale="No direct co-mention found - novel transfer candidate"
        )

    else:  # 1-2 hits
        return NoveltyResult(
            mode=NoveltyMode.NEEDS_REVIEW,
            pubmed_hits=pubmed_count,
            s2_hits=s2_count,
            query_used=query,
            evidence_urls=all_urls,
            confidence=0.5,
            rationale=f"Found {combined_hits} papers - manual review needed to assess novelty"
        )


def check_bridge_novelty(bridge_info: dict) -> NoveltyResult:
    """
    Check novelty for a bridge given its info dict.
    """
    source = bridge_info.get('source_concept', '')
    target = bridge_info.get('target_concept', '')

    return check_novelty(source, target)


def main():
    """Test novelty checking."""

    test_cases = [
        # Likely EXISTING (well-studied area)
        ("attention mechanism", "image classification"),

        # Likely NOVEL_CANDIDATE (unusual transfer)
        ("tomographic imaging", "spatial transcriptomics"),

        # Possibly NEEDS_REVIEW
        ("contrastive learning", "cell segmentation"),

        # Very niche
        ("learned primal dual", "spatial transcriptomics"),
    ]

    print("Novelty Check Tests:\n")
    for source, target in test_cases:
        print(f"Checking: {source} → {target}")
        result = check_novelty(source, target)
        print(f"  MODE: {result.mode.value}")
        print(f"  PubMed hits: {result.pubmed_hits}")
        print(f"  S2 hits: {result.s2_hits}")
        print(f"  Query: {result.query_used}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Rationale: {result.rationale}")
        if result.evidence_urls:
            print(f"  Evidence URLs:")
            for url in result.evidence_urls[:3]:
                print(f"    - {url}")
        print()
        time.sleep(1)  # Rate limiting


if __name__ == "__main__":
    main()
