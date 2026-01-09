#!/usr/bin/env python3
"""
Semantic Scholar Source Connector

Semantic Scholar Academic Graph API provides:
- 200M+ papers across all fields
- Citation graphs and influence metrics
- Open access detection
- Author disambiguation
- Highly cited paper detection

Rate Limit: 1 request per second with API key
API Docs: https://api.semanticscholar.org/api-docs/
"""

import os
import time
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging

try:
    import httpx
except ImportError:
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "pip", "install", "httpx", "-q"])
    import httpx

from ..config import RATE_LIMITS

logger = logging.getLogger(__name__)

BASE_URL = "https://api.semanticscholar.org/graph/v1"


class SemanticScholarSource:
    """
    Semantic Scholar API connector.

    Features:
    - 200M+ papers indexed
    - Citation and influence metrics
    - Open access links
    - Author disambiguation
    - Highly cited detection

    API Docs: https://api.semanticscholar.org/api-docs/
    Rate Limit: 1 request/second with API key
    """

    name = "semanticscholar"
    rate_limit = RATE_LIMITS.get("semanticscholar", 1.0)  # 1 req/sec with API key

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Semantic Scholar connector.

        Args:
            api_key: API key (reads from env SEMANTICSCHOLAR_API_KEY if not provided)
        """
        self.api_key = api_key or os.environ.get("SEMANTICSCHOLAR_API_KEY")
        if not self.api_key:
            logger.warning("No Semantic Scholar API key - rate limits will be strict (100 req/day)")

        headers = {"x-api-key": self.api_key} if self.api_key else {}
        self.client = httpx.Client(timeout=30, headers=headers)
        self.last_request = 0

    def _rate_limit(self):
        """Enforce rate limiting (1 req/sec with key)."""
        elapsed = time.time() - self.last_request
        min_interval = 1.0 / self.rate_limit
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self.last_request = time.time()

    def discover(
        self,
        query: str,
        year_min: Optional[int] = None,
        max_results: int = 50,
        fields_of_study: Optional[List[str]] = None,
        min_citations: int = 0,
        open_access_only: bool = False
    ) -> List[Dict]:
        """
        Search Semantic Scholar for papers.

        Args:
            query: Search query
            year_min: Minimum publication year (e.g., 2020)
            max_results: Maximum results to return
            fields_of_study: Filter by fields (e.g., ['Computer Science', 'Biology'])
            min_citations: Minimum citation count
            open_access_only: Only return open access papers

        Returns:
            List of paper dicts with metadata
        """
        self._rate_limit()

        params = {
            "query": query,
            "limit": min(max_results, 100),
            "fields": "paperId,title,abstract,year,authors,citationCount,influentialCitationCount,isOpenAccess,openAccessPdf,fieldsOfStudy,publicationTypes,publicationDate,journal,externalIds"
        }

        if year_min:
            params["year"] = f"{year_min}-"

        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)

        if min_citations > 0:
            params["minCitationCount"] = min_citations

        if open_access_only:
            params["openAccessPdf"] = ""  # Filter for papers with OA PDFs

        try:
            response = self.client.get(f"{BASE_URL}/paper/search", params=params)
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("data", []):
                parsed = self._parse_result(item)
                if parsed:
                    results.append(parsed)

            logger.info(f"Semantic Scholar: Found {len(results)} results for '{query}'")
            return results

        except Exception as e:
            logger.error(f"Semantic Scholar search failed: {e}")
            return []

    def get_paper_by_id(
        self,
        paper_id: str,
        id_type: str = "DOI"
    ) -> Optional[Dict]:
        """
        Retrieve paper by ID.

        Args:
            paper_id: Paper identifier
            id_type: ID type - one of: DOI, ArXiv, MAG, ACL, PubMed, CorpusId, SemanticScholar

        Returns:
            Paper dict or None
        """
        self._rate_limit()

        # Map ID types to API format
        id_prefix_map = {
            "DOI": "DOI:",
            "ArXiv": "ARXIV:",
            "PubMed": "PMID:",
            "CorpusId": "CorpusID:",
            "SemanticScholar": "",
            "MAG": "MAG:",
            "ACL": "ACL:"
        }

        prefix = id_prefix_map.get(id_type, "")
        full_id = f"{prefix}{paper_id}" if prefix else paper_id

        fields = "paperId,title,abstract,year,authors,citationCount,influentialCitationCount,isOpenAccess,openAccessPdf,fieldsOfStudy,publicationTypes,publicationDate,journal,externalIds,references,citations"

        try:
            response = self.client.get(
                f"{BASE_URL}/paper/{full_id}",
                params={"fields": fields}
            )
            response.raise_for_status()
            item = response.json()

            return self._parse_result(item)

        except Exception as e:
            logger.error(f"Failed to retrieve paper {paper_id}: {e}")
            return None

    def get_highly_cited(
        self,
        field: str = "Computer Science",
        year: Optional[int] = None,
        min_citations: int = 100,
        max_results: int = 50
    ) -> List[Dict]:
        """
        Get highly cited papers in a field.

        Args:
            field: Field of study
            year: Publication year (optional)
            min_citations: Minimum citation count
            max_results: Maximum results

        Returns:
            List of highly cited papers
        """
        query = f"fieldsOfStudy:{field}"
        if year:
            query += f" year:{year}"

        return self.discover(
            query=query,
            min_citations=min_citations,
            max_results=max_results,
            fields_of_study=[field]
        )

    def get_author_papers(
        self,
        author_id: str,
        max_results: int = 50
    ) -> List[Dict]:
        """
        Get papers by author ID.

        Args:
            author_id: Semantic Scholar author ID
            max_results: Maximum results

        Returns:
            List of author's papers
        """
        self._rate_limit()

        fields = "paperId,title,year,citationCount,influentialCitationCount,isOpenAccess"

        try:
            response = self.client.get(
                f"{BASE_URL}/author/{author_id}/papers",
                params={"fields": fields, "limit": min(max_results, 100)}
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("data", []):
                parsed = self._parse_result(item)
                if parsed:
                    results.append(parsed)

            return results

        except Exception as e:
            logger.error(f"Failed to get author papers: {e}")
            return []

    def _parse_result(self, item: Dict) -> Optional[Dict]:
        """Parse Semantic Scholar result into standard format."""
        try:
            # Extract external IDs
            ext_ids = item.get("externalIds", {}) or {}
            doi = ext_ids.get("DOI")
            pmid = ext_ids.get("PubMed")
            arxiv = ext_ids.get("ArXiv")

            # Extract PDF URL
            pdf_url = None
            oa_pdf = item.get("openAccessPdf")
            if oa_pdf and isinstance(oa_pdf, dict):
                pdf_url = oa_pdf.get("url")

            # Extract authors
            authors = []
            for author in item.get("authors", []):
                if isinstance(author, dict):
                    name = author.get("name", "Unknown")
                    authors.append(name)

            # Citation metrics
            citation_count = item.get("citationCount", 0)
            influential_count = item.get("influentialCitationCount", 0)

            # Fields of study
            fields = item.get("fieldsOfStudy", [])
            if not isinstance(fields, list):
                fields = []

            # Publication info
            pub_date = item.get("publicationDate")
            year = item.get("year")
            journal = item.get("journal", {})
            if isinstance(journal, dict):
                journal_name = journal.get("name")
            else:
                journal_name = None

            return {
                "title": item.get("title", "Untitled"),
                "abstract": item.get("abstract"),
                "authors": authors,
                "year": year,
                "doi": doi,
                "pmid": pmid,
                "arxiv_id": arxiv,
                "pdf_url": pdf_url,
                "is_open_access": item.get("isOpenAccess", False),
                "citation_count": citation_count,
                "influential_citation_count": influential_count,
                "fields_of_study": fields,
                "publication_date": pub_date,
                "journal": journal_name,
                "source": "semanticscholar",
                "source_id": item.get("paperId"),
                "url": f"https://www.semanticscholar.org/paper/{item.get('paperId')}",
                # Metadata for scoring
                "metadata": {
                    "citation_count": citation_count,
                    "influential_citations": influential_count,
                    "fields": fields,
                    "is_oa": item.get("isOpenAccess", False)
                }
            }

        except Exception as e:
            logger.warning(f"Failed to parse Semantic Scholar result: {e}")
            return None

    def search_by_citations(
        self,
        seed_paper_id: str,
        direction: str = "references",
        max_results: int = 50
    ) -> List[Dict]:
        """
        Get papers that cite (or are cited by) a seed paper.

        Args:
            seed_paper_id: Semantic Scholar paper ID
            direction: "references" or "citations"
            max_results: Maximum results

        Returns:
            List of related papers
        """
        self._rate_limit()

        endpoint = "references" if direction == "references" else "citations"
        fields = "paperId,title,year,citationCount,isOpenAccess"

        try:
            response = self.client.get(
                f"{BASE_URL}/paper/{seed_paper_id}/{endpoint}",
                params={"fields": fields, "limit": min(max_results, 100)}
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("data", []):
                # References/citations return nested structure
                paper = item.get("citedPaper") if direction == "references" else item.get("citingPaper")
                if paper:
                    parsed = self._parse_result(paper)
                    if parsed:
                        results.append(parsed)

            return results

        except Exception as e:
            logger.error(f"Failed to get {direction}: {e}")
            return []
