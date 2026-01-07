#!/usr/bin/env python3
"""
OpenAlex Source Connector

OpenAlex is free, open, and indexes 250M+ works. Use for:
- Initial discovery (comprehensive metadata, citation counts)
- Author and institution tracking
- Cross-domain research discovery

Note: OpenAlex doesn't provide full-text PDFs. Use EuropePMC for PDF fetching.
"""

import time
from typing import List, Dict, Optional
from datetime import datetime, timedelta
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


class OpenAlexSource:
    """
    OpenAlex API source for comprehensive research discovery.

    API Docs: https://docs.openalex.org/
    Rate Limit: 10 req/sec with polite pool (email required)
    """

    name = "openalex"
    rate_limit = RATE_LIMITS.get("openalex", 10.0)
    base_url = "https://api.openalex.org"

    def __init__(self, email: str = None, api_key: str = None):
        """Initialize with optional email and API key for polite pool access.

        Args:
            email: Email for polite pool (10 req/sec vs 1 req/sec)
            api_key: OpenAlex API key for premium features
        """
        from lib.config import OPENALEX_EMAIL, OPENALEX_API_KEY
        self.email = email or OPENALEX_EMAIL
        self.api_key = api_key or OPENALEX_API_KEY
        self.client = httpx.Client(timeout=30.0)
        self.last_request = 0.0

    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self.last_request
        min_interval = 1.0 / self.rate_limit
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self.last_request = time.time()

    def discover(
        self,
        query: str,
        max_results: int = 50,
        since_days: int = 365,
        sort: str = "cited_by_count:desc",
        filter_oa: bool = False
    ) -> List[Dict]:
        """Search OpenAlex for works matching query.

        Args:
            query: Search query (searches title, abstract)
            max_results: Maximum results to return
            since_days: Only include works from last N days
            sort: Sort order (cited_by_count:desc, publication_date:desc, relevance_score:desc)
            filter_oa: Only return open access works

        Returns:
            List of work dicts in standard format
        """
        self._rate_limit()

        # Build filter
        filters = []
        if since_days:
            from_date = (datetime.now() - timedelta(days=since_days)).strftime("%Y-%m-%d")
            filters.append(f"from_publication_date:{from_date}")
        if filter_oa:
            filters.append("is_oa:true")

        params = {
            "search": query,
            "per_page": min(max_results, 200),  # API max is 200
            "sort": sort,
            "mailto": self.email,
        }
        if self.api_key:
            params["api_key"] = self.api_key
        if filters:
            params["filter"] = ",".join(filters)

        try:
            response = self.client.get(f"{self.base_url}/works", params=params)
            response.raise_for_status()

            data = response.json()
            results = []

            for work in data.get("results", [])[:max_results]:
                parsed = self._parse_work(work)
                if parsed:
                    results.append(parsed)

            logger.info(f"OpenAlex: Found {len(results)} results for '{query}'")
            return results

        except Exception as e:
            logger.error(f"OpenAlex search failed: {e}")
            return []

    def get_by_doi(self, doi: str) -> Optional[Dict]:
        """Fetch a specific work by DOI."""
        self._rate_limit()

        # OpenAlex uses DOI URLs as IDs
        openalex_id = f"https://doi.org/{doi}"
        try:
            response = self.client.get(
                f"{self.base_url}/works/{openalex_id}",
                params={"mailto": self.email}
            )

            if response.status_code == 404:
                return None
            response.raise_for_status()

            return self._parse_work(response.json())
        except Exception as e:
            logger.error(f"OpenAlex DOI lookup failed: {e}")
            return None

    def get_author_works(self, author_id: str, max_results: int = 50) -> List[Dict]:
        """Get recent works by an author.

        Args:
            author_id: OpenAlex author ID (e.g., "A5004308400")
            max_results: Maximum results to return
        """
        self._rate_limit()

        params = {
            "filter": f"author.id:{author_id}",
            "sort": "publication_date:desc",
            "per_page": min(max_results, 200),
            "mailto": self.email,
        }

        try:
            response = self.client.get(f"{self.base_url}/works", params=params)
            response.raise_for_status()

            return [self._parse_work(w) for w in response.json().get("results", []) if w]
        except Exception as e:
            logger.error(f"OpenAlex author lookup failed: {e}")
            return []

    def search_authors(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search for authors by name.

        Args:
            query: Author name to search
            max_results: Maximum results to return

        Returns:
            List of author dicts with id, name, affiliation, works_count
        """
        self._rate_limit()

        params = {
            "search": query,
            "per_page": min(max_results, 50),
            "mailto": self.email,
        }

        try:
            response = self.client.get(f"{self.base_url}/authors", params=params)
            response.raise_for_status()

            authors = []
            for author in response.json().get("results", []):
                authors.append({
                    "id": author.get("id", "").replace("https://openalex.org/", ""),
                    "name": author.get("display_name", ""),
                    "affiliation": author.get("last_known_institution", {}).get("display_name", ""),
                    "works_count": author.get("works_count", 0),
                    "cited_by_count": author.get("cited_by_count", 0),
                    "orcid": author.get("orcid"),
                })
            return authors
        except Exception as e:
            logger.error(f"OpenAlex author search failed: {e}")
            return []

    def _parse_work(self, work: Dict) -> Optional[Dict]:
        """Parse OpenAlex work to standard format."""
        if not work:
            return None

        # Extract authors
        authors = []
        for authorship in work.get("authorships", [])[:10]:
            author = authorship.get("author", {})
            if author.get("display_name"):
                authors.append(author["display_name"])

        # Extract best OA URL
        oa_url = None
        pdf_url = None
        if work.get("open_access", {}).get("is_oa"):
            oa_url = work.get("open_access", {}).get("oa_url")
            # Check for PDF in locations
            for loc in work.get("locations", []):
                if loc.get("pdf_url"):
                    pdf_url = loc["pdf_url"]
                    break

        # Extract concepts/topics
        concepts = [c.get("display_name") for c in work.get("concepts", [])[:5] if c.get("display_name")]

        # Clean DOI
        doi = work.get("doi", "")
        if doi:
            doi = doi.replace("https://doi.org/", "")

        return {
            "source": "openalex",
            "external_id": work.get("id", "").replace("https://openalex.org/", ""),
            "doi": doi if doi else None,
            "title": work.get("title", "Untitled"),
            "abstract": work.get("abstract"),  # May be None
            "authors": authors,
            "published_at": work.get("publication_date"),
            "url": work.get("doi") or work.get("id"),
            "citations": work.get("cited_by_count", 0),
            "is_open_access": work.get("open_access", {}).get("is_oa", False),
            "oa_url": oa_url,
            "pdf_url": pdf_url,
            "venue": work.get("primary_location", {}).get("source", {}).get("display_name"),
            "concepts": concepts,
            "type": work.get("type"),  # journal-article, preprint, etc.
        }

    def close(self):
        """Close HTTP client."""
        self.client.close()
