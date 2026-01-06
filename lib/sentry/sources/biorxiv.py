#!/usr/bin/env python3
"""
bioRxiv/medRxiv Source Connector

For biology and medicine preprints. Critical for your spatial transcriptomics work.
Supports both bioRxiv and medRxiv via the same API.

API Docs: https://api.biorxiv.org/
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

# bioRxiv API base URL
BASE_URL = "https://api.biorxiv.org"

# bioRxiv categories relevant to your work
RELEVANT_CATEGORIES = {
    "bioinformatics",
    "cancer_biology",
    "cell_biology",
    "genomics",
    "microbiology",
    "molecular_biology",
    "pathology",
    "systems_biology",
    "synthetic_biology",
}


class BioRxivSource:
    """
    bioRxiv/medRxiv API connector.

    API provides:
    - Content API: Search by date range, server, category
    - Details API: Get paper by DOI
    - Publisher API: Get papers with journal acceptance info

    Note: No full-text search - uses date ranges and categories.
    For semantic search, we search returned abstracts locally.
    """

    name = "biorxiv"
    rate_limit = RATE_LIMITS.get("biorxiv", 1.0)  # 1 request per second

    def __init__(self, server: str = "biorxiv"):
        """
        Initialize bioRxiv connector.

        Args:
            server: 'biorxiv' or 'medrxiv'
        """
        self.server = server
        self.client = httpx.Client(timeout=30)
        self.last_request = 0

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
        days_back: int = 90,
        categories: List[str] = None,
    ) -> List[Dict]:
        """
        Search bioRxiv for papers.

        Note: bioRxiv API doesn't support keyword search - we get recent papers
        and filter locally by matching query terms in title/abstract.

        Args:
            query: Search query (matched against title/abstract)
            max_results: Maximum results to return
            days_back: How many days back to search
            categories: bioRxiv categories to filter (optional)

        Returns:
            List of paper dicts matching query
        """
        # Get recent papers
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        all_papers = self._fetch_by_date_range(
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
            max_pages=10  # ~1000 papers max
        )

        if not all_papers:
            return []

        # Filter by query terms
        query_terms = query.lower().split()
        matched = []

        for paper in all_papers:
            # Search in title and abstract
            searchable = (
                (paper.get("title") or "").lower() +
                " " +
                (paper.get("abstract") or "").lower()
            )

            # Check if all query terms present
            if all(term in searchable for term in query_terms):
                # Category filter
                if categories:
                    paper_cat = paper.get("category", "").lower()
                    if not any(cat.lower() in paper_cat for cat in categories):
                        continue
                matched.append(paper)

        # Sort by recency
        matched.sort(key=lambda x: x.get("published_at") or "", reverse=True)

        logger.info(f"bioRxiv: Found {len(matched)} results for '{query}' from {len(all_papers)} papers")
        return matched[:max_results]

    def _fetch_by_date_range(
        self,
        start_date: str,
        end_date: str,
        max_pages: int = 5
    ) -> List[Dict]:
        """
        Fetch papers within date range.

        API format: /details/{server}/{start_date}/{end_date}/{cursor}
        """
        papers = []
        cursor = 0

        for page in range(max_pages):
            self._rate_limit()

            url = f"{BASE_URL}/details/{self.server}/{start_date}/{end_date}/{cursor}"

            try:
                response = self.client.get(url)
                response.raise_for_status()
                data = response.json()

                collection = data.get("collection", [])
                if not collection:
                    break

                for item in collection:
                    parsed = self._parse_paper(item)
                    if parsed:
                        papers.append(parsed)

                # Check for more pages
                messages = data.get("messages", [])
                total_msg = next((m for m in messages if "total" in m.get("status", "")), None)
                if total_msg:
                    total = int(total_msg["status"].split()[-1])
                    if cursor + len(collection) >= total:
                        break

                cursor += 100  # API returns 100 per page

            except Exception as e:
                logger.error(f"bioRxiv fetch failed: {e}")
                break

        return papers

    def get_by_doi(self, doi: str) -> Optional[Dict]:
        """Get paper by DOI."""
        self._rate_limit()

        # Clean DOI
        doi = doi.replace("https://doi.org/", "").strip()

        url = f"{BASE_URL}/details/{self.server}/{doi}"

        try:
            response = self.client.get(url)
            response.raise_for_status()
            data = response.json()

            collection = data.get("collection", [])
            if collection:
                return self._parse_paper(collection[0])
            return None

        except Exception as e:
            logger.error(f"bioRxiv DOI lookup failed: {e}")
            return None

    def get_recent(self, days: int = 7, max_results: int = 100) -> List[Dict]:
        """Get most recent papers."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        papers = self._fetch_by_date_range(
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
            max_pages=max_results // 100 + 1
        )

        return papers[:max_results]

    def _parse_paper(self, item: Dict) -> Optional[Dict]:
        """Parse bioRxiv API response item."""
        try:
            doi = item.get("doi", "")
            biorxiv_doi = item.get("biorxiv_doi", doi)

            # Parse date
            pub_date = None
            date_str = item.get("date")
            if date_str:
                try:
                    pub_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                except ValueError:
                    pass

            # Parse authors (format: "Last, First; Last, First")
            authors_str = item.get("authors", "")
            authors = [a.strip() for a in authors_str.split(";") if a.strip()]

            return {
                "source": self.server,
                "external_id": biorxiv_doi or doi,
                "doi": doi,
                "title": item.get("title", ""),
                "abstract": item.get("abstract", ""),
                "authors": authors,
                "published_at": pub_date,
                "category": item.get("category", ""),
                "url": f"https://www.{self.server}.org/content/{doi}",
                "pdf_url": f"https://www.{self.server}.org/content/{doi}.full.pdf",
                "oa_url": f"https://www.{self.server}.org/content/{doi}.full.pdf",
                "is_open_access": True,  # bioRxiv is always OA
                "citations": 0,  # bioRxiv doesn't provide citation counts
                "version": item.get("version", "1"),
                "jatsxml": item.get("jatsxml"),  # JATS XML URL if available
            }

        except Exception as e:
            logger.error(f"Failed to parse bioRxiv paper: {e}")
            return None

    def close(self):
        """Close HTTP client."""
        self.client.close()


class MedRxivSource(BioRxivSource):
    """medRxiv connector (same API, different server)."""

    name = "medrxiv"

    def __init__(self):
        super().__init__(server="medrxiv")


# Convenience functions
def search_biorxiv(query: str, max_results: int = 50, days_back: int = 90) -> List[Dict]:
    """Quick search function for bioRxiv."""
    source = BioRxivSource()
    try:
        return source.discover(query, max_results, days_back)
    finally:
        source.close()


def search_medrxiv(query: str, max_results: int = 50, days_back: int = 90) -> List[Dict]:
    """Quick search function for medRxiv."""
    source = MedRxivSource()
    try:
        return source.discover(query, max_results, days_back)
    finally:
        source.close()


def get_recent_preprints(days: int = 7, max_results: int = 100, server: str = "biorxiv") -> List[Dict]:
    """Get recent preprints without search filter."""
    source = BioRxivSource(server=server)
    try:
        return source.get_recent(days, max_results)
    finally:
        source.close()
