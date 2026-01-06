#!/usr/bin/env python3
"""
Europe PMC Source Connector

Preferred over PubMed for:
- Better open access detection
- Full-text availability
- Preprint indexing
"""

import time
from typing import List, Dict, Optional, Any
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

BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest"


class EuropePMCSource:
    """
    Europe PMC API connector.

    API Docs: https://europepmc.org/RestfulWebService
    """

    name = "europepmc"
    rate_limit = RATE_LIMITS.get("europepmc", 10.0)

    def __init__(self):
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
        since_days: int = 30,
        max_results: int = 50,
        sort: str = "CITED desc"
    ) -> List[Dict]:
        """
        Search Europe PMC for papers.

        Args:
            query: Search query
            since_days: Only papers from last N days
            max_results: Maximum results to return
            sort: Sort order (CITED desc, P_PDATE_D desc, RELEVANCE)

        Returns:
            List of paper dicts
        """
        self._rate_limit()

        # Build date filter
        since_date = (datetime.now() - timedelta(days=since_days)).strftime("%Y-%m-%d")
        full_query = f"({query}) AND (FIRST_PDATE:[{since_date} TO *])"

        params = {
            "query": full_query,
            "format": "json",
            "pageSize": min(max_results, 100),
            "sort": sort,
            "resultType": "core",  # Include full metadata
        }

        try:
            response = self.client.get(f"{BASE_URL}/search", params=params)
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("resultList", {}).get("result", []):
                parsed = self._parse_result(item)
                if parsed:
                    results.append(parsed)

            logger.info(f"Europe PMC: Found {len(results)} results for '{query}'")
            return results

        except Exception as e:
            logger.error(f"Europe PMC search failed: {e}")
            return []

    def get_by_pmid(self, pmid: str) -> Optional[Dict]:
        """Get paper by PMID."""
        self._rate_limit()

        try:
            response = self.client.get(
                f"{BASE_URL}/search",
                params={
                    "query": f"EXT_ID:{pmid}",
                    "format": "json",
                    "resultType": "core"
                }
            )
            response.raise_for_status()
            data = response.json()

            results = data.get("resultList", {}).get("result", [])
            if results:
                return self._parse_result(results[0])
            return None

        except Exception as e:
            logger.error(f"Europe PMC PMID lookup failed: {e}")
            return None

    def get_citations(self, pmid: str) -> int:
        """Get citation count for a paper."""
        self._rate_limit()

        try:
            response = self.client.get(
                f"{BASE_URL}/{pmid}/citations",
                params={"format": "json", "page": 1, "pageSize": 1}
            )
            response.raise_for_status()
            data = response.json()

            return data.get("hitCount", 0)

        except Exception:
            return 0

    def get_fulltext_url(self, item: Dict) -> Optional[str]:
        """
        Get best available fulltext URL.

        Priority: PMC PDF > Europe PMC > DOI
        """
        # Check for PMC PDF
        pmcid = item.get("pmcid")
        if pmcid:
            return f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmcid}&blobtype=pdf"

        # Check for OA location in fulltext URLs
        fulltext_urls = item.get("fullTextUrlList", {}).get("fullTextUrl", [])
        for url_info in fulltext_urls:
            if url_info.get("documentStyle") == "pdf":
                return url_info.get("url")

        # Fallback to DOI
        doi = item.get("doi")
        if doi:
            return f"https://doi.org/{doi}"

        return None

    def _parse_result(self, item: Dict) -> Optional[Dict]:
        """Parse Europe PMC result to standard format."""
        try:
            # Extract authors
            authors = []
            author_list = item.get("authorList", {}).get("author", [])
            for author in author_list:
                name = author.get("fullName", "")
                if name:
                    authors.append(name)

            # Parse date
            pub_date = None
            date_str = item.get("firstPublicationDate") or item.get("electronicPublicationDate")
            if date_str:
                try:
                    pub_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                except ValueError:
                    pass

            # Get citation count
            cited_by = item.get("citedByCount", 0)

            # Build result
            result = {
                "source": "europepmc",
                "external_id": item.get("pmid") or item.get("id", ""),
                "pmid": item.get("pmid"),
                "pmcid": item.get("pmcid"),
                "doi": item.get("doi"),
                "title": item.get("title", ""),
                "abstract": item.get("abstractText", ""),
                "authors": authors,
                "journal": item.get("journalTitle", ""),
                "published_at": pub_date,
                "citations": cited_by,
                "cited_by_count": cited_by,
                "url": f"https://europepmc.org/article/MED/{item.get('pmid')}" if item.get('pmid') else None,
                "oa_url": self.get_fulltext_url(item),
                "is_open_access": item.get("isOpenAccess") == "Y",
                "mesh_terms": [
                    mesh.get("descriptorName", "")
                    for mesh in item.get("meshHeadingList", {}).get("meshHeading", [])
                ],
                "keywords": item.get("keywordList", {}).get("keyword", []),
            }

            return result

        except Exception as e:
            logger.error(f"Failed to parse Europe PMC result: {e}")
            return None

    def close(self):
        """Close HTTP client."""
        self.client.close()


# Convenience function
def search_europepmc(query: str, since_days: int = 30, max_results: int = 50) -> List[Dict]:
    """Quick search function."""
    source = EuropePMCSource()
    try:
        return source.discover(query, since_days, max_results)
    finally:
        source.close()
