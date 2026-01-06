#!/usr/bin/env python3
"""
arXiv Source Connector

For bleeding-edge preprints. Speed is the key value here.
"""

import time
import re
from typing import List, Dict, Optional
from datetime import datetime
import logging
import xml.etree.ElementTree as ET

try:
    import httpx
except ImportError:
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "pip", "install", "httpx", "-q"])
    import httpx

from ..config import RATE_LIMITS

logger = logging.getLogger(__name__)

BASE_URL = "https://export.arxiv.org/api/query"

# arXiv categories relevant to your work
RELEVANT_CATEGORIES = {
    "cs.LG",      # Machine Learning
    "cs.CV",      # Computer Vision
    "cs.AI",      # Artificial Intelligence
    "cs.CL",      # Computation and Language
    "stat.ML",    # Statistics - Machine Learning
    "q-bio.QM",   # Quantitative Biology - Quantitative Methods
    "q-bio.GN",   # Quantitative Biology - Genomics
    "physics.bio-ph",  # Biological Physics
    "eess.IV",    # Image and Video Processing
}


class ArxivSource:
    """
    arXiv API connector.

    API Docs: https://arxiv.org/help/api
    Note: Rate limit is 1 request per 3 seconds
    """

    name = "arxiv"
    rate_limit = RATE_LIMITS.get("arxiv", 0.33)

    def __init__(self):
        self.client = httpx.Client(timeout=30)
        self.last_request = 0

    def _rate_limit(self):
        """Enforce rate limiting (arXiv is strict)."""
        elapsed = time.time() - self.last_request
        min_interval = 1.0 / self.rate_limit
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self.last_request = time.time()

    def discover(
        self,
        query: str,
        max_results: int = 50,
        categories: List[str] = None,
        sort_by: str = "submittedDate",
        sort_order: str = "descending"
    ) -> List[Dict]:
        """
        Search arXiv for papers.

        Args:
            query: Search query
            max_results: Maximum results
            categories: arXiv categories to search (e.g., ["cs.LG", "cs.CV"])
            sort_by: submittedDate, lastUpdatedDate, relevance
            sort_order: ascending, descending

        Returns:
            List of paper dicts
        """
        self._rate_limit()

        # Build query with categories
        search_query = f"all:{query}"
        if categories:
            cat_query = " OR ".join(f"cat:{cat}" for cat in categories)
            search_query = f"({search_query}) AND ({cat_query})"

        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": min(max_results, 100),
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }

        try:
            response = self.client.get(BASE_URL, params=params)
            response.raise_for_status()

            results = self._parse_atom_feed(response.text)
            logger.info(f"arXiv: Found {len(results)} results for '{query}'")
            return results

        except Exception as e:
            logger.error(f"arXiv search failed: {e}")
            return []

    def get_by_id(self, arxiv_id: str) -> Optional[Dict]:
        """Get paper by arXiv ID (e.g., '2301.00001')."""
        self._rate_limit()

        # Clean ID
        arxiv_id = arxiv_id.replace("arXiv:", "").strip()

        try:
            response = self.client.get(
                BASE_URL,
                params={"id_list": arxiv_id, "max_results": 1}
            )
            response.raise_for_status()

            results = self._parse_atom_feed(response.text)
            return results[0] if results else None

        except Exception as e:
            logger.error(f"arXiv ID lookup failed: {e}")
            return None

    def _parse_atom_feed(self, xml_content: str) -> List[Dict]:
        """Parse arXiv Atom feed."""
        results = []

        # Parse XML
        root = ET.fromstring(xml_content)

        # Namespace
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }

        for entry in root.findall("atom:entry", ns):
            try:
                parsed = self._parse_entry(entry, ns)
                if parsed:
                    results.append(parsed)
            except Exception as e:
                logger.error(f"Failed to parse arXiv entry: {e}")

        return results

    def _parse_entry(self, entry, ns: Dict) -> Optional[Dict]:
        """Parse single arXiv entry."""
        # Extract ID
        id_elem = entry.find("atom:id", ns)
        if id_elem is None:
            return None

        full_id = id_elem.text
        arxiv_id = full_id.split("/abs/")[-1] if "/abs/" in full_id else full_id

        # Extract title
        title_elem = entry.find("atom:title", ns)
        title = title_elem.text.strip().replace("\n", " ") if title_elem is not None else ""

        # Extract abstract
        summary_elem = entry.find("atom:summary", ns)
        abstract = summary_elem.text.strip().replace("\n", " ") if summary_elem is not None else ""

        # Extract authors
        authors = []
        for author in entry.findall("atom:author", ns):
            name_elem = author.find("atom:name", ns)
            if name_elem is not None:
                authors.append(name_elem.text)

        # Extract dates
        published_elem = entry.find("atom:published", ns)
        updated_elem = entry.find("atom:updated", ns)

        pub_date = None
        if published_elem is not None:
            try:
                pub_date = datetime.fromisoformat(
                    published_elem.text.replace("Z", "+00:00")
                ).date()
            except ValueError:
                pass

        # Extract categories
        categories = []
        for cat in entry.findall("arxiv:primary_category", ns):
            term = cat.get("term")
            if term:
                categories.append(term)
        for cat in entry.findall("atom:category", ns):
            term = cat.get("term")
            if term and term not in categories:
                categories.append(term)

        # Extract PDF link
        pdf_url = None
        for link in entry.findall("atom:link", ns):
            if link.get("title") == "pdf":
                pdf_url = link.get("href")
                break

        # DOI (if present)
        doi = None
        doi_elem = entry.find("arxiv:doi", ns)
        if doi_elem is not None:
            doi = doi_elem.text

        return {
            "source": "arxiv",
            "external_id": arxiv_id,
            "arxiv_id": arxiv_id,
            "doi": doi,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "published_at": pub_date,
            "categories": categories,
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "pdf_url": pdf_url or f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            "oa_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            "is_open_access": True,  # arXiv is always OA
            "citations": 0,  # arXiv doesn't provide citation counts
        }

    def close(self):
        """Close HTTP client."""
        self.client.close()


# Convenience function
def search_arxiv(query: str, max_results: int = 50, categories: List[str] = None) -> List[Dict]:
    """Quick search function."""
    source = ArxivSource()
    try:
        return source.discover(query, max_results, categories or list(RELEVANT_CATEGORIES))
    finally:
        source.close()
