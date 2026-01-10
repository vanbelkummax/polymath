#!/usr/bin/env python3
"""
CourtListener Legal Intelligence MCP Server

Provides legal case law search, precedent discovery, and NLI-based
relevance verification using Free Law Project's CourtListener API.

API Documentation: https://www.courtlistener.com/help/api/
Rate Limits: 5000 requests/hour for authenticated users
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Any, Optional
from collections import Counter
import hashlib

import httpx

# Add parent paths for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mcp.server import Server
from mcp.types import Tool, TextContent

# CourtListener API configuration
COURTLISTENER_BASE_URL = "https://www.courtlistener.com/api/rest/v3"
COURTLISTENER_SEARCH_URL = "https://www.courtlistener.com/api/rest/v3/search/"

# Rate limiting
MAX_REQUESTS_PER_MINUTE = 80  # Conservative limit
REQUEST_DELAY = 60.0 / MAX_REQUESTS_PER_MINUTE

# Cache configuration
CACHE_DIR = os.path.expanduser("~/.cache/polymath/legal")
CACHE_EXPIRY_HOURS = 24

# Jurisdictions
FEDERAL_COURTS = [
    "scotus",  # Supreme Court
    "ca1", "ca2", "ca3", "ca4", "ca5", "ca6", "ca7", "ca8", "ca9", "ca10", "ca11", "cadc", "cafc",  # Circuit Courts
]

STATE_COURTS = [
    "cal", "calctapp",  # California
    "ny", "nyappdiv",   # New York
    "tex", "texapp",    # Texas
    "fla", "fladistctapp",  # Florida
    "ill", "illappct",  # Illinois
]


def get_cache_path(key: str) -> str:
    """Get cache file path for a key."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    hash_key = hashlib.md5(key.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{hash_key}.json")


def get_cached(key: str) -> Optional[dict]:
    """Get cached result if not expired."""
    cache_path = get_cache_path(key)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                cached = json.load(f)
            cached_time = datetime.fromisoformat(cached['cached_at'])
            if datetime.now() - cached_time < timedelta(hours=CACHE_EXPIRY_HOURS):
                return cached['data']
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
    return None


def set_cached(key: str, data: dict):
    """Cache result with timestamp."""
    cache_path = get_cache_path(key)
    with open(cache_path, 'w') as f:
        json.dump({
            'cached_at': datetime.now().isoformat(),
            'data': data
        }, f)


class CourtListenerClient:
    """Async client for CourtListener API."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("COURTLISTENER_API_KEY")
        self.last_request_time = 0
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Token {self.api_key}"
            self._client = httpx.AsyncClient(
                headers=headers,
                timeout=30.0,
                follow_redirects=True
            )
        return self._client

    async def _rate_limit(self):
        """Enforce rate limiting."""
        now = asyncio.get_event_loop().time()
        elapsed = now - self.last_request_time
        if elapsed < REQUEST_DELAY:
            await asyncio.sleep(REQUEST_DELAY - elapsed)
        self.last_request_time = asyncio.get_event_loop().time()

    async def search_opinions(
        self,
        query: str,
        court: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        per_page: int = 20,
        page: int = 1,
        order_by: str = "score desc"
    ) -> dict:
        """
        Search court opinions.

        Args:
            query: Search query (supports Lucene syntax)
            court: Court ID filter (e.g., 'scotus', 'ca9')
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)
            per_page: Results per page (max 100)
            order_by: Sort order ('score desc', 'dateFiled desc', etc.)
        """
        cache_key = f"opinions:{query}:{court}:{date_from}:{date_to}:{per_page}:{page}"
        cached = get_cached(cache_key)
        if cached:
            return cached

        await self._rate_limit()
        client = await self._get_client()

        params = {
            "q": query,
            "type": "o",  # opinions
            "order_by": order_by,
            "page": page,
        }

        if court:
            params["court"] = court
        if date_from:
            params["filed_after"] = date_from
        if date_to:
            params["filed_before"] = date_to

        try:
            resp = await client.get(COURTLISTENER_SEARCH_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

            # Transform results
            results = []
            for item in data.get("results", [])[:per_page]:
                results.append({
                    "id": item.get("id"),
                    "case_name": item.get("caseName"),
                    "court": item.get("court"),
                    "date_filed": item.get("dateFiled"),
                    "citation": item.get("citation", []),
                    "docket_number": item.get("docketNumber"),
                    "snippet": item.get("snippet", ""),
                    "absolute_url": f"https://www.courtlistener.com{item.get('absolute_url', '')}",
                    "score": item.get("score"),
                })

            result = {
                "count": data.get("count", 0),
                "results": results,
                "query": query,
                "filters": {
                    "court": court,
                    "date_from": date_from,
                    "date_to": date_to
                }
            }

            set_cached(cache_key, result)
            return result

        except httpx.HTTPError as e:
            return {"error": str(e), "query": query}

    async def get_opinion(self, opinion_id: int) -> dict:
        """Get full opinion details by ID."""
        cache_key = f"opinion:{opinion_id}"
        cached = get_cached(cache_key)
        if cached:
            return cached

        await self._rate_limit()
        client = await self._get_client()

        try:
            resp = await client.get(f"{COURTLISTENER_BASE_URL}/opinions/{opinion_id}/")
            resp.raise_for_status()
            data = resp.json()

            result = {
                "id": data.get("id"),
                "absolute_url": f"https://www.courtlistener.com{data.get('absolute_url', '')}",
                "cluster": data.get("cluster"),
                "author": data.get("author"),
                "type": data.get("type"),
                "plain_text": data.get("plain_text", "")[:10000],  # Truncate for context
                "html": data.get("html", "")[:5000] if data.get("html") else None,
                "extracted_by_ocr": data.get("extracted_by_ocr"),
                "date_created": data.get("date_created"),
            }

            set_cached(cache_key, result)
            return result

        except httpx.HTTPError as e:
            return {"error": str(e), "opinion_id": opinion_id}

    async def search_dockets(
        self,
        query: str,
        court: Optional[str] = None,
        per_page: int = 20
    ) -> dict:
        """Search dockets (case records)."""
        cache_key = f"dockets:{query}:{court}:{per_page}"
        cached = get_cached(cache_key)
        if cached:
            return cached

        await self._rate_limit()
        client = await self._get_client()

        params = {
            "q": query,
            "type": "d",  # dockets
        }
        if court:
            params["court"] = court

        try:
            resp = await client.get(COURTLISTENER_SEARCH_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

            results = []
            for item in data.get("results", [])[:per_page]:
                results.append({
                    "id": item.get("id"),
                    "case_name": item.get("caseName"),
                    "court": item.get("court"),
                    "docket_number": item.get("docketNumber"),
                    "date_filed": item.get("dateFiled"),
                    "date_terminated": item.get("dateTerminated"),
                    "absolute_url": f"https://www.courtlistener.com{item.get('absolute_url', '')}",
                })

            result = {
                "count": data.get("count", 0),
                "results": results,
                "query": query
            }

            set_cached(cache_key, result)
            return result

        except httpx.HTTPError as e:
            return {"error": str(e), "query": query}

    async def get_citation_network(
        self,
        opinion_id: int,
        depth: int = 1
    ) -> dict:
        """
        Get citation network for an opinion.

        Args:
            opinion_id: The opinion to analyze
            depth: How many levels of citations to traverse (1-2)
        """
        cache_key = f"citations:{opinion_id}:{depth}"
        cached = get_cached(cache_key)
        if cached:
            return cached

        await self._rate_limit()
        client = await self._get_client()

        try:
            # Get the opinion's citing opinions
            resp = await client.get(
                f"{COURTLISTENER_BASE_URL}/opinions/",
                params={"cites": opinion_id}
            )
            resp.raise_for_status()
            citing = resp.json()

            # Get opinions this one cites
            resp2 = await client.get(
                f"{COURTLISTENER_BASE_URL}/opinions/",
                params={"cited_by": opinion_id}
            )
            resp2.raise_for_status()
            cited_by = resp2.json()

            result = {
                "opinion_id": opinion_id,
                "citing_count": citing.get("count", 0),
                "citing_opinions": [
                    {
                        "id": o.get("id"),
                        "cluster": o.get("cluster"),
                        "type": o.get("type")
                    }
                    for o in citing.get("results", [])[:20]
                ],
                "cited_by_count": cited_by.get("count", 0),
                "cited_opinions": [
                    {
                        "id": o.get("id"),
                        "cluster": o.get("cluster"),
                        "type": o.get("type")
                    }
                    for o in cited_by.get("results", [])[:20]
                ]
            }

            set_cached(cache_key, result)
            return result

        except httpx.HTTPError as e:
            return {"error": str(e), "opinion_id": opinion_id}

    async def find_precedents(
        self,
        legal_issue: str,
        jurisdiction: str = "federal",
        years: int = 10
    ) -> dict:
        """
        Find precedent cases for a legal issue.

        Args:
            legal_issue: Description of the legal issue
            jurisdiction: 'federal', 'state', or specific court ID
            years: How many years back to search
        """
        date_from = (datetime.now() - timedelta(days=years*365)).strftime("%Y-%m-%d")

        # Determine courts to search
        if jurisdiction == "federal":
            courts = FEDERAL_COURTS
        elif jurisdiction == "state":
            courts = STATE_COURTS
        else:
            courts = [jurisdiction]

        all_results = []
        for court in courts[:5]:  # Limit to avoid too many requests
            result = await self.search_opinions(
                query=legal_issue,
                court=court,
                date_from=date_from,
                per_page=10,
                order_by="score desc"
            )
            if "results" in result:
                for r in result["results"]:
                    r["searched_court"] = court
                all_results.extend(result["results"])

        # Sort by score and deduplicate
        seen_ids = set()
        unique_results = []
        for r in sorted(all_results, key=lambda x: x.get("score", 0), reverse=True):
            if r["id"] not in seen_ids:
                seen_ids.add(r["id"])
                unique_results.append(r)

        return {
            "legal_issue": legal_issue,
            "jurisdiction": jurisdiction,
            "years_searched": years,
            "total_found": len(unique_results),
            "precedents": unique_results[:20]
        }

    async def analyze_jurisdiction_coverage(
        self,
        query: str,
        years: int = 5
    ) -> dict:
        """
        Analyze which jurisdictions have case law on a topic.

        Useful for understanding where precedents exist.
        """
        date_from = (datetime.now() - timedelta(days=years*365)).strftime("%Y-%m-%d")

        # Search across all major courts
        all_courts = FEDERAL_COURTS + STATE_COURTS
        coverage = {}

        for court in all_courts:
            result = await self.search_opinions(
                query=query,
                court=court,
                date_from=date_from,
                per_page=1  # Just need count
            )
            if "count" in result:
                coverage[court] = result["count"]

        # Analyze coverage
        total_cases = sum(coverage.values())
        top_courts = sorted(coverage.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "query": query,
            "years_analyzed": years,
            "total_cases": total_cases,
            "coverage_by_court": coverage,
            "top_jurisdictions": [
                {"court": court, "cases": count, "percentage": round(100*count/total_cases, 1) if total_cases > 0 else 0}
                for court, count in top_courts
            ],
            "federal_cases": sum(coverage.get(c, 0) for c in FEDERAL_COURTS),
            "state_cases": sum(coverage.get(c, 0) for c in STATE_COURTS)
        }


# MCP Server setup
server = Server("legal-intelligence")
client = CourtListenerClient()


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_cases",
            description="Search court opinions and case law using CourtListener. Supports Lucene query syntax.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (supports Lucene syntax: AND, OR, phrases in quotes)"
                    },
                    "court": {
                        "type": "string",
                        "description": "Court filter: 'scotus', 'ca9', 'cal', etc."
                    },
                    "date_from": {
                        "type": "string",
                        "description": "Start date (YYYY-MM-DD)"
                    },
                    "date_to": {
                        "type": "string",
                        "description": "End date (YYYY-MM-DD)"
                    },
                    "per_page": {
                        "type": "integer",
                        "description": "Results per page (default 20, max 100)"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_opinion_details",
            description="Get full details of a court opinion including the text.",
            inputSchema={
                "type": "object",
                "properties": {
                    "opinion_id": {
                        "type": "integer",
                        "description": "CourtListener opinion ID"
                    }
                },
                "required": ["opinion_id"]
            }
        ),
        Tool(
            name="find_precedents",
            description="Find precedent cases for a legal issue across jurisdictions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "legal_issue": {
                        "type": "string",
                        "description": "Description of the legal issue to find precedents for"
                    },
                    "jurisdiction": {
                        "type": "string",
                        "description": "'federal', 'state', or specific court ID (default: federal)"
                    },
                    "years": {
                        "type": "integer",
                        "description": "Years to search back (default: 10)"
                    }
                },
                "required": ["legal_issue"]
            }
        ),
        Tool(
            name="get_citation_network",
            description="Get the citation network for an opinion (what cites it, what it cites).",
            inputSchema={
                "type": "object",
                "properties": {
                    "opinion_id": {
                        "type": "integer",
                        "description": "CourtListener opinion ID"
                    },
                    "depth": {
                        "type": "integer",
                        "description": "Depth of citation traversal (1-2, default 1)"
                    }
                },
                "required": ["opinion_id"]
            }
        ),
        Tool(
            name="search_dockets",
            description="Search case dockets (procedural records) in CourtListener.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for dockets"
                    },
                    "court": {
                        "type": "string",
                        "description": "Court filter"
                    },
                    "per_page": {
                        "type": "integer",
                        "description": "Results per page (default 20)"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="analyze_jurisdiction_coverage",
            description="Analyze which courts/jurisdictions have case law on a topic.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Legal topic to analyze"
                    },
                    "years": {
                        "type": "integer",
                        "description": "Years to analyze (default: 5)"
                    }
                },
                "required": ["query"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        if name == "search_cases":
            result = await client.search_opinions(
                query=arguments["query"],
                court=arguments.get("court"),
                date_from=arguments.get("date_from"),
                date_to=arguments.get("date_to"),
                per_page=arguments.get("per_page", 20)
            )

        elif name == "get_opinion_details":
            result = await client.get_opinion(arguments["opinion_id"])

        elif name == "find_precedents":
            result = await client.find_precedents(
                legal_issue=arguments["legal_issue"],
                jurisdiction=arguments.get("jurisdiction", "federal"),
                years=arguments.get("years", 10)
            )

        elif name == "get_citation_network":
            result = await client.get_citation_network(
                opinion_id=arguments["opinion_id"],
                depth=arguments.get("depth", 1)
            )

        elif name == "search_dockets":
            result = await client.search_dockets(
                query=arguments["query"],
                court=arguments.get("court"),
                per_page=arguments.get("per_page", 20)
            )

        elif name == "analyze_jurisdiction_coverage":
            result = await client.analyze_jurisdiction_coverage(
                query=arguments["query"],
                years=arguments.get("years", 5)
            )

        else:
            result = {"error": f"Unknown tool: {name}"}

        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def main():
    from mcp.server.stdio import stdio_server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
