#!/usr/bin/env python3
"""
OpenCorporates Corporate Data MCP Server

Provides company lookup, corporate structure, and officer information
using the OpenCorporates API (235M+ companies worldwide).

API Documentation: https://api.opencorporates.com/documentation
Rate Limits: Varies by plan (free tier available for public benefit)
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Any, Optional
import hashlib

import httpx

# Add parent paths for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mcp.server import Server
from mcp.types import Tool, TextContent

# OpenCorporates API configuration
OPENCORPORATES_BASE_URL = "https://api.opencorporates.com/v0.4"

# Rate limiting (conservative for free tier)
MAX_REQUESTS_PER_MINUTE = 10
REQUEST_DELAY = 60.0 / MAX_REQUESTS_PER_MINUTE

# Cache configuration
CACHE_DIR = os.path.expanduser("~/.cache/polymath/corporate")
CACHE_EXPIRY_HOURS = 72  # Corporate data changes slowly

# Common jurisdiction codes
JURISDICTIONS = {
    "us": ["us_de", "us_ca", "us_ny", "us_tx", "us_fl", "us_nv", "us_il"],
    "uk": ["gb"],
    "eu": ["de", "fr", "nl", "ie", "lu"],
    "asia": ["hk", "sg", "jp"],
}


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


class OpenCorporatesClient:
    """Async client for OpenCorporates API."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENCORPORATES_API_KEY")
        self.last_request_time = 0
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
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

    def _build_url(self, endpoint: str, **params) -> str:
        """Build URL with optional API key."""
        url = f"{OPENCORPORATES_BASE_URL}/{endpoint}"
        if self.api_key:
            params["api_token"] = self.api_key
        if params:
            query = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
            url = f"{url}?{query}"
        return url

    async def search_companies(
        self,
        query: str,
        jurisdiction: Optional[str] = None,
        company_type: Optional[str] = None,
        status: Optional[str] = None,
        per_page: int = 30,
        page: int = 1
    ) -> dict:
        """
        Search for companies by name or other criteria.

        Args:
            query: Company name or search term
            jurisdiction: Jurisdiction code (e.g., 'us_de', 'gb')
            company_type: Filter by company type
            status: Company status ('active', 'dissolved', etc.)
            per_page: Results per page (max 100)
            page: Page number
        """
        cache_key = f"search:{query}:{jurisdiction}:{company_type}:{status}:{per_page}:{page}"
        cached = get_cached(cache_key)
        if cached:
            return cached

        await self._rate_limit()
        client = await self._get_client()

        params = {
            "q": query,
            "per_page": per_page,
            "page": page
        }
        if jurisdiction:
            params["jurisdiction_code"] = jurisdiction
        if company_type:
            params["company_type"] = company_type
        if status:
            params["current_status"] = status

        url = self._build_url("companies/search", **params)

        try:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

            companies = data.get("results", {}).get("companies", [])
            result = {
                "total_count": data.get("results", {}).get("total_count", 0),
                "page": data.get("results", {}).get("page", 1),
                "per_page": data.get("results", {}).get("per_page", 30),
                "companies": [
                    {
                        "company_number": c["company"]["company_number"],
                        "name": c["company"]["name"],
                        "jurisdiction_code": c["company"]["jurisdiction_code"],
                        "incorporation_date": c["company"].get("incorporation_date"),
                        "dissolution_date": c["company"].get("dissolution_date"),
                        "company_type": c["company"].get("company_type"),
                        "current_status": c["company"].get("current_status"),
                        "registered_address": c["company"].get("registered_address_in_full"),
                        "opencorporates_url": c["company"].get("opencorporates_url"),
                    }
                    for c in companies
                ],
                "query": query
            }

            set_cached(cache_key, result)
            return result

        except httpx.HTTPError as e:
            return {"error": str(e), "query": query}

    async def get_company(
        self,
        jurisdiction: str,
        company_number: str
    ) -> dict:
        """
        Get detailed company information.

        Args:
            jurisdiction: Jurisdiction code (e.g., 'us_de', 'gb')
            company_number: Company registration number
        """
        cache_key = f"company:{jurisdiction}:{company_number}"
        cached = get_cached(cache_key)
        if cached:
            return cached

        await self._rate_limit()
        client = await self._get_client()

        url = self._build_url(f"companies/{jurisdiction}/{company_number}")

        try:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

            company = data.get("results", {}).get("company", {})
            result = {
                "company_number": company.get("company_number"),
                "name": company.get("name"),
                "jurisdiction_code": company.get("jurisdiction_code"),
                "incorporation_date": company.get("incorporation_date"),
                "dissolution_date": company.get("dissolution_date"),
                "company_type": company.get("company_type"),
                "registry_url": company.get("registry_url"),
                "current_status": company.get("current_status"),
                "registered_address": company.get("registered_address_in_full"),
                "agent_name": company.get("agent_name"),
                "agent_address": company.get("agent_address"),
                "previous_names": company.get("previous_names", []),
                "alternative_names": company.get("alternative_names", []),
                "branch": company.get("branch"),
                "branch_status": company.get("branch_status"),
                "home_company": company.get("home_company"),
                "industry_codes": company.get("industry_codes", []),
                "opencorporates_url": company.get("opencorporates_url"),
                "source": company.get("source", {}),
            }

            set_cached(cache_key, result)
            return result

        except httpx.HTTPError as e:
            return {"error": str(e), "jurisdiction": jurisdiction, "company_number": company_number}

    async def get_officers(
        self,
        jurisdiction: str,
        company_number: str
    ) -> dict:
        """
        Get officers/directors for a company.
        """
        cache_key = f"officers:{jurisdiction}:{company_number}"
        cached = get_cached(cache_key)
        if cached:
            return cached

        await self._rate_limit()
        client = await self._get_client()

        url = self._build_url(f"companies/{jurisdiction}/{company_number}/officers")

        try:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

            officers = data.get("results", {}).get("officers", [])
            result = {
                "company": {
                    "jurisdiction": jurisdiction,
                    "company_number": company_number
                },
                "total_count": len(officers),
                "officers": [
                    {
                        "id": o["officer"].get("id"),
                        "name": o["officer"].get("name"),
                        "position": o["officer"].get("position"),
                        "start_date": o["officer"].get("start_date"),
                        "end_date": o["officer"].get("end_date"),
                        "nationality": o["officer"].get("nationality"),
                        "occupation": o["officer"].get("occupation"),
                        "address": o["officer"].get("address"),
                        "current": o["officer"].get("end_date") is None
                    }
                    for o in officers
                ]
            }

            set_cached(cache_key, result)
            return result

        except httpx.HTTPError as e:
            return {"error": str(e), "jurisdiction": jurisdiction, "company_number": company_number}

    async def search_officers(
        self,
        query: str,
        jurisdiction: Optional[str] = None,
        per_page: int = 30,
        page: int = 1
    ) -> dict:
        """
        Search for corporate officers by name.

        Args:
            query: Officer name
            jurisdiction: Optional jurisdiction filter
            per_page: Results per page
            page: Page number
        """
        cache_key = f"officer_search:{query}:{jurisdiction}:{per_page}:{page}"
        cached = get_cached(cache_key)
        if cached:
            return cached

        await self._rate_limit()
        client = await self._get_client()

        params = {
            "q": query,
            "per_page": per_page,
            "page": page
        }
        if jurisdiction:
            params["jurisdiction_code"] = jurisdiction

        url = self._build_url("officers/search", **params)

        try:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

            officers = data.get("results", {}).get("officers", [])
            result = {
                "total_count": data.get("results", {}).get("total_count", 0),
                "page": page,
                "per_page": per_page,
                "officers": [
                    {
                        "id": o["officer"].get("id"),
                        "name": o["officer"].get("name"),
                        "position": o["officer"].get("position"),
                        "company_name": o["officer"].get("company", {}).get("name"),
                        "company_number": o["officer"].get("company", {}).get("company_number"),
                        "jurisdiction": o["officer"].get("company", {}).get("jurisdiction_code"),
                        "start_date": o["officer"].get("start_date"),
                        "end_date": o["officer"].get("end_date"),
                        "opencorporates_url": o["officer"].get("opencorporates_url"),
                    }
                    for o in officers
                ],
                "query": query
            }

            set_cached(cache_key, result)
            return result

        except httpx.HTTPError as e:
            return {"error": str(e), "query": query}

    async def get_filings(
        self,
        jurisdiction: str,
        company_number: str
    ) -> dict:
        """
        Get regulatory filings for a company.
        """
        cache_key = f"filings:{jurisdiction}:{company_number}"
        cached = get_cached(cache_key)
        if cached:
            return cached

        await self._rate_limit()
        client = await self._get_client()

        url = self._build_url(f"companies/{jurisdiction}/{company_number}/filings")

        try:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

            filings = data.get("results", {}).get("filings", [])
            result = {
                "company": {
                    "jurisdiction": jurisdiction,
                    "company_number": company_number
                },
                "total_count": len(filings),
                "filings": [
                    {
                        "title": f["filing"].get("title"),
                        "date": f["filing"].get("date"),
                        "filing_type": f["filing"].get("filing_type"),
                        "url": f["filing"].get("url"),
                        "opencorporates_url": f["filing"].get("opencorporates_url"),
                    }
                    for f in filings[:50]  # Limit to avoid huge responses
                ]
            }

            set_cached(cache_key, result)
            return result

        except httpx.HTTPError as e:
            return {"error": str(e), "jurisdiction": jurisdiction, "company_number": company_number}

    async def get_corporate_groupings(
        self,
        query: str,
        per_page: int = 20
    ) -> dict:
        """
        Search for corporate groupings (parent companies, subsidiaries).
        """
        cache_key = f"groupings:{query}:{per_page}"
        cached = get_cached(cache_key)
        if cached:
            return cached

        await self._rate_limit()
        client = await self._get_client()

        params = {"q": query, "per_page": per_page}
        url = self._build_url("corporate_groupings/search", **params)

        try:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

            groupings = data.get("results", {}).get("corporate_groupings", [])
            result = {
                "total_count": data.get("results", {}).get("total_count", 0),
                "groupings": [
                    {
                        "name": g["corporate_grouping"].get("name"),
                        "wikipedia_id": g["corporate_grouping"].get("wikipedia_id"),
                        "companies_count": g["corporate_grouping"].get("companies_count"),
                        "opencorporates_url": g["corporate_grouping"].get("opencorporates_url"),
                    }
                    for g in groupings
                ],
                "query": query
            }

            set_cached(cache_key, result)
            return result

        except httpx.HTTPError as e:
            return {"error": str(e), "query": query}

    async def analyze_jurisdiction_presence(
        self,
        company_name: str
    ) -> dict:
        """
        Analyze a company's presence across jurisdictions.

        Useful for understanding corporate structure and tax planning.
        """
        all_results = []

        # Search in major jurisdictions
        for region, codes in JURISDICTIONS.items():
            for code in codes[:3]:  # Limit searches
                result = await self.search_companies(
                    query=company_name,
                    jurisdiction=code,
                    per_page=5
                )
                if "companies" in result:
                    for c in result["companies"]:
                        c["region"] = region
                    all_results.extend(result["companies"])

        # Analyze presence
        jurisdictions_found = list(set(c["jurisdiction_code"] for c in all_results))
        status_counts = {}
        for c in all_results:
            status = c.get("current_status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "company_name": company_name,
            "total_entities": len(all_results),
            "jurisdictions": jurisdictions_found,
            "jurisdiction_count": len(jurisdictions_found),
            "status_breakdown": status_counts,
            "entities": all_results[:20],
            "analysis": {
                "has_delaware": "us_de" in jurisdictions_found,
                "has_uk": "gb" in jurisdictions_found,
                "has_offshore": any(j in jurisdictions_found for j in ["vg", "ky", "bm"]),
                "multi_jurisdiction": len(jurisdictions_found) > 1
            }
        }


# MCP Server setup
server = Server("corporate-intelligence")
client = OpenCorporatesClient()


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_companies",
            description="Search for companies by name in OpenCorporates (235M+ companies worldwide).",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Company name or search term"
                    },
                    "jurisdiction": {
                        "type": "string",
                        "description": "Jurisdiction code (e.g., 'us_de' for Delaware, 'gb' for UK)"
                    },
                    "status": {
                        "type": "string",
                        "description": "Company status filter ('active', 'dissolved')"
                    },
                    "per_page": {
                        "type": "integer",
                        "description": "Results per page (default 30, max 100)"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_company_details",
            description="Get detailed information about a specific company.",
            inputSchema={
                "type": "object",
                "properties": {
                    "jurisdiction": {
                        "type": "string",
                        "description": "Jurisdiction code (e.g., 'us_de')"
                    },
                    "company_number": {
                        "type": "string",
                        "description": "Company registration number"
                    }
                },
                "required": ["jurisdiction", "company_number"]
            }
        ),
        Tool(
            name="get_company_officers",
            description="Get officers and directors for a company.",
            inputSchema={
                "type": "object",
                "properties": {
                    "jurisdiction": {
                        "type": "string",
                        "description": "Jurisdiction code"
                    },
                    "company_number": {
                        "type": "string",
                        "description": "Company registration number"
                    }
                },
                "required": ["jurisdiction", "company_number"]
            }
        ),
        Tool(
            name="search_officers",
            description="Search for corporate officers/directors by name.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Officer name to search"
                    },
                    "jurisdiction": {
                        "type": "string",
                        "description": "Optional jurisdiction filter"
                    },
                    "per_page": {
                        "type": "integer",
                        "description": "Results per page (default 30)"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_company_filings",
            description="Get regulatory filings for a company.",
            inputSchema={
                "type": "object",
                "properties": {
                    "jurisdiction": {
                        "type": "string",
                        "description": "Jurisdiction code"
                    },
                    "company_number": {
                        "type": "string",
                        "description": "Company registration number"
                    }
                },
                "required": ["jurisdiction", "company_number"]
            }
        ),
        Tool(
            name="search_corporate_groups",
            description="Search for corporate groupings (parent/subsidiary relationships).",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Corporate group name"
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
            name="analyze_jurisdiction_presence",
            description="Analyze a company's presence across multiple jurisdictions (useful for corporate structure analysis).",
            inputSchema={
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "Company name to analyze"
                    }
                },
                "required": ["company_name"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        if name == "search_companies":
            result = await client.search_companies(
                query=arguments["query"],
                jurisdiction=arguments.get("jurisdiction"),
                status=arguments.get("status"),
                per_page=arguments.get("per_page", 30)
            )

        elif name == "get_company_details":
            result = await client.get_company(
                jurisdiction=arguments["jurisdiction"],
                company_number=arguments["company_number"]
            )

        elif name == "get_company_officers":
            result = await client.get_officers(
                jurisdiction=arguments["jurisdiction"],
                company_number=arguments["company_number"]
            )

        elif name == "search_officers":
            result = await client.search_officers(
                query=arguments["query"],
                jurisdiction=arguments.get("jurisdiction"),
                per_page=arguments.get("per_page", 30)
            )

        elif name == "get_company_filings":
            result = await client.get_filings(
                jurisdiction=arguments["jurisdiction"],
                company_number=arguments["company_number"]
            )

        elif name == "search_corporate_groups":
            result = await client.get_corporate_groupings(
                query=arguments["query"],
                per_page=arguments.get("per_page", 20)
            )

        elif name == "analyze_jurisdiction_presence":
            result = await client.analyze_jurisdiction_presence(
                company_name=arguments["company_name"]
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
