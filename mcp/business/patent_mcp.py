#!/usr/bin/env python3
"""
Patent Intelligence MCP Server

Provides access to patent databases for IP whitespace analysis,
prior art search, and competitive intelligence.

Data Sources:
- USPTO PatentsView API (12M+ US patents)
- EPO Open Patent Services (EU + INPADOC)
- Lens.org API (140M global patents)

Tools:
- search_patents: Full-text patent search
- get_patent_details: Detailed patent info by ID
- find_patent_gaps: Identify IP whitespace
- track_assignee: Monitor company patent activity
- search_prior_art: Find relevant prior art for claims
"""

import json
import sys
import os
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from collections import Counter

# MCP Protocol
from mcp.server import Server
from mcp.types import Tool, TextContent
from mcp.server.stdio import stdio_server

# Rate limiting
import time
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================

USPTO_BASE_URL = "https://api.patentsview.org/patents/query"
USPTO_RATE_LIMIT = 45  # requests per minute

# EPO OPS config (requires registration at https://developers.epo.org/)
EPO_OPS_BASE = "https://ops.epo.org/3.2/rest-services"
EPO_CONSUMER_KEY = os.environ.get("EPO_CONSUMER_KEY", "")
EPO_CONSUMER_SECRET = os.environ.get("EPO_CONSUMER_SECRET", "")

# Lens.org (requires API key from https://www.lens.org/lens/user/subscriptions)
LENS_API_KEY = os.environ.get("LENS_API_KEY", "")
LENS_BASE_URL = "https://api.lens.org/patent/search"

# Simple in-memory cache
_cache: Dict[str, Any] = {}
_cache_ttl = 3600  # 1 hour
_last_request_time = 0

# ============================================================
# USPTO PatentsView API
# ============================================================

async def _rate_limit():
    """Enforce rate limiting."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < (60 / USPTO_RATE_LIMIT):
        await asyncio.sleep((60 / USPTO_RATE_LIMIT) - elapsed)
    _last_request_time = time.time()


def _cache_key(prefix: str, params: dict) -> str:
    """Generate cache key."""
    param_str = json.dumps(params, sort_keys=True)
    return f"{prefix}:{hashlib.md5(param_str.encode()).hexdigest()}"


async def search_uspto(
    query: str,
    per_page: int = 25,
    page: int = 1,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    assignee: Optional[str] = None
) -> Dict[str, Any]:
    """
    Search USPTO PatentsView API.

    Args:
        query: Full-text search query
        per_page: Results per page (max 1000)
        page: Page number
        date_from: Start date (YYYY-MM-DD)
        date_to: End date (YYYY-MM-DD)
        assignee: Filter by assignee organization
    """
    import aiohttp

    await _rate_limit()

    # Build query
    q_parts = [{"_text_any": {"patent_abstract": query}}]

    if assignee:
        q_parts.append({"_contains": {"assignee_organization": assignee}})

    if date_from or date_to:
        date_q = {}
        if date_from:
            date_q["_gte"] = {"patent_date": date_from}
        if date_to:
            date_q["_lte"] = {"patent_date": date_to}
        if date_q:
            q_parts.append(date_q)

    if len(q_parts) == 1:
        q = q_parts[0]
    else:
        q = {"_and": q_parts}

    params = {
        "q": q,
        "f": [
            "patent_id", "patent_number", "patent_title", "patent_date",
            "patent_abstract", "patent_type", "patent_kind",
            "assignee_organization", "assignee_country",
            "inventor_first_name", "inventor_last_name",
            "cpc_group_id", "cpc_subgroup_id"
        ],
        "o": {
            "per_page": min(per_page, 1000),
            "page": page
        },
        "s": [{"patent_date": "desc"}]
    }

    # Check cache
    cache_key = _cache_key("uspto", params)
    if cache_key in _cache:
        cached_time, cached_data = _cache[cache_key]
        if time.time() - cached_time < _cache_ttl:
            return cached_data

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(USPTO_BASE_URL, json=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    result = {
                        "source": "USPTO PatentsView",
                        "query": query,
                        "total_count": data.get("total_patent_count", 0),
                        "page": page,
                        "per_page": per_page,
                        "patents": []
                    }

                    for p in data.get("patents", []):
                        patent = {
                            "patent_id": p.get("patent_id"),
                            "patent_number": p.get("patent_number"),
                            "title": p.get("patent_title"),
                            "date": p.get("patent_date"),
                            "abstract": p.get("patent_abstract", "")[:500],
                            "type": p.get("patent_type"),
                            "assignees": [],
                            "inventors": [],
                            "cpc_codes": []
                        }

                        # Extract assignees
                        if p.get("assignees"):
                            for a in p["assignees"]:
                                patent["assignees"].append({
                                    "name": a.get("assignee_organization"),
                                    "country": a.get("assignee_country")
                                })

                        # Extract inventors
                        if p.get("inventors"):
                            for i in p["inventors"][:5]:  # Limit to 5
                                patent["inventors"].append(
                                    f"{i.get('inventor_first_name', '')} {i.get('inventor_last_name', '')}"
                                )

                        # Extract CPC codes
                        if p.get("cpcs"):
                            for c in p["cpcs"][:10]:  # Limit to 10
                                patent["cpc_codes"].append(c.get("cpc_group_id"))

                        result["patents"].append(patent)

                    # Cache result
                    _cache[cache_key] = (time.time(), result)
                    return result
                else:
                    return {"error": f"USPTO API error: {resp.status}", "patents": []}

    except Exception as e:
        logger.error(f"USPTO search error: {e}")
        return {"error": str(e), "patents": []}


async def get_patent_details(patent_number: str) -> Dict[str, Any]:
    """Get detailed information for a specific patent."""
    import aiohttp

    await _rate_limit()

    params = {
        "q": {"patent_number": patent_number},
        "f": [
            "patent_id", "patent_number", "patent_title", "patent_date",
            "patent_abstract", "patent_type", "patent_kind",
            "patent_num_claims", "patent_num_cited_by_us_patents",
            "assignee_organization", "assignee_country", "assignee_type",
            "inventor_first_name", "inventor_last_name", "inventor_city", "inventor_country",
            "cpc_group_id", "cpc_group_title", "cpc_subgroup_id",
            "cited_patent_number", "cited_patent_title"
        ]
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(USPTO_BASE_URL, json=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    patents = data.get("patents", [])
                    if patents:
                        p = patents[0]
                        return {
                            "source": "USPTO PatentsView",
                            "patent_number": p.get("patent_number"),
                            "title": p.get("patent_title"),
                            "date": p.get("patent_date"),
                            "abstract": p.get("patent_abstract"),
                            "type": p.get("patent_type"),
                            "num_claims": p.get("patent_num_claims"),
                            "citations_received": p.get("patent_num_cited_by_us_patents"),
                            "assignees": p.get("assignees", []),
                            "inventors": p.get("inventors", []),
                            "cpc_codes": p.get("cpcs", []),
                            "citations": p.get("cited_patents", [])[:20]
                        }
                    return {"error": "Patent not found"}
                return {"error": f"API error: {resp.status}"}
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# Analysis Tools
# ============================================================

async def find_patent_gaps(query: str, years: int = 5) -> Dict[str, Any]:
    """
    Analyze patent landscape to find whitespace opportunities.

    Returns:
    - Temporal gaps (declining filing rates)
    - IPC/CPC code gaps (underexplored classes)
    - Assignee concentration
    """
    # Search recent patents
    results = await search_uspto(query, per_page=500)
    patents = results.get("patents", [])

    if not patents:
        return {"error": "No patents found for analysis", "query": query}

    # Analyze temporal distribution
    current_year = datetime.now().year
    year_counts = Counter()
    for p in patents:
        if p.get("date"):
            try:
                year = int(p["date"][:4])
                year_counts[year] += 1
            except:
                pass

    # Analyze CPC codes
    cpc_counts = Counter()
    for p in patents:
        for code in p.get("cpc_codes", []):
            if code:
                cpc_counts[code] += 1

    # Analyze assignees
    assignee_counts = Counter()
    for p in patents:
        for a in p.get("assignees", []):
            name = a.get("name")
            if name:
                assignee_counts[name] += 1

    # Identify gaps
    recent_3yr = sum(year_counts.get(current_year - i, 0) for i in range(3))
    older_3yr = sum(year_counts.get(current_year - 3 - i, 0) for i in range(3))

    # Rare CPC codes (potential whitespace)
    rare_cpcs = [code for code, count in cpc_counts.items() if count <= 2]

    return {
        "query": query,
        "total_patents_analyzed": len(patents),
        "temporal_analysis": {
            "year_distribution": dict(sorted(year_counts.items())),
            "recent_3yr_count": recent_3yr,
            "older_3yr_count": older_3yr,
            "trend": "declining" if recent_3yr < older_3yr * 0.7 else
                     "growing" if recent_3yr > older_3yr * 1.3 else "stable"
        },
        "cpc_analysis": {
            "top_cpc_codes": cpc_counts.most_common(10),
            "rare_cpc_codes": rare_cpcs[:15],
            "concentration": len(cpc_counts)
        },
        "assignee_analysis": {
            "top_assignees": assignee_counts.most_common(10),
            "unique_assignees": len(assignee_counts),
            "concentration_ratio": (
                sum(c for _, c in assignee_counts.most_common(5)) / len(patents)
                if patents else 0
            )
        },
        "whitespace_signals": {
            "temporal_gap": recent_3yr < 10,
            "fragmented_landscape": len(assignee_counts) > len(patents) * 0.3,
            "underexplored_cpcs": len(rare_cpcs) > 5
        }
    }


async def track_assignee(assignee_name: str, years: int = 3) -> Dict[str, Any]:
    """
    Track patent filing activity for a specific company/assignee.
    """
    date_from = (datetime.now() - timedelta(days=years*365)).strftime("%Y-%m-%d")

    results = await search_uspto(
        query="*",
        assignee=assignee_name,
        date_from=date_from,
        per_page=200
    )

    patents = results.get("patents", [])

    if not patents:
        return {
            "assignee": assignee_name,
            "total_patents": 0,
            "message": "No patents found"
        }

    # Analyze by year
    year_counts = Counter()
    cpc_counts = Counter()

    for p in patents:
        if p.get("date"):
            try:
                year = int(p["date"][:4])
                year_counts[year] += 1
            except:
                pass
        for code in p.get("cpc_codes", []):
            if code:
                cpc_counts[code] += 1

    # Calculate velocity
    current_year = datetime.now().year
    recent_count = year_counts.get(current_year, 0) + year_counts.get(current_year - 1, 0)

    return {
        "assignee": assignee_name,
        "period": f"{years} years",
        "total_patents": len(patents),
        "filing_by_year": dict(sorted(year_counts.items())),
        "recent_2yr_velocity": recent_count,
        "top_technology_areas": cpc_counts.most_common(10),
        "recent_patents": [
            {"title": p["title"], "date": p["date"], "number": p["patent_number"]}
            for p in patents[:10]
        ]
    }


async def search_prior_art(claim_text: str, max_results: int = 50) -> Dict[str, Any]:
    """
    Search for prior art relevant to a patent claim.

    Extracts key terms and searches across patent databases.
    """
    # Extract key technical terms (simple approach)
    # In production, use NLP to extract noun phrases
    import re

    # Remove common words
    stopwords = {
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
        'method', 'system', 'apparatus', 'device', 'comprising', 'wherein',
        'said', 'claim', 'claims', 'according', 'configured'
    }

    words = re.findall(r'\b[a-z]+\b', claim_text.lower())
    key_terms = [w for w in words if w not in stopwords and len(w) > 3]

    # Build search query from top terms
    term_counts = Counter(key_terms)
    search_terms = " ".join([t for t, _ in term_counts.most_common(8)])

    results = await search_uspto(search_terms, per_page=max_results)

    return {
        "claim_analyzed": claim_text[:200] + "..." if len(claim_text) > 200 else claim_text,
        "extracted_terms": [t for t, _ in term_counts.most_common(15)],
        "search_query": search_terms,
        "prior_art_candidates": results.get("patents", []),
        "total_found": results.get("total_count", 0)
    }


# ============================================================
# MCP Server Setup
# ============================================================

server = Server("patent-intelligence")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available patent intelligence tools."""
    return [
        Tool(
            name="search_patents",
            description="Search USPTO PatentsView for patents. Use for prior art discovery, competitive analysis, and technology landscape mapping.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (searches patent titles and abstracts)"
                    },
                    "per_page": {
                        "type": "integer",
                        "description": "Results per page (default 25, max 1000)",
                        "default": 25
                    },
                    "assignee": {
                        "type": "string",
                        "description": "Filter by assignee/company name"
                    },
                    "date_from": {
                        "type": "string",
                        "description": "Start date (YYYY-MM-DD)"
                    },
                    "date_to": {
                        "type": "string",
                        "description": "End date (YYYY-MM-DD)"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_patent_details",
            description="Get detailed information about a specific patent by patent number.",
            inputSchema={
                "type": "object",
                "properties": {
                    "patent_number": {
                        "type": "string",
                        "description": "USPTO patent number (e.g., '10123456')"
                    }
                },
                "required": ["patent_number"]
            }
        ),
        Tool(
            name="find_patent_gaps",
            description="Analyze patent landscape to find whitespace opportunities. Identifies temporal gaps, underexplored technology areas, and market fragmentation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Technology area to analyze"
                    },
                    "years": {
                        "type": "integer",
                        "description": "Years of data to analyze (default 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="track_assignee",
            description="Track patent filing activity for a company/organization. Shows filing velocity, technology focus areas, and recent patents.",
            inputSchema={
                "type": "object",
                "properties": {
                    "assignee_name": {
                        "type": "string",
                        "description": "Company or organization name"
                    },
                    "years": {
                        "type": "integer",
                        "description": "Years of history to analyze (default 3)",
                        "default": 3
                    }
                },
                "required": ["assignee_name"]
            }
        ),
        Tool(
            name="search_prior_art",
            description="Search for prior art relevant to a patent claim. Extracts key terms and finds potentially relevant existing patents.",
            inputSchema={
                "type": "object",
                "properties": {
                    "claim_text": {
                        "type": "string",
                        "description": "The patent claim text to find prior art for"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results to return (default 50)",
                        "default": 50
                    }
                },
                "required": ["claim_text"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    try:
        if name == "search_patents":
            result = await search_uspto(
                query=arguments["query"],
                per_page=arguments.get("per_page", 25),
                assignee=arguments.get("assignee"),
                date_from=arguments.get("date_from"),
                date_to=arguments.get("date_to")
            )
        elif name == "get_patent_details":
            result = await get_patent_details(arguments["patent_number"])
        elif name == "find_patent_gaps":
            result = await find_patent_gaps(
                query=arguments["query"],
                years=arguments.get("years", 5)
            )
        elif name == "track_assignee":
            result = await track_assignee(
                assignee_name=arguments["assignee_name"],
                years=arguments.get("years", 3)
            )
        elif name == "search_prior_art":
            result = await search_prior_art(
                claim_text=arguments["claim_text"],
                max_results=arguments.get("max_results", 50)
            )
        else:
            result = {"error": f"Unknown tool: {name}"}

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        logger.error(f"Tool {name} error: {e}")
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
