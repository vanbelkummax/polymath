#!/usr/bin/env python3
"""
NIH Reporter MCP Server

Provides access to NIH grant data for:
- Finding funded research in your area
- Competitive intelligence (who's funded to do what)
- Identifying potential collaborators
- Tracking funding trends

API: https://api.reporter.nih.gov/ (public, no auth required)
"""

import json
import logging
import sys
from datetime import datetime
from typing import Any, Optional
import httpx

# MCP SDK
from mcp.server import Server
from mcp.types import Tool, TextContent
from mcp.server.stdio import stdio_server

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# NIH Reporter API base
NIH_REPORTER_API = "https://api.reporter.nih.gov/v2"

# Rate limiting
import asyncio
_last_request = 0
_min_delay = 0.2  # 200ms between requests


async def rate_limit():
    """Enforce rate limiting"""
    global _last_request
    now = asyncio.get_event_loop().time()
    elapsed = now - _last_request
    if elapsed < _min_delay:
        await asyncio.sleep(_min_delay - elapsed)
    _last_request = asyncio.get_event_loop().time()


async def nih_api_request(endpoint: str, payload: dict) -> dict:
    """Make request to NIH Reporter API"""
    await rate_limit()

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{NIH_REPORTER_API}/{endpoint}",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()


def format_grant(grant: dict) -> str:
    """Format a grant for display"""
    lines = []

    # Core info
    project_num = grant.get("project_num", "N/A")
    title = grant.get("project_title", "No title")
    pi_names = ", ".join([
        f"{pi.get('first_name', '')} {pi.get('last_name', '')}"
        for pi in grant.get("principal_investigators", [])
    ])
    org = grant.get("organization", {}).get("org_name", "Unknown")

    lines.append(f"**{project_num}**")
    lines.append(f"Title: {title}")
    lines.append(f"PI(s): {pi_names}")
    lines.append(f"Organization: {org}")

    # Funding
    award_amount = grant.get("award_amount")
    if award_amount:
        lines.append(f"Award: ${award_amount:,.0f}")

    fiscal_year = grant.get("fiscal_year")
    if fiscal_year:
        lines.append(f"Fiscal Year: {fiscal_year}")

    # Dates
    start_date = grant.get("project_start_date")
    end_date = grant.get("project_end_date")
    if start_date and end_date:
        lines.append(f"Period: {start_date[:10]} to {end_date[:10]}")

    # Abstract
    abstract = grant.get("abstract_text", "")
    if abstract:
        # Truncate for display
        if len(abstract) > 500:
            abstract = abstract[:500] + "..."
        lines.append(f"\nAbstract: {abstract}")

    # Terms
    terms = grant.get("terms", "")
    if terms:
        lines.append(f"Keywords: {terms[:200]}")

    return "\n".join(lines)


# Create MCP server
server = Server("nih-reporter")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools"""
    return [
        Tool(
            name="search_grants",
            description="""Search NIH grants by keywords, PI name, organization, or other criteria.

Use cases:
- Find funded research in your area
- Competitive intelligence (who's funded to do what)
- Identify potential collaborators
- Track funding trends

Examples:
- search_grants(text="spatial transcriptomics")
- search_grants(pi_name="Lau, Ken")
- search_grants(org_name="Vanderbilt", text="cancer")
- search_grants(text="foundation model", fiscal_years=[2024, 2025])""",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Free text search (searches title, abstract, terms)"
                    },
                    "pi_name": {
                        "type": "string",
                        "description": "Principal investigator name (e.g., 'Smith, John')"
                    },
                    "org_name": {
                        "type": "string",
                        "description": "Organization name (e.g., 'Vanderbilt')"
                    },
                    "fiscal_years": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Fiscal years to search (default: last 3 years)"
                    },
                    "activity_codes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Activity codes (e.g., ['R01', 'R21', 'U01'])"
                    },
                    "include_active_only": {
                        "type": "boolean",
                        "description": "Only include currently active projects (default: true)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default: 25, max: 100)"
                    }
                }
            }
        ),
        Tool(
            name="get_grant_details",
            description="""Get full details for a specific grant by project number.

Example: get_grant_details(project_num="R01CA123456")""",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_num": {
                        "type": "string",
                        "description": "NIH project number (e.g., 'R01CA123456')"
                    }
                },
                "required": ["project_num"]
            }
        ),
        Tool(
            name="get_pi_grants",
            description="""Get all grants for a specific PI.

Example: get_pi_grants(pi_name="Lau, Ken", org_name="Vanderbilt")""",
            inputSchema={
                "type": "object",
                "properties": {
                    "pi_name": {
                        "type": "string",
                        "description": "PI name (e.g., 'Lau, Ken' or 'Ken Lau')"
                    },
                    "org_name": {
                        "type": "string",
                        "description": "Organization name to disambiguate"
                    },
                    "include_past": {
                        "type": "boolean",
                        "description": "Include past (inactive) grants (default: false)"
                    }
                },
                "required": ["pi_name"]
            }
        ),
        Tool(
            name="funding_trends",
            description="""Analyze funding trends for a topic over time.

Returns yearly funding totals and grant counts.

Example: funding_trends(text="single cell", years_back=5)""",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Topic to analyze"
                    },
                    "years_back": {
                        "type": "integer",
                        "description": "Number of years to analyze (default: 5)"
                    },
                    "activity_codes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by activity codes"
                    }
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="find_collaborators",
            description="""Find potential collaborators funded to work on a topic.

Returns PIs with active grants in the area, sorted by funding.

Example: find_collaborators(text="spatial transcriptomics", location="Tennessee")""",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Research topic"
                    },
                    "location": {
                        "type": "string",
                        "description": "State or city to filter by"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default: 20)"
                    }
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="compare_funding",
            description="""Compare funding between topics or institutions.

Example: compare_funding(topics=["spatial transcriptomics", "single cell RNA-seq"])
Example: compare_funding(orgs=["Vanderbilt", "Stanford", "Harvard"])""",
            inputSchema={
                "type": "object",
                "properties": {
                    "topics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Topics to compare"
                    },
                    "orgs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Organizations to compare"
                    },
                    "fiscal_year": {
                        "type": "integer",
                        "description": "Fiscal year (default: current)"
                    }
                }
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls"""
    try:
        if name == "search_grants":
            result = await search_grants(**arguments)
        elif name == "get_grant_details":
            result = await get_grant_details(**arguments)
        elif name == "get_pi_grants":
            result = await get_pi_grants(**arguments)
        elif name == "funding_trends":
            result = await funding_trends(**arguments)
        elif name == "find_collaborators":
            result = await find_collaborators(**arguments)
        elif name == "compare_funding":
            result = await compare_funding(**arguments)
        else:
            result = f"Unknown tool: {name}"

        return [TextContent(type="text", text=result)]

    except Exception as e:
        logger.exception(f"Error in {name}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def search_grants(
    text: Optional[str] = None,
    pi_name: Optional[str] = None,
    org_name: Optional[str] = None,
    fiscal_years: Optional[list] = None,
    activity_codes: Optional[list] = None,
    include_active_only: bool = True,
    limit: int = 25
) -> str:
    """Search NIH grants"""

    # Build criteria
    criteria = {}

    if text:
        criteria["advanced_text_search"] = {
            "operator": "and",
            "search_field": "all",
            "search_text": text
        }

    if pi_name:
        criteria["pi_names"] = [{"any_name": pi_name}]

    if org_name:
        criteria["org_names"] = [org_name]

    if fiscal_years:
        criteria["fiscal_years"] = fiscal_years
    else:
        current_year = datetime.now().year
        criteria["fiscal_years"] = [current_year, current_year - 1, current_year - 2]

    if activity_codes:
        criteria["activity_codes"] = activity_codes

    if include_active_only:
        criteria["is_active"] = True

    payload = {
        "criteria": criteria,
        "offset": 0,
        "limit": min(limit, 100),
        "sort_field": "award_amount",
        "sort_order": "desc"
    }

    response = await nih_api_request("projects/search", payload)

    results = response.get("results", [])
    total = response.get("meta", {}).get("total", 0)

    if not results:
        return f"No grants found matching criteria.\nTotal in database: {total}"

    lines = [f"Found {total} grants (showing {len(results)}):\n"]

    for grant in results:
        lines.append(format_grant(grant))
        lines.append("\n" + "-" * 60 + "\n")

    return "\n".join(lines)


async def get_grant_details(project_num: str) -> str:
    """Get details for a specific grant"""

    criteria = {
        "project_nums": [project_num]
    }

    payload = {
        "criteria": criteria,
        "offset": 0,
        "limit": 10
    }

    response = await nih_api_request("projects/search", payload)

    results = response.get("results", [])
    if not results:
        return f"No grant found with project number: {project_num}"

    grant = results[0]

    # Full format for single grant
    lines = []
    lines.append(f"# Grant: {grant.get('project_num', 'N/A')}")
    lines.append(f"\n## {grant.get('project_title', 'No title')}")

    # PIs
    lines.append("\n### Principal Investigators")
    for pi in grant.get("principal_investigators", []):
        name = f"{pi.get('first_name', '')} {pi.get('last_name', '')}"
        lines.append(f"- {name}")

    # Organization
    org = grant.get("organization", {})
    lines.append(f"\n### Organization")
    lines.append(f"- Name: {org.get('org_name', 'Unknown')}")
    lines.append(f"- City: {org.get('org_city', '')}, {org.get('org_state', '')}")

    # Funding
    lines.append(f"\n### Funding")
    lines.append(f"- Award Amount: ${grant.get('award_amount', 0):,.0f}")
    lines.append(f"- Activity Code: {grant.get('activity_code', 'N/A')}")
    lines.append(f"- Fiscal Year: {grant.get('fiscal_year', 'N/A')}")

    # Dates
    lines.append(f"\n### Project Period")
    lines.append(f"- Start: {grant.get('project_start_date', 'N/A')[:10] if grant.get('project_start_date') else 'N/A'}")
    lines.append(f"- End: {grant.get('project_end_date', 'N/A')[:10] if grant.get('project_end_date') else 'N/A'}")

    # Abstract
    abstract = grant.get("abstract_text", "")
    if abstract:
        lines.append(f"\n### Abstract\n{abstract}")

    # Public Health Relevance
    phr = grant.get("phr_text", "")
    if phr:
        lines.append(f"\n### Public Health Relevance\n{phr}")

    # Terms
    terms = grant.get("terms", "")
    if terms:
        lines.append(f"\n### Keywords\n{terms}")

    return "\n".join(lines)


async def get_pi_grants(
    pi_name: str,
    org_name: Optional[str] = None,
    include_past: bool = False
) -> str:
    """Get all grants for a PI"""

    criteria = {
        "pi_names": [{"any_name": pi_name}]
    }

    if org_name:
        criteria["org_names"] = [org_name]

    if not include_past:
        criteria["is_active"] = True

    payload = {
        "criteria": criteria,
        "offset": 0,
        "limit": 100,
        "sort_field": "fiscal_year",
        "sort_order": "desc"
    }

    response = await nih_api_request("projects/search", payload)

    results = response.get("results", [])
    total = response.get("meta", {}).get("total", 0)

    if not results:
        return f"No grants found for PI: {pi_name}"

    # Calculate totals
    total_funding = sum(g.get("award_amount", 0) or 0 for g in results)

    lines = [f"# Grants for {pi_name}"]
    lines.append(f"\nTotal: {total} grants, ${total_funding:,.0f} in funding\n")

    for grant in results:
        lines.append(format_grant(grant))
        lines.append("\n" + "-" * 60 + "\n")

    return "\n".join(lines)


async def funding_trends(
    text: str,
    years_back: int = 5,
    activity_codes: Optional[list] = None
) -> str:
    """Analyze funding trends over time"""

    current_year = datetime.now().year
    years = list(range(current_year - years_back + 1, current_year + 1))

    trends = []

    for year in years:
        criteria = {
            "advanced_text_search": {
                "operator": "and",
                "search_field": "all",
                "search_text": text
            },
            "fiscal_years": [year]
        }

        if activity_codes:
            criteria["activity_codes"] = activity_codes

        payload = {
            "criteria": criteria,
            "offset": 0,
            "limit": 1  # Just need metadata
        }

        response = await nih_api_request("projects/search", payload)

        total = response.get("meta", {}).get("total", 0)

        # Get total funding for this year (need to fetch more)
        if total > 0:
            payload["limit"] = min(total, 500)
            response = await nih_api_request("projects/search", payload)
            results = response.get("results", [])
            total_funding = sum(g.get("award_amount", 0) or 0 for g in results)
        else:
            total_funding = 0

        trends.append({
            "year": year,
            "grants": total,
            "funding": total_funding
        })

    # Format output
    lines = [f"# Funding Trends: {text}\n"]

    lines.append("| Year | Grants | Total Funding |")
    lines.append("|------|--------|---------------|")

    for t in trends:
        lines.append(f"| {t['year']} | {t['grants']} | ${t['funding']:,.0f} |")

    # Calculate growth
    if len(trends) >= 2 and trends[0]["funding"] > 0:
        growth = ((trends[-1]["funding"] / trends[0]["funding"]) - 1) * 100
        lines.append(f"\nFunding growth over {years_back} years: {growth:.1f}%")

    return "\n".join(lines)


async def find_collaborators(
    text: str,
    location: Optional[str] = None,
    limit: int = 20
) -> str:
    """Find potential collaborators"""

    current_year = datetime.now().year

    criteria = {
        "advanced_text_search": {
            "operator": "and",
            "search_field": "all",
            "search_text": text
        },
        "fiscal_years": [current_year, current_year - 1],
        "is_active": True
    }

    if location:
        criteria["org_states"] = [location]

    payload = {
        "criteria": criteria,
        "offset": 0,
        "limit": 100,
        "sort_field": "award_amount",
        "sort_order": "desc"
    }

    response = await nih_api_request("projects/search", payload)

    results = response.get("results", [])

    if not results:
        return f"No active grants found for: {text}"

    # Aggregate by PI
    pi_funding = {}
    for grant in results:
        for pi in grant.get("principal_investigators", []):
            name = f"{pi.get('first_name', '')} {pi.get('last_name', '')}"
            org = grant.get("organization", {}).get("org_name", "Unknown")
            key = f"{name} ({org})"

            if key not in pi_funding:
                pi_funding[key] = {
                    "name": name,
                    "org": org,
                    "grants": [],
                    "total": 0
                }

            pi_funding[key]["grants"].append(grant.get("project_num"))
            pi_funding[key]["total"] += grant.get("award_amount", 0) or 0

    # Sort by funding
    sorted_pis = sorted(pi_funding.values(), key=lambda x: x["total"], reverse=True)[:limit]

    lines = [f"# Potential Collaborators: {text}\n"]

    for i, pi in enumerate(sorted_pis, 1):
        lines.append(f"{i}. **{pi['name']}** - {pi['org']}")
        lines.append(f"   Funding: ${pi['total']:,.0f}")
        lines.append(f"   Grants: {', '.join(pi['grants'][:3])}")
        lines.append("")

    return "\n".join(lines)


async def compare_funding(
    topics: Optional[list] = None,
    orgs: Optional[list] = None,
    fiscal_year: Optional[int] = None
) -> str:
    """Compare funding between topics or orgs"""

    if not topics and not orgs:
        return "Please provide either topics or orgs to compare"

    year = fiscal_year or datetime.now().year
    results = []

    if topics:
        for topic in topics:
            criteria = {
                "advanced_text_search": {
                    "operator": "and",
                    "search_field": "all",
                    "search_text": topic
                },
                "fiscal_years": [year],
                "is_active": True
            }

            payload = {
                "criteria": criteria,
                "offset": 0,
                "limit": 500
            }

            response = await nih_api_request("projects/search", payload)
            grants = response.get("results", [])
            total = response.get("meta", {}).get("total", 0)
            funding = sum(g.get("award_amount", 0) or 0 for g in grants)

            results.append({
                "name": topic,
                "grants": total,
                "funding": funding
            })

    if orgs:
        for org in orgs:
            criteria = {
                "org_names": [org],
                "fiscal_years": [year],
                "is_active": True
            }

            payload = {
                "criteria": criteria,
                "offset": 0,
                "limit": 500
            }

            response = await nih_api_request("projects/search", payload)
            grants = response.get("results", [])
            total = response.get("meta", {}).get("total", 0)
            funding = sum(g.get("award_amount", 0) or 0 for g in grants)

            results.append({
                "name": org,
                "grants": total,
                "funding": funding
            })

    # Format output
    lines = [f"# Funding Comparison (FY{year})\n"]

    lines.append("| Name | Grants | Total Funding |")
    lines.append("|------|--------|---------------|")

    for r in sorted(results, key=lambda x: x["funding"], reverse=True):
        lines.append(f"| {r['name']} | {r['grants']} | ${r['funding']:,.0f} |")

    return "\n".join(lines)


async def main():
    """Run the MCP server"""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
