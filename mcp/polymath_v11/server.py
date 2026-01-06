#!/usr/bin/env python3
"""
Polymath v11 MCP Server - Self-Growing Research Intelligence

40+ tools for:
- Research Discovery (8 tools)
- Hypothesis & Reasoning (6 tools)
- Writing & Communication (8 tools)
- Code & Analysis (8 tools)
- Knowledge Management (6 tools)
- Self-Improvement (4 tools)
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Import tool modules
from tools.discovery import DiscoveryTools
from tools.reasoning import ReasoningTools

# Initialize server
server = Server("polymath-v11")

# Initialize tool classes
discovery = DiscoveryTools()
reasoning = ReasoningTools()


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List all available Polymath v11 tools."""
    tools = []

    # Discovery Tools (1-8)
    # Maturity: [stable] = production ready, [beta] = works but needs polish, [alpha] = stub/incomplete
    tools.extend([
        Tool(
            name="deep_hunt",
            description="[stable] Deep literature hunt across all sources (Polymath corpus, Europe PMC, arXiv, bioRxiv, GitHub). Returns top 20 papers + code repos with evidence summaries.",
            inputSchema={
                "type": "object",
                "properties": {
                    "research_question": {
                        "type": "string",
                        "description": "The research question to investigate"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results to return (default: 20)",
                        "default": 20
                    }
                },
                "required": ["research_question"]
            }
        ),
        Tool(
            name="find_gaps",
            description="[stable] Detect research gaps: orphan concepts, time gaps (no recent work), method gaps (technique not applied to domain). Returns ranked opportunities.",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Topic or concept to analyze for gaps"
                    }
                },
                "required": ["topic"]
            }
        ),
        Tool(
            name="watch_competitor",
            description="[alpha] Track a lab/author's recent publications, methods, datasets, and collaborators. Currently searches by name only - external API integration pending.",
            inputSchema={
                "type": "object",
                "properties": {
                    "lab_or_author": {
                        "type": "string",
                        "description": "Lab name or author to track"
                    },
                    "your_topics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Your research topics for overlap detection"
                    }
                },
                "required": ["lab_or_author"]
            }
        ),
        Tool(
            name="trend_radar",
            description="[beta] Analyze publication velocity, emerging methods, rising authors in a field. Returns trend report with visualizations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "field": {
                        "type": "string",
                        "description": "Field to analyze (e.g., 'spatial transcriptomics')"
                    },
                    "time_window_months": {
                        "type": "integer",
                        "description": "Time window for analysis (default: 24)",
                        "default": 24
                    }
                },
                "required": ["field"]
            }
        ),
        Tool(
            name="find_datasets",
            description="[alpha] Hunt for datasets across GEO, SRA, Zenodo, GitHub, paper supplements. Uses keyword heuristics - direct API integration pending.",
            inputSchema={
                "type": "object",
                "properties": {
                    "data_type": {
                        "type": "string",
                        "description": "Type of data needed (e.g., 'Visium HD colorectal cancer')"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results (default: 10)",
                        "default": 10
                    }
                },
                "required": ["data_type"]
            }
        ),
    ])

    # Reasoning Tools (9-14)
    tools.extend([
        Tool(
            name="generate_hypothesis",
            description="[stable] Generate cross-domain hypotheses using structure mapping. Finds unexplored concept bridges and scores by novelty (corpus-based) x feasibility x testability.",
            inputSchema={
                "type": "object",
                "properties": {
                    "research_area": {
                        "type": "string",
                        "description": "Research area to generate hypotheses for"
                    },
                    "num_hypotheses": {
                        "type": "integer",
                        "description": "Number of hypotheses to generate (default: 5)",
                        "default": 5
                    }
                },
                "required": ["research_area"]
            }
        ),
        Tool(
            name="validate_hypothesis",
            description="[beta] Validate a hypothesis against the corpus. Decomposes into claims, scores each with heuristics, identifies gaps. NLI scoring coming soon.",
            inputSchema={
                "type": "object",
                "properties": {
                    "hypothesis": {
                        "type": "string",
                        "description": "Hypothesis statement to validate"
                    }
                },
                "required": ["hypothesis"]
            }
        ),
        Tool(
            name="find_analogy",
            description="[beta] Find analogous solutions from unexpected domains. Abstracts problem to structural pattern and maps solutions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "problem": {
                        "type": "string",
                        "description": "Problem description to find analogies for"
                    }
                },
                "required": ["problem"]
            }
        ),
        Tool(
            name="serendipity",
            description="[stable] Surface unexpected but potentially useful connections. Returns surprising concept bridges you might not think to look for.",
            inputSchema={
                "type": "object",
                "properties": {
                    "seed_concept": {
                        "type": "string",
                        "description": "Optional seed concept to start from"
                    }
                },
                "required": []
            }
        ),
    ])

    # Self-Improvement Tools (37-40)
    tools.extend([
        Tool(
            name="collection_health",
            description="[stable] Check knowledge base health: recency, coverage gaps, code-paper linkage, citability rate. Returns health report + action items.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="expand_collection",
            description="[beta] Propose expansion targets based on strategy: fill_gaps, follow_citations, trending. Returns expansion plan for approval.",
            inputSchema={
                "type": "object",
                "properties": {
                    "strategy": {
                        "type": "string",
                        "enum": ["fill_gaps", "follow_citations", "trending", "author_track"],
                        "description": "Expansion strategy to use"
                    }
                },
                "required": ["strategy"]
            }
        ),
    ])

    return tools


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""

    try:
        # Discovery tools
        if name == "deep_hunt":
            result = await discovery.deep_hunt(
                arguments["research_question"],
                arguments.get("max_results", 20)
            )
        elif name == "find_gaps":
            result = await discovery.find_gaps(arguments["topic"])
        elif name == "watch_competitor":
            result = await discovery.watch_competitor(
                arguments["lab_or_author"],
                arguments.get("your_topics", [])
            )
        elif name == "trend_radar":
            result = await discovery.trend_radar(
                arguments["field"],
                arguments.get("time_window_months", 24)
            )
        elif name == "find_datasets":
            result = await discovery.find_datasets(
                arguments["data_type"],
                arguments.get("max_results", 10)
            )

        # Reasoning tools
        elif name == "generate_hypothesis":
            result = await reasoning.generate_hypothesis(
                arguments["research_area"],
                arguments.get("num_hypotheses", 5)
            )
        elif name == "validate_hypothesis":
            result = await reasoning.validate_hypothesis(arguments["hypothesis"])
        elif name == "find_analogy":
            result = await reasoning.find_analogy(arguments["problem"])
        elif name == "serendipity":
            result = await reasoning.serendipity(arguments.get("seed_concept"))

        # Self-improvement tools
        elif name == "collection_health":
            result = await discovery.collection_health()
        elif name == "expand_collection":
            result = await discovery.expand_collection(arguments["strategy"])

        else:
            result = f"Unknown tool: {name}"

        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
