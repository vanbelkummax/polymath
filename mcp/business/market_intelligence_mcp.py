#!/usr/bin/env python3
"""
GDELT Market Intelligence MCP Server

Provides real-time news monitoring, event tracking, and sentiment analysis
using the GDELT Project API (updates every 15 minutes).

API Documentation: https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
Rate Limits: No hard limits, but be respectful
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Any, Optional
from collections import Counter
import hashlib
from urllib.parse import quote

import httpx

# Add parent paths for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mcp.server import Server
from mcp.types import Tool, TextContent

# GDELT API endpoints
GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_TV_API = "https://api.gdeltproject.org/api/v2/tv/tv"
GDELT_GEO_API = "https://api.gdeltproject.org/api/v2/geo/geo"

# Rate limiting (conservative)
MAX_REQUESTS_PER_MINUTE = 30
REQUEST_DELAY = 60.0 / MAX_REQUESTS_PER_MINUTE

# Cache configuration
CACHE_DIR = os.path.expanduser("~/.cache/polymath/market_intel")
CACHE_EXPIRY_MINUTES = 15  # GDELT updates every 15 minutes

# GDELT themes relevant to business/finance
BUSINESS_THEMES = [
    "ECON_", "BUS_", "CORP_", "TAX_", "TRADE_", "INVEST_",
    "BANKRUPTCY", "MERGER", "ACQUISITION", "IPO", "LAWSUIT",
    "REGULATION", "SANCTION", "TARIFF"
]

# Tone thresholds
TONE_THRESHOLDS = {
    "very_positive": 5.0,
    "positive": 1.5,
    "neutral": -1.5,
    "negative": -5.0,
    "very_negative": -10.0
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
            if datetime.now() - cached_time < timedelta(minutes=CACHE_EXPIRY_MINUTES):
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


def classify_tone(tone: float) -> str:
    """Classify a GDELT tone score."""
    if tone >= TONE_THRESHOLDS["very_positive"]:
        return "very_positive"
    elif tone >= TONE_THRESHOLDS["positive"]:
        return "positive"
    elif tone >= TONE_THRESHOLDS["neutral"]:
        return "neutral"
    elif tone >= TONE_THRESHOLDS["negative"]:
        return "negative"
    else:
        return "very_negative"


class GDELTClient:
    """Async client for GDELT API."""

    def __init__(self):
        self.last_request_time = 0
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=60.0,  # GDELT can be slow
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

    async def search_news(
        self,
        query: str,
        mode: str = "artlist",
        timespan: str = "24h",
        max_records: int = 75,
        source_country: Optional[str] = None,
        source_lang: str = "English",
        sort: str = "tonedesc"
    ) -> dict:
        """
        Search GDELT news articles.

        Args:
            query: Search query (company, topic, etc.)
            mode: 'artlist' for articles, 'timeline' for trends, 'tonechart' for sentiment
            timespan: Time range (e.g., '24h', '7d', '30d')
            max_records: Maximum articles to return (max 250)
            source_country: Filter by source country code
            source_lang: Source language
            sort: Sort order ('tonedesc', 'toneasc', 'datedesc', 'dateasc')
        """
        cache_key = f"news:{query}:{mode}:{timespan}:{max_records}:{source_country}:{sort}"
        cached = get_cached(cache_key)
        if cached:
            return cached

        await self._rate_limit()
        client = await self._get_client()

        params = {
            "query": quote(query),
            "mode": mode,
            "format": "json",
            "timespan": timespan,
            "maxrecords": min(max_records, 250),
            "sort": sort,
            "sourcelang": source_lang
        }

        if source_country:
            params["sourcecountry"] = source_country

        url = f"{GDELT_DOC_API}?" + "&".join(f"{k}={v}" for k, v in params.items())

        try:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

            if mode == "artlist":
                articles = data.get("articles", [])
                result = {
                    "query": query,
                    "timespan": timespan,
                    "total_found": len(articles),
                    "articles": [
                        {
                            "title": a.get("title"),
                            "url": a.get("url"),
                            "source": a.get("domain"),
                            "source_country": a.get("sourcecountry"),
                            "language": a.get("language"),
                            "seendate": a.get("seendate"),
                            "tone": a.get("tone"),
                            "tone_class": classify_tone(float(a.get("tone", 0))),
                            "socialimage": a.get("socialimage"),
                        }
                        for a in articles
                    ]
                }
            elif mode == "timeline":
                result = {
                    "query": query,
                    "timespan": timespan,
                    "timeline": data.get("timeline", [])
                }
            elif mode == "tonechart":
                result = {
                    "query": query,
                    "timespan": timespan,
                    "tonechart": data.get("tonechart", [])
                }
            else:
                result = data

            set_cached(cache_key, result)
            return result

        except httpx.HTTPError as e:
            return {"error": str(e), "query": query}

    async def get_sentiment_trend(
        self,
        query: str,
        timespan: str = "7d"
    ) -> dict:
        """
        Get sentiment trend for a topic over time.

        Uses GDELT's timeline mode with tone data.
        """
        cache_key = f"sentiment:{query}:{timespan}"
        cached = get_cached(cache_key)
        if cached:
            return cached

        await self._rate_limit()
        client = await self._get_client()

        params = {
            "query": quote(query),
            "mode": "timelinetone",
            "format": "json",
            "timespan": timespan,
        }

        url = f"{GDELT_DOC_API}?" + "&".join(f"{k}={v}" for k, v in params.items())

        try:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

            timeline = data.get("timeline", [])

            # Analyze trend
            if len(timeline) >= 2:
                tones = [float(t.get("value", 0)) for t in timeline if t.get("value")]
                if tones:
                    avg_tone = sum(tones) / len(tones)
                    recent_tone = sum(tones[-3:]) / min(3, len(tones)) if len(tones) >= 3 else tones[-1]
                    early_tone = sum(tones[:3]) / min(3, len(tones))
                    tone_change = recent_tone - early_tone
                else:
                    avg_tone = recent_tone = early_tone = tone_change = 0
            else:
                avg_tone = recent_tone = early_tone = tone_change = 0

            result = {
                "query": query,
                "timespan": timespan,
                "timeline": timeline,
                "analysis": {
                    "average_tone": round(avg_tone, 2),
                    "recent_tone": round(recent_tone, 2),
                    "tone_change": round(tone_change, 2),
                    "trend_direction": "improving" if tone_change > 0.5 else ("declining" if tone_change < -0.5 else "stable"),
                    "current_sentiment": classify_tone(recent_tone),
                    "data_points": len(timeline)
                }
            }

            set_cached(cache_key, result)
            return result

        except httpx.HTTPError as e:
            return {"error": str(e), "query": query}

    async def get_news_volume(
        self,
        query: str,
        timespan: str = "7d"
    ) -> dict:
        """
        Get news volume/velocity over time for a topic.
        """
        cache_key = f"volume:{query}:{timespan}"
        cached = get_cached(cache_key)
        if cached:
            return cached

        await self._rate_limit()
        client = await self._get_client()

        params = {
            "query": quote(query),
            "mode": "timelinevolraw",
            "format": "json",
            "timespan": timespan,
        }

        url = f"{GDELT_DOC_API}?" + "&".join(f"{k}={v}" for k, v in params.items())

        try:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

            timeline = data.get("timeline", [])

            # Analyze velocity
            volumes = [int(t.get("value", 0)) for t in timeline if t.get("value")]
            if volumes:
                total_volume = sum(volumes)
                avg_volume = total_volume / len(volumes)
                max_volume = max(volumes)
                recent_volume = sum(volumes[-3:]) if len(volumes) >= 3 else volumes[-1] if volumes else 0
                velocity_trend = (recent_volume / avg_volume) if avg_volume > 0 else 0
            else:
                total_volume = avg_volume = max_volume = recent_volume = velocity_trend = 0

            result = {
                "query": query,
                "timespan": timespan,
                "timeline": timeline,
                "analysis": {
                    "total_articles": total_volume,
                    "average_per_period": round(avg_volume, 1),
                    "peak_volume": max_volume,
                    "recent_volume": recent_volume,
                    "velocity_ratio": round(velocity_trend, 2),
                    "trending": velocity_trend > 1.5,
                    "data_points": len(timeline)
                }
            }

            set_cached(cache_key, result)
            return result

        except httpx.HTTPError as e:
            return {"error": str(e), "query": query}

    async def compare_entities(
        self,
        entities: list[str],
        timespan: str = "24h"
    ) -> dict:
        """
        Compare news coverage and sentiment across multiple entities.

        Useful for competitive intelligence.
        """
        results = {}

        for entity in entities[:5]:  # Limit to 5 entities
            # Get volume
            volume_data = await self.get_news_volume(entity, timespan)
            # Get sentiment
            sentiment_data = await self.get_sentiment_trend(entity, timespan)

            results[entity] = {
                "volume": volume_data.get("analysis", {}),
                "sentiment": sentiment_data.get("analysis", {})
            }

        # Rank entities
        ranked = sorted(
            results.items(),
            key=lambda x: x[1]["volume"].get("total_articles", 0),
            reverse=True
        )

        return {
            "entities": entities,
            "timespan": timespan,
            "comparison": results,
            "volume_ranking": [e[0] for e in ranked],
            "sentiment_ranking": sorted(
                results.items(),
                key=lambda x: x[1]["sentiment"].get("recent_tone", 0),
                reverse=True
            ),
            "trending_entities": [
                e for e, d in results.items()
                if d["volume"].get("trending", False)
            ]
        }

    async def track_event_mentions(
        self,
        company: str,
        event_types: Optional[list[str]] = None,
        timespan: str = "7d"
    ) -> dict:
        """
        Track mentions of a company in context of specific business events.

        Event types: merger, acquisition, lawsuit, bankruptcy, IPO, earnings, etc.
        """
        if event_types is None:
            event_types = ["merger", "acquisition", "lawsuit", "bankruptcy", "IPO", "earnings", "layoff"]

        event_results = {}

        for event_type in event_types[:7]:
            query = f'{company} "{event_type}"'
            news_data = await self.search_news(
                query=query,
                timespan=timespan,
                max_records=25
            )

            articles = news_data.get("articles", [])
            tones = [float(a.get("tone", 0)) for a in articles]
            avg_tone = sum(tones) / len(tones) if tones else 0

            event_results[event_type] = {
                "article_count": len(articles),
                "average_tone": round(avg_tone, 2),
                "sentiment": classify_tone(avg_tone),
                "top_articles": [
                    {
                        "title": a.get("title"),
                        "source": a.get("source"),
                        "url": a.get("url"),
                        "date": a.get("seendate")
                    }
                    for a in articles[:3]
                ]
            }

        # Identify significant events
        significant_events = [
            {"event": e, "count": d["article_count"], "tone": d["average_tone"]}
            for e, d in event_results.items()
            if d["article_count"] >= 3
        ]

        return {
            "company": company,
            "timespan": timespan,
            "event_coverage": event_results,
            "significant_events": sorted(significant_events, key=lambda x: x["count"], reverse=True),
            "total_event_mentions": sum(d["article_count"] for d in event_results.values()),
            "analysis": {
                "most_covered_event": max(event_results.items(), key=lambda x: x[1]["article_count"])[0] if event_results else None,
                "most_positive_event": max(event_results.items(), key=lambda x: x[1]["average_tone"])[0] if event_results else None,
                "most_negative_event": min(event_results.items(), key=lambda x: x[1]["average_tone"])[0] if event_results else None,
            }
        }

    async def get_theme_breakdown(
        self,
        query: str,
        timespan: str = "24h"
    ) -> dict:
        """
        Get GDELT theme breakdown for a query.

        Shows what topics/themes are associated with coverage.
        """
        cache_key = f"themes:{query}:{timespan}"
        cached = get_cached(cache_key)
        if cached:
            return cached

        await self._rate_limit()
        client = await self._get_client()

        params = {
            "query": quote(query),
            "mode": "artlist",
            "format": "json",
            "timespan": timespan,
            "maxrecords": 100,
        }

        url = f"{GDELT_DOC_API}?" + "&".join(f"{k}={v}" for k, v in params.items())

        try:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

            articles = data.get("articles", [])

            # Extract themes (GDELT returns themes as comma-separated in some responses)
            # For now, analyze by source diversity
            sources = Counter([a.get("domain") for a in articles if a.get("domain")])
            countries = Counter([a.get("sourcecountry") for a in articles if a.get("sourcecountry")])

            result = {
                "query": query,
                "timespan": timespan,
                "article_count": len(articles),
                "source_diversity": {
                    "unique_sources": len(sources),
                    "top_sources": sources.most_common(10)
                },
                "geographic_distribution": {
                    "unique_countries": len(countries),
                    "top_countries": countries.most_common(10)
                }
            }

            set_cached(cache_key, result)
            return result

        except httpx.HTTPError as e:
            return {"error": str(e), "query": query}


# MCP Server setup
server = Server("market-intelligence")
client = GDELTClient()


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_news",
            description="Search global news using GDELT (updates every 15 minutes). Great for monitoring companies, topics, and events.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (company name, topic, etc.)"
                    },
                    "timespan": {
                        "type": "string",
                        "description": "Time range: '24h', '7d', '30d', etc. (default: 24h)"
                    },
                    "max_records": {
                        "type": "integer",
                        "description": "Maximum articles to return (default: 75, max: 250)"
                    },
                    "source_country": {
                        "type": "string",
                        "description": "Filter by source country code (e.g., 'US', 'UK')"
                    },
                    "sort": {
                        "type": "string",
                        "description": "Sort: 'tonedesc', 'toneasc', 'datedesc', 'dateasc'"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_sentiment_trend",
            description="Get sentiment trend over time for a topic/company. Shows how public perception is changing.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Topic or company to analyze"
                    },
                    "timespan": {
                        "type": "string",
                        "description": "Time range for analysis (default: '7d')"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_news_volume",
            description="Get news coverage volume/velocity over time. Detect if a topic is trending.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Topic to analyze"
                    },
                    "timespan": {
                        "type": "string",
                        "description": "Time range (default: '7d')"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="compare_entities",
            description="Compare news coverage and sentiment across multiple companies/entities. Competitive intelligence.",
            inputSchema={
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of entities to compare (max 5)"
                    },
                    "timespan": {
                        "type": "string",
                        "description": "Time range (default: '24h')"
                    }
                },
                "required": ["entities"]
            }
        ),
        Tool(
            name="track_event_mentions",
            description="Track a company's mentions in context of specific business events (M&A, lawsuits, etc.).",
            inputSchema={
                "type": "object",
                "properties": {
                    "company": {
                        "type": "string",
                        "description": "Company name to track"
                    },
                    "event_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Event types to track (default: merger, acquisition, lawsuit, bankruptcy, IPO, earnings, layoff)"
                    },
                    "timespan": {
                        "type": "string",
                        "description": "Time range (default: '7d')"
                    }
                },
                "required": ["company"]
            }
        ),
        Tool(
            name="get_theme_breakdown",
            description="Get source and geographic breakdown of coverage for a topic.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Topic to analyze"
                    },
                    "timespan": {
                        "type": "string",
                        "description": "Time range (default: '24h')"
                    }
                },
                "required": ["query"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        if name == "search_news":
            result = await client.search_news(
                query=arguments["query"],
                timespan=arguments.get("timespan", "24h"),
                max_records=arguments.get("max_records", 75),
                source_country=arguments.get("source_country"),
                sort=arguments.get("sort", "datedesc")
            )

        elif name == "get_sentiment_trend":
            result = await client.get_sentiment_trend(
                query=arguments["query"],
                timespan=arguments.get("timespan", "7d")
            )

        elif name == "get_news_volume":
            result = await client.get_news_volume(
                query=arguments["query"],
                timespan=arguments.get("timespan", "7d")
            )

        elif name == "compare_entities":
            result = await client.compare_entities(
                entities=arguments["entities"],
                timespan=arguments.get("timespan", "24h")
            )

        elif name == "track_event_mentions":
            result = await client.track_event_mentions(
                company=arguments["company"],
                event_types=arguments.get("event_types"),
                timespan=arguments.get("timespan", "7d")
            )

        elif name == "get_theme_breakdown":
            result = await client.get_theme_breakdown(
                query=arguments["query"],
                timespan=arguments.get("timespan", "24h")
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
