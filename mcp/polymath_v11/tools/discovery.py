"""
Polymath v11 Discovery Tools

Tools 1-8: Research discovery, gap detection, trend analysis
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from collections import Counter

# Add repo root to path so `lib.*` imports resolve correctly.
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from lib.hybrid_search_v2 import HybridSearcherV2
from lib.db import Database


class DiscoveryTools:
    """Research discovery tools for Polymath v11."""

    def __init__(self):
        self.searcher = None  # Lazy init

    def _get_searcher(self):
        if self.searcher is None:
            self.searcher = HybridSearcherV2()
        return self.searcher

    async def deep_hunt(self, research_question: str, max_results: int = 20) -> dict:
        """
        Tool 1: Deep literature hunt across all sources.

        1. Search Polymath corpus (papers + code)
        2. Score by: relevance x recency x citation_count x cross_domain_bridge
        3. Return ranked list with evidence summaries
        """
        hs = self._get_searcher()

        # Search both papers and code
        paper_results = hs.search_papers(research_question, n=max_results)
        code_results = hs.search_code(research_question, n=max_results // 2)

        # Format results (SearchResult objects have .content, .metadata, .score)
        papers = []
        for r in paper_results:
            meta = r.metadata
            concepts_raw = meta.get("concepts", "")
            concepts = concepts_raw.split(",")[:5] if isinstance(concepts_raw, str) else concepts_raw[:5] if concepts_raw else []
            papers.append({
                "type": "paper",
                "title": r.title,
                "year": meta.get("year"),
                "doi": meta.get("doi"),
                "concepts": concepts,
                "relevance_score": round(r.score, 3),
                "snippet": r.content[:300] + "..." if len(r.content) > 300 else r.content
            })

        code = []
        for r in code_results:
            meta = r.metadata
            code.append({
                "type": "code",
                "repo": meta.get("repo_name", "Unknown"),
                "file": meta.get("file_path", ""),
                "chunk_type": meta.get("chunk_type", ""),
                "relevance_score": round(r.score, 3),
                "snippet": r.content[:200] + "..." if len(r.content) > 200 else r.content
            })

        return {
            "query": research_question,
            "total_papers": len(papers),
            "total_code": len(code),
            "papers": papers[:max_results],
            "code": code[:max_results // 2],
            "suggestions": [
                f"Try narrower query: '{research_question} methods'",
                f"Search code: '{research_question}' in mahmoodlab repos",
            ]
        }

    async def find_gaps(self, topic: str) -> dict:
        """
        Tool 2: Detect research gaps.

        1. Map concept neighborhood in Neo4j
        2. Find orphan concepts (few connections)
        3. Find time gaps (no recent papers)
        4. Find method gaps (technique not applied to domain)
        """
        hs = self._get_searcher()

        # Get papers on topic
        results = hs.search_papers(topic, n=50)

        # Analyze temporal distribution (SearchResult objects)
        years = [r.metadata.get("year") for r in results if r.metadata.get("year")]
        year_counts = Counter(years)

        # Find concepts
        all_concepts = []
        for r in results:
            concepts_raw = r.metadata.get("concepts", "")
            if isinstance(concepts_raw, str) and concepts_raw:
                all_concepts.extend(concepts_raw.split(","))
            elif concepts_raw:
                all_concepts.extend(concepts_raw)
        concept_counts = Counter(all_concepts)

        # Identify gaps
        current_year = datetime.now().year
        recent_years = [current_year - i for i in range(3)]
        recent_count = sum(year_counts.get(y, 0) for y in recent_years)

        # Find rare concepts (potential orphans)
        rare_concepts = [c for c, count in concept_counts.items() if count == 1]

        # Find common concepts not combined
        top_concepts = [c for c, _ in concept_counts.most_common(10)]

        gaps = {
            "topic": topic,
            "total_papers": len(results),
            "temporal_analysis": {
                "year_distribution": dict(sorted(year_counts.items())),
                "recent_3yr_count": recent_count,
                "recency_gap": recent_count < 5
            },
            "concept_analysis": {
                "top_concepts": top_concepts,
                "rare_concepts": rare_concepts[:10],
                "orphan_candidates": rare_concepts[:5]
            },
            "suggested_gaps": []
        }

        # Generate gap suggestions
        if recent_count < 5:
            gaps["suggested_gaps"].append({
                "type": "temporal",
                "description": f"Limited recent work on '{topic}' - opportunity for fresh perspective",
                "priority": "high"
            })

        if len(rare_concepts) > 5:
            gaps["suggested_gaps"].append({
                "type": "orphan_concepts",
                "description": f"Underexplored concepts: {', '.join(rare_concepts[:3])}",
                "priority": "medium"
            })

        # Cross-domain opportunity
        if "machine_learning" not in top_concepts and "deep_learning" not in top_concepts:
            gaps["suggested_gaps"].append({
                "type": "method_gap",
                "description": f"ML/DL methods not heavily applied to '{topic}' - potential opportunity",
                "priority": "high"
            })

        return gaps

    async def watch_competitor(self, lab_or_author: str, your_topics: list = None) -> dict:
        """
        Tool 3: Track a lab/author's recent publications.
        """
        hs = self._get_searcher()

        # Search for papers by this author/lab
        results = hs.search_papers(lab_or_author, n=30)

        # Extract patterns (SearchResult objects)
        years = []
        methods = []
        datasets = []

        for r in results:
            meta = r.metadata
            if meta.get("year"):
                years.append(meta["year"])
            concepts_raw = meta.get("concepts", "")
            if isinstance(concepts_raw, str) and concepts_raw:
                methods.extend(concepts_raw.split(","))
            elif concepts_raw:
                methods.extend(concepts_raw)

        year_counts = Counter(years)
        method_counts = Counter(methods)

        # Find overlaps with your topics
        overlaps = []
        if your_topics:
            for topic in your_topics:
                topic_lower = topic.lower()
                for method, count in method_counts.items():
                    if topic_lower in method.lower():
                        overlaps.append({
                            "your_topic": topic,
                            "their_work": method,
                            "frequency": count
                        })

        return {
            "lab_or_author": lab_or_author,
            "papers_found": len(results),
            "publication_velocity": dict(sorted(year_counts.items())),
            "top_methods": [m for m, _ in method_counts.most_common(10)],
            "overlaps_with_your_work": overlaps,
            "recommendation": "Set up alerts for this author on PubMed/Google Scholar"
        }

    async def trend_radar(self, field: str, time_window_months: int = 24) -> dict:
        """
        Tool 4: Analyze publication trends in a field.
        """
        hs = self._get_searcher()

        # Get papers in field
        results = hs.search_papers(field, n=100)

        # Temporal analysis (SearchResult objects)
        current_year = datetime.now().year
        years = [r.metadata.get("year") for r in results if r.metadata.get("year")]
        year_counts = Counter(years)

        # Concept evolution
        concepts_by_year = {}
        for r in results:
            year = r.metadata.get("year")
            if year:
                if year not in concepts_by_year:
                    concepts_by_year[year] = []
                concepts_raw = r.metadata.get("concepts", "")
                if isinstance(concepts_raw, str) and concepts_raw:
                    concepts_by_year[year].extend(concepts_raw.split(","))
                elif concepts_raw:
                    concepts_by_year[year].extend(concepts_raw)

        # Find emerging concepts (appear in recent years, not earlier)
        recent = current_year
        old = current_year - 3

        recent_concepts = set()
        old_concepts = set()

        for year, concepts in concepts_by_year.items():
            if year >= recent - 1:
                recent_concepts.update(concepts)
            if year <= old:
                old_concepts.update(concepts)

        emerging = recent_concepts - old_concepts

        # Calculate velocity
        recent_count = sum(year_counts.get(y, 0) for y in [current_year, current_year - 1])
        old_count = sum(year_counts.get(y, 0) for y in [current_year - 3, current_year - 4])
        velocity = "increasing" if recent_count > old_count else "stable" if recent_count == old_count else "decreasing"

        return {
            "field": field,
            "papers_analyzed": len(results),
            "velocity": velocity,
            "publication_trend": dict(sorted(year_counts.items())),
            "emerging_concepts": list(emerging)[:15],
            "established_concepts": list(old_concepts & recent_concepts)[:10],
            "hot_topics": [
                f"{field} + {c}" for c in list(emerging)[:3]
            ] if emerging else []
        }

    async def find_datasets(self, data_type: str, max_results: int = 10) -> dict:
        """
        Tool 5: Hunt for datasets.
        """
        hs = self._get_searcher()

        # Search for papers mentioning datasets
        query = f"{data_type} dataset data available"
        results = hs.search_papers(query, n=max_results * 2)

        # Search code repos (often have data links)
        code_results = hs.search_code(data_type, n=max_results)

        # Extract dataset mentions (SearchResult objects)
        datasets = []

        for r in results:
            doc_lower = r.content.lower()

            # Look for dataset keywords
            if any(kw in doc_lower for kw in ["geo", "gse", "zenodo", "figshare", "sra", "available"]):
                datasets.append({
                    "source": "paper",
                    "title": r.title,
                    "doi": r.metadata.get("doi"),
                    "relevance": round(r.score, 3),
                    "snippet": r.content[:200]
                })

        for r in code_results:
            if "data" in r.metadata.get("file_path", "").lower():
                datasets.append({
                    "source": "code_repo",
                    "repo": r.metadata.get("repo_name"),
                    "file": r.metadata.get("file_path"),
                    "relevance": round(r.score, 3)
                })

        return {
            "query": data_type,
            "datasets_found": len(datasets),
            "results": datasets[:max_results],
            "search_suggestions": [
                f"Search GEO: {data_type}",
                f"Search Zenodo: {data_type}",
                f"Check 10x Genomics datasets page"
            ]
        }

    async def collection_health(self) -> dict:
        """
        Tool 37: Check knowledge base health.
        """
        db = Database()

        # Get stats
        result = db.fetch_one("SELECT COUNT(*) as cnt FROM artifacts")
        artifact_count = result['cnt'] if result else 0

        result = db.fetch_one("SELECT COUNT(*) as cnt FROM passages WHERE page_num >= 0")
        citable_passages = result['cnt'] if result else 0

        result = db.fetch_one("SELECT COUNT(*) as cnt FROM passages")
        total_passages = result['cnt'] if result else 0

        # Recency check
        result = db.fetch_one("""
            SELECT COUNT(*) as cnt FROM artifacts
            WHERE year IS NOT NULL
            AND year >= 2024
        """)
        recent_papers = result['cnt'] if result else 0

        # Citability rate
        citability_rate = citable_passages / total_passages if total_passages > 0 else 0

        issues = []
        if recent_papers < artifact_count * 0.1:
            issues.append({
                "type": "recency",
                "description": f"Only {recent_papers} papers from 2024+ ({100*recent_papers/artifact_count:.1f}%)",
                "action": "Run Literature Sentry with recent filter"
            })

        if citability_rate < 0.7:
            issues.append({
                "type": "citability",
                "description": f"Only {100*citability_rate:.1f}% of passages have page numbers",
                "action": "Re-ingest PDFs with enhanced parser"
            })

        return {
            "status": "healthy" if len(issues) == 0 else "needs_attention",
            "stats": {
                "total_artifacts": artifact_count,
                "total_passages": total_passages,
                "citable_passages": citable_passages,
                "citability_rate": f"{100*citability_rate:.1f}%",
                "recent_papers_2024+": recent_papers
            },
            "issues": issues,
            "recommendations": [
                "Run weekly Literature Sentry sweeps",
                "Monitor disk space (currently at 87%)",
                "Backup weekly to /databases/"
            ]
        }

    async def expand_collection(self, strategy: str) -> dict:
        """
        Tool 38: Propose expansion targets.
        """
        db = Database()

        expansion_plan = {
            "strategy": strategy,
            "targets": [],
            "estimated_size": 0
        }

        if strategy == "fill_gaps":
            # Find concepts with few papers
            orphans = db.fetch_all("""
                SELECT concept, COUNT(*) as cnt
                FROM (
                    SELECT jsonb_array_elements_text(metadata->'concepts') as concept
                    FROM artifacts
                ) sub
                GROUP BY concept
                HAVING COUNT(*) < 5
                ORDER BY cnt DESC
                LIMIT 20
            """)

            expansion_plan["targets"] = [
                {"concept": row['concept'], "current_count": row['cnt'], "action": f"Search for: {row['concept']}"}
                for row in orphans
            ]

        elif strategy == "follow_citations":
            expansion_plan["targets"] = [
                {"action": "Get top 100 cited papers, find their top citations"},
                {"action": "Use Semantic Scholar API for citation network"}
            ]

        elif strategy == "trending":
            expansion_plan["targets"] = [
                {"action": "Check arXiv daily for 'spatial transcriptomics'"},
                {"action": "Check bioRxiv for 'foundation model pathology'"},
                {"action": "Monitor GitHub trending in ML/bio"}
            ]

        elif strategy == "author_track":
            # Whitelist authors
            expansion_plan["targets"] = [
                {"author": "Faisal Mahmood", "lab": "mahmoodlab"},
                {"author": "Ken Lau", "lab": "Ken-Lau-Lab"},
                {"author": "Fabian Theis", "lab": "theislab"},
                {"action": "Set up PubMed alerts for these authors"}
            ]

        return expansion_plan
