#!/usr/bin/env python3
"""
Literature Sentry Orchestrator

The brain that coordinates discovery, scoring, fetching, and ingestion.
"""

import sys
import json
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Set, Any
from datetime import datetime
from dataclasses import asdict
import logging

# Ensure imports work
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import psycopg2
    from psycopg2.extras import Json, execute_values
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "psycopg2-binary", "-q"])
    import psycopg2
    from psycopg2.extras import Json, execute_values

from .scoring import Scorer, ScoredItem, normalize_concept_name
from .fetcher import Fetcher, FetchResult
from .sources.europepmc import EuropePMCSource
from .sources.arxiv import ArxivSource
from .sources.biorxiv import BioRxivSource, MedRxivSource
from .sources.github import GitHubSource
from .sources.openalex import OpenAlexSource
from .sources.semanticscholar import SemanticScholarSource

logger = logging.getLogger(__name__)


class Sentry:
    """
    Literature Sentry - Autonomous polymathic resource curator.

    Orchestrates:
    1. Discovery - Search multiple sources
    2. Scoring - Apply quality gates and EIG
    3. Fetching - Download content
    4. Ingestion - Add to knowledge graph
    """

    def __init__(self, db_url: str = None):
        """Initialize Sentry with database connection."""
        self.db_url = db_url
        self.conn = self._get_connection()
        self.scorer = Scorer()
        self.fetcher = Fetcher()

        # Source connectors - OpenAlex first for comprehensive metadata discovery
        self.sources = {
            "openalex": OpenAlexSource(),  # 250M+ works, global metadata
            "semanticscholar": SemanticScholarSource(),  # 200M+ papers, citation graphs
            "europepmc": EuropePMCSource(),  # Full-text availability
            "arxiv": ArxivSource(),
            "biorxiv": BioRxivSource(),
            "medrxiv": MedRxivSource(),
            "github": GitHubSource(),
        }

    def _get_connection(self):
        """Get database connection."""
        try:
            return psycopg2.connect(
                dbname='polymath',
                user='polymath',
                host='/var/run/postgresql'
            )
        except Exception as e:
            logger.warning(f"Database connection failed: {e}")
            return None

    def discover(
        self,
        query: str,
        sources: List[str] = None,
        max_per_source: int = 20,
        min_stars: int = None,
    ) -> List[Dict]:
        """
        Discover resources across sources.

        Args:
            query: Search query
            sources: List of sources to search (None = all)
            max_per_source: Max results per source
            min_stars: Min GitHub stars (GitHub only)

        Returns:
            List of discovered items
        """
        sources = sources or list(self.sources.keys())
        all_results = []

        for source_name in sources:
            if source_name not in self.sources:
                logger.warning(f"Unknown source: {source_name}")
                continue

            source = self.sources[source_name]

            try:
                if source_name == "github":
                    results = source.discover(
                        query,
                        min_stars=min_stars or 50,
                        max_results=max_per_source
                    )
                elif source_name == "arxiv":
                    results = source.discover(query, max_results=max_per_source)
                elif source_name in ("biorxiv", "medrxiv"):
                    results = source.discover(query, max_results=max_per_source, days_back=90)
                else:
                    results = source.discover(query, max_results=max_per_source)

                all_results.extend(results)
                logger.info(f"Discovered {len(results)} from {source_name}")

            except Exception as e:
                logger.error(f"Discovery failed for {source_name}: {e}")

        return all_results

    def score_all(self, items: List[Dict], query: str = "") -> List[Dict]:
        """
        Score all discovered items.

        Returns items enriched with scores.
        """
        scored_items = []

        for item in items:
            try:
                scored = self.scorer.score(item, query)

                # Merge scores back into item dict
                item["priority_score"] = scored.priority_score
                item["bridge_score"] = scored.bridge_score
                item["is_hidden_gem"] = scored.is_hidden_gem
                item["is_trusted_lab"] = scored.is_trusted_lab
                item["concept_domains"] = [
                    normalize_concept_name(c) for c in scored.concept_domains
                ]

                scored_items.append(item)

            except Exception as e:
                logger.error(f"Scoring failed for {item.get('title', '?')}: {e}")
                item["priority_score"] = 0.0
                scored_items.append(item)

        # Sort by score
        return sorted(scored_items, key=lambda x: x.get("priority_score", 0), reverse=True)

    def fetch_and_ingest(self, item: Dict) -> Dict:
        """
        Fetch and ingest a single item.

        Returns result dict with status.
        """
        # Check if already in database
        if self._is_duplicate(item):
            return {"status": "duplicate", "item": item}

        # Record in source_items
        source_item_id = self._record_discovery(item)

        # Fetch content
        fetch_result = self.fetcher.fetch(item)

        if fetch_result.status == "success":
            # Ingest to knowledge graph
            ingest_result = self._ingest_content(item, fetch_result, source_item_id)

            # Update status
            self._update_status(source_item_id, "ingested")

            return {"status": "ingested", "item": item, "chunks": ingest_result.get("chunks", 0)}

        elif fetch_result.status == "paywalled":
            # Create ticket for manual download
            self._create_ticket(source_item_id, item, fetch_result.error or "paywalled")
            self._update_status(source_item_id, "needs_user")

            return {"status": "paywalled", "item": item}

        else:
            # Failed
            self._update_status(source_item_id, "failed", fetch_result.error)
            return {"status": "failed", "item": item, "error": fetch_result.error}

    def _is_duplicate(self, item: Dict) -> bool:
        """Check if item already exists in database."""
        if not self.conn:
            return False

        try:
            cur = self.conn.cursor()
            cur.execute(
                "SELECT id FROM source_items WHERE source = %s AND external_id = %s",
                (item.get("source"), item.get("external_id"))
            )
            result = cur.fetchone()
            cur.close()
            return result is not None

        except Exception:
            return False

    def _record_discovery(self, item: Dict) -> Optional[int]:
        """Record discovered item in database."""
        if not self.conn:
            return None

        try:
            cur = self.conn.cursor()

            # Build meta_json with scores
            meta = {
                "priority_score": item.get("priority_score", 0),
                "bridge_score": item.get("bridge_score", 0),
                "is_hidden_gem": item.get("is_hidden_gem", False),
                "is_trusted_lab": item.get("is_trusted_lab", False),
                "concept_domains": item.get("concept_domains", []),
                "citations": item.get("citations", 0),
                "stars": item.get("stars", 0),
            }

            cur.execute("""
                INSERT INTO source_items (source, external_id, url, title, authors, doi, meta_json, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, 'discovered')
                ON CONFLICT (source, external_id) DO UPDATE
                SET meta_json = EXCLUDED.meta_json, updated_at = NOW()
                RETURNING id
            """, (
                item.get("source"),
                item.get("external_id"),
                item.get("url"),
                item.get("title"),
                item.get("authors", []),
                item.get("doi"),
                Json(meta),
            ))

            result = cur.fetchone()
            self.conn.commit()
            cur.close()

            return result[0] if result else None

        except Exception as e:
            logger.error(f"Failed to record discovery: {e}")
            self.conn.rollback()
            return None

    def _update_status(self, source_item_id: int, status: str, error: str = None):
        """Update item status in database."""
        if not self.conn or not source_item_id:
            return

        try:
            cur = self.conn.cursor()
            cur.execute(
                "UPDATE source_items SET status = %s, last_error = %s, updated_at = NOW() WHERE id = %s",
                (status, error, source_item_id)
            )
            self.conn.commit()
            cur.close()

        except Exception as e:
            logger.error(f"Failed to update status: {e}")
            self.conn.rollback()

    def _create_ticket(self, source_item_id: int, item: Dict, reason: str):
        """Create a needs_user_tickets entry."""
        if not self.conn:
            return

        try:
            cur = self.conn.cursor()
            cur.execute("""
                INSERT INTO needs_user_tickets (source_item_id, doi, title, url, reason)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (
                source_item_id,
                item.get("doi"),
                item.get("title"),
                item.get("url"),
                reason,
            ))
            self.conn.commit()
            cur.close()

        except Exception as e:
            logger.error(f"Failed to create ticket: {e}")
            self.conn.rollback()

    def _ingest_content(
        self,
        item: Dict,
        fetch_result: FetchResult,
        source_item_id: Optional[int] = None,
    ) -> Dict:
        """Ingest fetched content to knowledge graph."""
        # Import unified ingestor
        try:
            from lib.unified_ingest import UnifiedIngestor

            ingestor = UnifiedIngestor()

            source = item.get("source")

            if source == "github":
                # For GitHub, content is already flattened text
                # Save to temp file and ingest as code
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(fetch_result.content)
                    temp_path = f.name

                result = ingestor.ingest_code(
                    temp_path,
                    repo_name=item.get("full_name", ""),
                    ingestion_method="sentry_code_ingest",
                    source_item_id=source_item_id,
                )
                Path(temp_path).unlink()  # Clean up

                return {"chunks": result.chunks_added if result else 0}

            elif fetch_result.file_path:
                # PDF file
                result = ingestor.ingest_pdf(
                    fetch_result.file_path,
                    ingestion_method="sentry_pdf_ingest",
                    source_item_id=source_item_id,
                )
                return {"chunks": result.chunks_added if result else 0}

            else:
                return {"chunks": 0}

        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            return {"chunks": 0, "error": str(e)}

    def get_pending_tickets(self, limit: int = 20) -> List[Dict]:
        """Get pending tickets that need manual download."""
        if not self.conn:
            return []

        try:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT id, doi, title, url, reason, created_at
                FROM needs_user_tickets
                WHERE resolved = FALSE
                ORDER BY created_at DESC
                LIMIT %s
            """, (limit,))

            columns = ["id", "doi", "title", "url", "reason", "created_at"]
            results = [dict(zip(columns, row)) for row in cur.fetchall()]
            cur.close()

            return results

        except Exception as e:
            logger.error(f"Failed to get tickets: {e}")
            return []

    def resolve_ticket(self, ticket_id: str):
        """Mark a ticket as resolved."""
        if not self.conn:
            return

        try:
            cur = self.conn.cursor()
            cur.execute(
                "UPDATE needs_user_tickets SET resolved = TRUE, resolved_at = NOW() WHERE id = %s",
                (ticket_id,)
            )
            self.conn.commit()
            cur.close()

        except Exception as e:
            logger.error(f"Failed to resolve ticket: {e}")
            self.conn.rollback()

    def get_stats(self) -> Dict:
        """Get sentry statistics."""
        if not self.conn:
            return {
                "total_discovered": 0,
                "ingested": 0,
                "pending": 0,
                "paywalled": 0,
                "failed": 0,
                "hidden_gems": 0,
                "by_source": {},
            }

        try:
            cur = self.conn.cursor()

            # Total counts by status
            cur.execute("""
                SELECT status, COUNT(*) FROM source_items GROUP BY status
            """)
            status_counts = dict(cur.fetchall())

            # Hidden gems
            cur.execute("""
                SELECT COUNT(*) FROM source_items
                WHERE (meta_json->>'is_hidden_gem')::boolean = true
            """)
            hidden_gems = cur.fetchone()[0]

            # By source
            cur.execute("""
                SELECT source, COUNT(*) FROM source_items GROUP BY source
            """)
            by_source = dict(cur.fetchall())

            cur.close()

            return {
                "total_discovered": sum(status_counts.values()),
                "ingested": status_counts.get("ingested", 0),
                "pending": status_counts.get("discovered", 0),
                "paywalled": status_counts.get("needs_user", 0),
                "failed": status_counts.get("failed", 0),
                "hidden_gems": hidden_gems,
                "by_source": by_source,
            }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}

    def close(self):
        """Close all connections."""
        for source in self.sources.values():
            source.close()
        if self.conn:
            self.conn.close()
