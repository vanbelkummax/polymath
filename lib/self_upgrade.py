#!/usr/bin/env python3
"""
Polymath Self-Upgrade System

Detects gaps in Polymath's knowledge and capabilities, then automatically
addresses them through ingestion, skill creation, or user prompting.

Gap types:
- missing_papers: Papers cited by our corpus but not in it
- missing_capability: Needed MCP/API but not available
- missing_skill: Claude Code skill for common workflow
- reasoning_failure: Incorrect answer due to retrieval/reasoning gap
- stale_data: Data that needs refresh (grants, preprints, etc.)
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any
import psycopg2

from lib.db import get_db_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GapType(Enum):
    MISSING_PAPERS = "missing_papers"
    MISSING_CAPABILITY = "missing_capability"
    MISSING_SKILL = "missing_skill"
    REASONING_FAILURE = "reasoning_failure"
    STALE_DATA = "stale_data"
    MISSING_METADATA = "missing_metadata"


class GapPriority(Enum):
    CRITICAL = 1  # Blocking user task
    HIGH = 2      # Frequent or impactful
    MEDIUM = 3    # Would improve quality
    LOW = 4       # Nice to have


@dataclass
class Gap:
    """A detected gap in Polymath's capabilities"""
    gap_type: GapType
    priority: GapPriority
    description: str
    evidence: Dict[str, Any]
    detected_at: datetime = field(default_factory=datetime.now)
    resolution: Optional[str] = None
    resolved_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "gap_type": self.gap_type.value,
            "priority": self.priority.value,
            "description": self.description,
            "evidence": self.evidence,
            "detected_at": self.detected_at.isoformat(),
            "resolution": self.resolution,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None
        }


@dataclass
class UpgradeResult:
    """Result of an upgrade attempt"""
    success: bool
    gap: Gap
    action_taken: str
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class GapDetector:
    """Identifies gaps in Polymath's knowledge and capabilities"""

    def __init__(self):
        self.conn = get_db_connection()

    def detect_all_gaps(self) -> List[Gap]:
        """Run all gap detection methods"""
        gaps = []

        # Run detectors
        gaps.extend(self.detect_missing_keystone_papers())
        gaps.extend(self.detect_missing_metadata())
        gaps.extend(self.detect_stale_data())
        gaps.extend(self.detect_missing_capabilities())

        # Sort by priority
        gaps.sort(key=lambda g: g.priority.value)

        return gaps

    def detect_missing_keystone_papers(self, min_citations: int = 3) -> List[Gap]:
        """Find papers frequently cited by our corpus but not in it"""
        gaps = []

        try:
            with self.conn.cursor() as cur:
                # Check if citation_graph exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.tables
                        WHERE table_name = 'citation_graph'
                    )
                """)
                if not cur.fetchone()[0]:
                    return []

                # Get missing keystones
                cur.execute("""
                    SELECT
                        target_openalex_id,
                        COUNT(DISTINCT source_doc_id) as cited_by_count
                    FROM citation_graph
                    WHERE target_doc_id IS NULL
                    GROUP BY target_openalex_id
                    HAVING COUNT(DISTINCT source_doc_id) >= %s
                    ORDER BY cited_by_count DESC
                    LIMIT 50
                """, (min_citations,))

                for openalex_id, cite_count in cur.fetchall():
                    gaps.append(Gap(
                        gap_type=GapType.MISSING_PAPERS,
                        priority=GapPriority.HIGH if cite_count >= 10 else GapPriority.MEDIUM,
                        description=f"Missing keystone paper cited by {cite_count} papers in corpus",
                        evidence={
                            "openalex_id": openalex_id,
                            "cited_by_count": cite_count
                        }
                    ))

        except Exception as e:
            logger.warning(f"Error detecting missing keystones: {e}")

        return gaps

    def detect_missing_metadata(self) -> List[Gap]:
        """Find documents missing critical metadata (DOI, title, etc.)"""
        gaps = []

        try:
            with self.conn.cursor() as cur:
                # Count documents with missing metadata
                cur.execute("""
                    SELECT
                        COUNT(*) FILTER (WHERE doi IS NULL) as missing_doi,
                        COUNT(*) FILTER (WHERE title IS NULL OR title = '') as missing_title,
                        COUNT(*) FILTER (WHERE year IS NULL) as missing_year,
                        COUNT(*) FILTER (WHERE authors IS NULL OR cardinality(authors) = 0) as missing_authors,
                        COUNT(*) as total
                    FROM documents
                """)
                row = cur.fetchone()

                if row:
                    missing_doi, missing_title, missing_year, missing_authors, total = row

                    if missing_doi > 0:
                        pct = (missing_doi / total) * 100
                        gaps.append(Gap(
                            gap_type=GapType.MISSING_METADATA,
                            priority=GapPriority.MEDIUM,
                            description=f"{missing_doi:,} documents ({pct:.1f}%) missing DOI",
                            evidence={"missing_count": missing_doi, "total": total}
                        ))

                    if missing_year > 0:
                        pct = (missing_year / total) * 100
                        gaps.append(Gap(
                            gap_type=GapType.MISSING_METADATA,
                            priority=GapPriority.LOW,
                            description=f"{missing_year:,} documents ({pct:.1f}%) missing publication year",
                            evidence={"missing_count": missing_year, "total": total}
                        ))

        except Exception as e:
            logger.warning(f"Error detecting missing metadata: {e}")

        return gaps

    def detect_stale_data(self) -> List[Gap]:
        """Find data sources that need refresh"""
        gaps = []

        try:
            with self.conn.cursor() as cur:
                # Check when we last ingested papers
                cur.execute("""
                    SELECT MAX(created_at) FROM documents
                """)
                last_ingest = cur.fetchone()[0]

                if last_ingest:
                    days_ago = (datetime.now(last_ingest.tzinfo) - last_ingest).days
                    if days_ago > 7:
                        gaps.append(Gap(
                            gap_type=GapType.STALE_DATA,
                            priority=GapPriority.MEDIUM,
                            description=f"No new papers ingested in {days_ago} days",
                            evidence={"last_ingest": last_ingest.isoformat(), "days_ago": days_ago}
                        ))

                # Check OpenAlex enrichment freshness
                cur.execute("""
                    SELECT
                        COUNT(*) FILTER (WHERE openalex_enriched_at IS NOT NULL) as enriched,
                        COUNT(*) as total
                    FROM documents
                """)
                enriched, total = cur.fetchone()

                if total > 0 and enriched < total * 0.5:
                    gaps.append(Gap(
                        gap_type=GapType.STALE_DATA,
                        priority=GapPriority.MEDIUM,
                        description=f"Only {enriched}/{total} documents have OpenAlex enrichment",
                        evidence={"enriched": enriched, "total": total}
                    ))

        except Exception as e:
            logger.warning(f"Error detecting stale data: {e}")

        return gaps

    def detect_missing_capabilities(self) -> List[Gap]:
        """Detect missing MCPs or APIs based on common query patterns"""
        gaps = []

        # Check for common capabilities we should have
        capabilities = [
            ("NIH_REPORTER_API_KEY", "NIH Reporter grants API", "Grant search and funding analysis"),
            ("BRAVE_API_KEY", "Brave Search API", "Web search for current information"),
            ("SEMANTIC_SCHOLAR_API_KEY", "Semantic Scholar API", "Paper metadata and citations"),
            ("OPENALEX_EMAIL", "OpenAlex polite pool", "Faster API access with email"),
        ]

        for env_var, name, description in capabilities:
            if not os.getenv(env_var):
                gaps.append(Gap(
                    gap_type=GapType.MISSING_CAPABILITY,
                    priority=GapPriority.LOW,
                    description=f"Missing {name}",
                    evidence={
                        "env_var": env_var,
                        "capability": name,
                        "benefit": description
                    }
                ))

        return gaps

    def detect_from_query_failure(
        self,
        query: str,
        retrieval_score: float,
        expected_topics: List[str] = None
    ) -> Optional[Gap]:
        """Detect gap from a failed query"""

        if retrieval_score < 0.3:
            return Gap(
                gap_type=GapType.REASONING_FAILURE,
                priority=GapPriority.HIGH,
                description=f"Low retrieval score ({retrieval_score:.2f}) for query",
                evidence={
                    "query": query,
                    "retrieval_score": retrieval_score,
                    "expected_topics": expected_topics or []
                }
            )

        return None


class SelfUpgrader:
    """Automatically addresses detected gaps"""

    def __init__(self, polymath_repo_path: str = "/home/user/polymath-repo"):
        self.repo_path = Path(polymath_repo_path)
        self.conn = get_db_connection()
        self.gap_log_path = self.repo_path / "data" / "gap_log.jsonl"
        self.gap_log_path.parent.mkdir(parents=True, exist_ok=True)

    def upgrade(self, gap: Gap) -> UpgradeResult:
        """Attempt to address a gap"""

        logger.info(f"Attempting to address gap: {gap.description}")

        try:
            match gap.gap_type:
                case GapType.MISSING_PAPERS:
                    return self._handle_missing_papers(gap)

                case GapType.MISSING_METADATA:
                    return self._handle_missing_metadata(gap)

                case GapType.MISSING_CAPABILITY:
                    return self._handle_missing_capability(gap)

                case GapType.STALE_DATA:
                    return self._handle_stale_data(gap)

                case GapType.REASONING_FAILURE:
                    return self._handle_reasoning_failure(gap)

                case _:
                    return UpgradeResult(
                        success=False,
                        gap=gap,
                        action_taken="unknown",
                        error=f"Unknown gap type: {gap.gap_type}"
                    )

        except Exception as e:
            logger.exception(f"Error upgrading gap: {e}")
            return UpgradeResult(
                success=False,
                gap=gap,
                action_taken="error",
                error=str(e)
            )
        finally:
            # Log the gap
            self._log_gap(gap)

    def _handle_missing_papers(self, gap: Gap) -> UpgradeResult:
        """Handle missing keystone papers"""
        openalex_id = gap.evidence.get("openalex_id")

        if not openalex_id:
            return UpgradeResult(
                success=False,
                gap=gap,
                action_taken="skip",
                error="No OpenAlex ID in evidence"
            )

        # Create a ticket for manual ingestion
        # (Automatic ingestion would require PDF access)
        return UpgradeResult(
            success=True,
            gap=gap,
            action_taken="ticket_created",
            details={
                "ticket_type": "paper_ingestion",
                "openalex_id": openalex_id,
                "priority": gap.priority.name
            }
        )

    def _handle_missing_metadata(self, gap: Gap) -> UpgradeResult:
        """Handle documents with missing metadata"""
        # Point to enrichment scripts
        return UpgradeResult(
            success=True,
            gap=gap,
            action_taken="remediation_suggested",
            details={
                "scripts": [
                    "scripts/enrich_openalex.py --missing-only",
                    "scripts/enrich_with_llm.py"
                ]
            }
        )

    def _handle_missing_capability(self, gap: Gap) -> UpgradeResult:
        """Handle missing MCP/API capability"""
        env_var = gap.evidence.get("env_var")

        return UpgradeResult(
            success=True,
            gap=gap,
            action_taken="user_input_required",
            details={
                "message": f"Please add {env_var} to .env file",
                "env_var": env_var,
                "benefit": gap.evidence.get("benefit")
            }
        )

    def _handle_stale_data(self, gap: Gap) -> UpgradeResult:
        """Handle stale data by suggesting refresh commands"""
        return UpgradeResult(
            success=True,
            gap=gap,
            action_taken="refresh_suggested",
            details={
                "commands": [
                    "python3 -m lib.sentry.cli 'spatial transcriptomics' --max 50",
                    "python3 scripts/enrich_openalex.py --missing-only"
                ]
            }
        )

    def _handle_reasoning_failure(self, gap: Gap) -> UpgradeResult:
        """Handle reasoning/retrieval failure"""
        # Log for analysis
        return UpgradeResult(
            success=True,
            gap=gap,
            action_taken="logged_for_analysis",
            details={
                "query": gap.evidence.get("query"),
                "score": gap.evidence.get("retrieval_score")
            }
        )

    def _log_gap(self, gap: Gap):
        """Append gap to log file"""
        with open(self.gap_log_path, "a") as f:
            f.write(json.dumps(gap.to_dict()) + "\n")

    def get_gap_statistics(self) -> Dict[str, Any]:
        """Get statistics on detected gaps"""
        stats = {
            "by_type": {},
            "by_priority": {},
            "total": 0,
            "resolved": 0
        }

        if not self.gap_log_path.exists():
            return stats

        with open(self.gap_log_path) as f:
            for line in f:
                try:
                    gap = json.loads(line.strip())
                    gap_type = gap.get("gap_type", "unknown")
                    priority = gap.get("priority", 4)

                    stats["by_type"][gap_type] = stats["by_type"].get(gap_type, 0) + 1
                    stats["by_priority"][priority] = stats["by_priority"].get(priority, 0) + 1
                    stats["total"] += 1

                    if gap.get("resolution"):
                        stats["resolved"] += 1

                except json.JSONDecodeError:
                    continue

        return stats


class ClaudeMdManager:
    """Manages CLAUDE.md knowledge retention with minimal context"""

    def __init__(self, claude_md_path: str = "/home/user/CLAUDE.md"):
        self.path = Path(claude_md_path)
        self.max_section_size = 2000  # Characters per section

    def should_update(self, learning: Dict[str, Any]) -> bool:
        """Determine if learning is important enough to persist"""
        criteria = [
            learning.get("reuse_potential", 0) > 0.7,
            learning.get("affects_future_tasks", False),
            not self._already_documented(learning)
        ]
        return all(criteria)

    def _already_documented(self, learning: Dict[str, Any]) -> bool:
        """Check if learning is already in CLAUDE.md"""
        if not self.path.exists():
            return False

        content = self.path.read_text()
        key_terms = learning.get("key_terms", [])

        # Check if key terms are already present
        for term in key_terms:
            if term.lower() in content.lower():
                return True

        return False

    def update(self, learning: Dict[str, Any]) -> bool:
        """Add learning with minimal context"""
        if not self.should_update(learning):
            return False

        try:
            content = self.path.read_text() if self.path.exists() else ""

            # Format the learning entry
            entry = self._format_entry(learning)

            # Find or create the appropriate section
            section = learning.get("category", "Recent Learnings")
            content = self._append_to_section(content, section, entry)

            self.path.write_text(content)
            logger.info(f"Updated CLAUDE.md with: {learning.get('summary', 'new learning')[:50]}")
            return True

        except Exception as e:
            logger.error(f"Failed to update CLAUDE.md: {e}")
            return False

    def _format_entry(self, learning: Dict[str, Any]) -> str:
        """Format learning as minimal CLAUDE.md entry"""
        summary = learning.get("summary", "")
        details = learning.get("details", "")

        # Keep it brief
        if len(details) > 100:
            details = details[:97] + "..."

        timestamp = datetime.now().strftime("%Y-%m-%d")
        return f"- [{timestamp}] {summary}: {details}"

    def _append_to_section(self, content: str, section: str, entry: str) -> str:
        """Append entry to the appropriate section"""
        section_marker = f"## {section}"

        if section_marker in content:
            # Find section and append
            parts = content.split(section_marker)
            before = parts[0]
            after = parts[1] if len(parts) > 1 else ""

            # Find end of section (next ## or end of file)
            next_section = after.find("\n## ")
            if next_section > 0:
                section_content = after[:next_section]
                rest = after[next_section:]
            else:
                section_content = after
                rest = ""

            # Append entry
            section_content = section_content.rstrip() + "\n" + entry + "\n"

            return before + section_marker + section_content + rest
        else:
            # Create new section at end
            return content.rstrip() + f"\n\n{section_marker}\n\n{entry}\n"

    def periodic_cleanup(self):
        """Remove outdated info and consolidate"""
        if not self.path.exists():
            return

        content = self.path.read_text()

        # Remove entries older than 30 days (marked with dates)
        lines = content.split("\n")
        cleaned_lines = []
        cutoff = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        for line in lines:
            # Check if line has a date marker
            if line.strip().startswith("- [20"):
                try:
                    date_str = line.split("[")[1].split("]")[0]
                    if date_str < cutoff:
                        continue  # Skip old entries
                except (IndexError, ValueError):
                    pass

            cleaned_lines.append(line)

        cleaned_content = "\n".join(cleaned_lines)
        self.path.write_text(cleaned_content)
        logger.info("CLAUDE.md cleanup complete")


def run_gap_detection():
    """Run gap detection and print results"""
    detector = GapDetector()
    gaps = detector.detect_all_gaps()

    print(f"\n# Polymath Gap Analysis\n")
    print(f"Detected {len(gaps)} gaps:\n")

    current_type = None
    for gap in gaps:
        if gap.gap_type != current_type:
            current_type = gap.gap_type
            print(f"\n## {current_type.value.replace('_', ' ').title()}\n")

        priority_emoji = {
            GapPriority.CRITICAL: "🔴",
            GapPriority.HIGH: "🟠",
            GapPriority.MEDIUM: "🟡",
            GapPriority.LOW: "🟢"
        }

        print(f"{priority_emoji[gap.priority]} [{gap.priority.name}] {gap.description}")

    return gaps


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Polymath Self-Upgrade System")
    parser.add_argument("--detect", action="store_true", help="Run gap detection")
    parser.add_argument("--upgrade", action="store_true", help="Attempt to upgrade detected gaps")
    parser.add_argument("--stats", action="store_true", help="Show gap statistics")
    parser.add_argument("--cleanup-claude-md", action="store_true", help="Clean up CLAUDE.md")

    args = parser.parse_args()

    if args.detect:
        gaps = run_gap_detection()

        if args.upgrade:
            print("\n# Attempting Upgrades\n")
            upgrader = SelfUpgrader()
            for gap in gaps:
                result = upgrader.upgrade(gap)
                status = "✓" if result.success else "✗"
                print(f"{status} {gap.description}: {result.action_taken}")

    elif args.stats:
        upgrader = SelfUpgrader()
        stats = upgrader.get_gap_statistics()
        print(json.dumps(stats, indent=2))

    elif args.cleanup_claude_md:
        manager = ClaudeMdManager()
        manager.periodic_cleanup()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
