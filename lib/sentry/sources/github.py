#!/usr/bin/env python3
"""
GitHub Source Connector

For code implementations. Velocity (recent commits) matters as much as stars.
"""

import time
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging

try:
    import httpx
except ImportError:
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "pip", "install", "httpx", "-q"])
    import httpx

from ..config import RATE_LIMITS, GITHUB_TOKEN

logger = logging.getLogger(__name__)

BASE_URL = "https://api.github.com"


class GitHubSource:
    """
    GitHub Search API connector.

    API Docs: https://docs.github.com/en/rest/search
    """

    name = "github"
    rate_limit = RATE_LIMITS.get("github", 0.5)

    def __init__(self, token: str = None):
        self.token = token or GITHUB_TOKEN
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "LiteratureSentry/1.0",
        }
        if self.token:
            headers["Authorization"] = f"token {self.token}"

        self.client = httpx.Client(timeout=30, headers=headers)
        self.last_request = 0

    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self.last_request
        min_interval = 1.0 / self.rate_limit
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self.last_request = time.time()

    def discover(
        self,
        query: str,
        min_stars: int = 50,
        max_results: int = 50,
        language: str = None,
        sort: str = "stars",
        pushed_after: int = 180,  # Days - filter dead repos
    ) -> List[Dict]:
        """
        Search GitHub for repositories.

        Args:
            query: Search query
            min_stars: Minimum star count
            max_results: Maximum results
            language: Filter by language (Python, R, etc.)
            sort: stars, forks, updated
            pushed_after: Only repos pushed to in last N days

        Returns:
            List of repo dicts
        """
        self._rate_limit()

        # Build query
        search_query = query

        if min_stars > 0:
            search_query += f" stars:>={min_stars}"

        if language:
            search_query += f" language:{language}"

        if pushed_after:
            since_date = (datetime.now() - timedelta(days=pushed_after)).strftime("%Y-%m-%d")
            search_query += f" pushed:>={since_date}"

        params = {
            "q": search_query,
            "sort": sort,
            "order": "desc",
            "per_page": min(max_results, 100),
        }

        try:
            response = self.client.get(f"{BASE_URL}/search/repositories", params=params)
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("items", []):
                parsed = self._parse_repo(item)
                if parsed:
                    results.append(parsed)

            logger.info(f"GitHub: Found {len(results)} results for '{query}'")
            return results

        except Exception as e:
            logger.error(f"GitHub search failed: {e}")
            return []

    def get_repo(self, owner: str, repo: str) -> Optional[Dict]:
        """Get repository details."""
        self._rate_limit()

        try:
            response = self.client.get(f"{BASE_URL}/repos/{owner}/{repo}")
            response.raise_for_status()
            return self._parse_repo(response.json())

        except Exception as e:
            logger.error(f"GitHub repo lookup failed: {e}")
            return None

    def get_readme(self, owner: str, repo: str) -> Optional[str]:
        """Get repository README content."""
        self._rate_limit()

        try:
            response = self.client.get(
                f"{BASE_URL}/repos/{owner}/{repo}/readme",
                headers={"Accept": "application/vnd.github.raw"}
            )
            if response.status_code == 200:
                return response.text
            return None

        except Exception:
            return None

    def get_recent_commits(self, owner: str, repo: str, days: int = 30) -> int:
        """Get count of recent commits (velocity indicator)."""
        self._rate_limit()

        since_date = (datetime.now() - timedelta(days=days)).isoformat()

        try:
            response = self.client.get(
                f"{BASE_URL}/repos/{owner}/{repo}/commits",
                params={"since": since_date, "per_page": 100}
            )
            if response.status_code == 200:
                return len(response.json())
            return 0

        except Exception:
            return 0

    def _parse_repo(self, item: Dict) -> Optional[Dict]:
        """Parse GitHub repo to standard format."""
        try:
            # Parse dates
            created_at = None
            pushed_at = None
            updated_at = None

            if item.get("created_at"):
                try:
                    created_at = datetime.fromisoformat(
                        item["created_at"].replace("Z", "+00:00")
                    ).date()
                except ValueError:
                    pass

            if item.get("pushed_at"):
                try:
                    pushed_at = datetime.fromisoformat(
                        item["pushed_at"].replace("Z", "+00:00")
                    ).date()
                except ValueError:
                    pass

            if item.get("updated_at"):
                try:
                    updated_at = datetime.fromisoformat(
                        item["updated_at"].replace("Z", "+00:00")
                    ).date()
                except ValueError:
                    pass

            # Calculate velocity score (recent activity)
            velocity_score = 0.0
            if pushed_at:
                days_since_push = (datetime.now().date() - pushed_at).days
                if days_since_push < 7:
                    velocity_score = 1.0
                elif days_since_push < 30:
                    velocity_score = 0.8
                elif days_since_push < 90:
                    velocity_score = 0.5
                elif days_since_push < 180:
                    velocity_score = 0.2
                else:
                    velocity_score = 0.0

            # Extract owner
            owner = item.get("owner", {})
            owner_login = owner.get("login", "") if isinstance(owner, dict) else ""

            return {
                "source": "github",
                "external_id": item.get("full_name", ""),
                "full_name": item.get("full_name", ""),
                "owner": owner_login,
                "title": item.get("name", ""),
                "description": item.get("description", "") or "",
                "url": item.get("html_url", ""),
                "clone_url": item.get("clone_url", ""),
                "language": item.get("language"),
                "stars": item.get("stargazers_count", 0),
                "stargazers_count": item.get("stargazers_count", 0),
                "forks": item.get("forks_count", 0),
                "forks_count": item.get("forks_count", 0),
                "watchers": item.get("watchers_count", 0),
                "open_issues": item.get("open_issues_count", 0),
                "topics": item.get("topics", []),
                "license": item.get("license", {}).get("spdx_id") if item.get("license") else None,
                "created_at": created_at,
                "pushed_at": pushed_at,
                "updated_at": updated_at,
                "last_commit": str(pushed_at) if pushed_at else None,
                "published_at": created_at,  # For consistency with paper sources
                "velocity_score": velocity_score,
                "is_open_access": True,  # GitHub is always accessible
                "authors": [owner_login] if owner_login else [],
            }

        except Exception as e:
            logger.error(f"Failed to parse GitHub repo: {e}")
            return None

    def close(self):
        """Close HTTP client."""
        self.client.close()


# Convenience function
def search_github(
    query: str,
    min_stars: int = 50,
    max_results: int = 50,
    language: str = "Python"
) -> List[Dict]:
    """Quick search function."""
    source = GitHubSource()
    try:
        return source.discover(query, min_stars, max_results, language)
    finally:
        source.close()
