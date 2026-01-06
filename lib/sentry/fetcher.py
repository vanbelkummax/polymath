#!/usr/bin/env python3
"""
Literature Sentry Fetcher Module

Handles downloading and processing content from various sources:
- PDFs from academic sources (with Brave fallback)
- GitHub repos (flattened to LLM-readable format)
- YouTube transcripts (via yt-dlp)

Key features:
- Token-aware GitHub flattening
- Jupyter notebook parsing
- Function skeleton extraction for truncated files
"""

import os
import re
import json
import hashlib
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from fnmatch import fnmatch
import logging

logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

MAX_FILE_LINES = 500          # Truncate huge files
MAX_TOTAL_TOKENS = 50000      # ~12k words, fits in context
CHARS_PER_TOKEN = 4           # Rough estimate

# File patterns for GitHub flattening
PRIORITY_PATTERNS = [
    "README.md", "README.rst", "README.txt",
    "setup.py", "pyproject.toml", "requirements.txt",
    "**/model*.py", "**/train*.py", "**/main*.py",
    "**/config*.py", "**/utils*.py",
    "**/*.ipynb",
    "src/**/*.py", "lib/**/*.py",
]

SKIP_PATTERNS = [
    "__pycache__/*", ".git/*", "node_modules/*", ".venv/*", "venv/*",
    "dist/*", "build/*", ".next/*", "target/*", ".cargo/*",
    "vendor/*", ".pytest_cache/*", ".mypy_cache/*",
    "*.lock", "*.min.js", "*.map", "*.pyc", "*.pyo",
    "test_*.py", "*_test.py", "tests/fixtures/*", "tests/data/*",
    "*.egg-info/*", ".eggs/*", "*.so", "*.dylib",
    "migrations/*", "**/migrations/*", "alembic/*",  # DB migrations
    "*.min.css", "*.bundle.js", "package-lock.json",  # Bundled assets
]

CODE_EXTENSIONS = {".py", ".ts", ".tsx", ".js", ".jsx", ".rs", ".go", ".r", ".R"}
DOC_EXTENSIONS = {".md", ".rst", ".txt"}


@dataclass
class FetchResult:
    """Result of fetching content."""
    status: str  # success, paywalled, failed
    content: Optional[str] = None
    file_path: Optional[str] = None
    file_size: int = 0
    sha256: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class GitHubFlattener:
    """
    Flatten a GitHub repository into a single, token-efficient document.

    Output format:
    ===== REPO: owner/name =====
    Stars: 234 | Language: Python | Last commit: 2025-12-15

    ----- README.md -----
    [full readme content]

    ----- STRUCTURE -----
    src/
      model.py (234 lines)
      train.py (156 lines)
    ...

    ----- KEY FILES -----
    ## src/model.py
    ```python
    [content]
    ```
    """

    def __init__(self, max_tokens: int = MAX_TOTAL_TOKENS, max_file_lines: int = MAX_FILE_LINES):
        self.max_tokens = max_tokens
        self.max_file_lines = max_file_lines

    def flatten(self, repo_path: str, metadata: Dict = None) -> str:
        """
        Flatten repository to single document.

        Args:
            repo_path: Path to cloned repository
            metadata: Optional metadata (stars, language, etc.)

        Returns:
            Flattened text representation
        """
        repo_path = Path(repo_path)
        if not repo_path.exists():
            return f"[Error: Repository path not found: {repo_path}]"

        parts = []
        metadata = metadata or {}

        # 1. Header with metadata
        parts.append(self._build_header(repo_path, metadata))

        # 2. README (always full)
        readme = self._get_readme(repo_path)
        if readme:
            parts.append(f"----- README -----\n{readme}")

        # 3. Structure tree
        tree = self._build_tree(repo_path)
        parts.append(f"----- STRUCTURE -----\n{tree}")

        # 4. Key files, prioritized, token-budgeted
        parts.append("----- KEY FILES -----")

        token_budget = self.max_tokens - self._estimate_tokens("\n\n".join(parts))
        key_files = self._select_key_files(repo_path)

        for fpath, priority in key_files:
            if token_budget <= 0:
                parts.append("\n[... token budget exhausted, remaining files skipped ...]")
                break

            content = self._read_file(fpath)
            tokens_needed = self._estimate_tokens(content)

            if tokens_needed > token_budget:
                # Try skeleton instead
                skeleton = self._extract_skeleton(fpath)
                if skeleton:
                    rel_path = fpath.relative_to(repo_path)
                    parts.append(f"\n## {rel_path} (SKELETON - full file too large)")
                    parts.append(f"```\n{skeleton}\n```")
                    token_budget -= self._estimate_tokens(skeleton)
                continue

            rel_path = fpath.relative_to(repo_path)
            ext = fpath.suffix
            lang = self._get_language(ext)

            parts.append(f"\n## {rel_path}")
            parts.append(f"```{lang}\n{content}\n```")
            token_budget -= tokens_needed

        return "\n\n".join(parts)

    def _build_header(self, repo_path: Path, metadata: Dict) -> str:
        """Build repository header."""
        name = metadata.get("full_name", repo_path.name)
        stars = metadata.get("stars", "?")
        language = metadata.get("language", "?")
        last_commit = metadata.get("last_commit", "?")

        return f"""===== REPO: {name} =====
Stars: {stars} | Language: {language} | Last commit: {last_commit}"""

    def _get_readme(self, repo_path: Path) -> Optional[str]:
        """Get README content."""
        for name in ["README.md", "README.rst", "README.txt", "README"]:
            readme_path = repo_path / name
            if readme_path.exists():
                try:
                    content = readme_path.read_text(errors="ignore")
                    # Limit README to reasonable size
                    if len(content) > 10000:
                        content = content[:10000] + "\n\n[... README truncated ...]"
                    return content
                except Exception:
                    pass
        return None

    def _build_tree(self, repo_path: Path, max_depth: int = 3) -> str:
        """Build directory tree with file sizes."""
        lines = []

        def walk(path: Path, prefix: str = "", depth: int = 0):
            if depth > max_depth:
                return

            try:
                items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
            except PermissionError:
                return

            for item in items:
                rel = str(item.relative_to(repo_path))

                # Skip patterns
                if any(fnmatch(rel, pat) or fnmatch(item.name, pat) for pat in SKIP_PATTERNS):
                    continue

                if item.is_dir():
                    lines.append(f"{prefix}{item.name}/")
                    walk(item, prefix + "  ", depth + 1)
                else:
                    try:
                        line_count = sum(1 for _ in open(item, errors="ignore"))
                        lines.append(f"{prefix}{item.name} ({line_count} lines)")
                    except Exception:
                        lines.append(f"{prefix}{item.name}")

        walk(repo_path)
        return "\n".join(lines[:100])  # Limit tree size

    def _select_key_files(self, repo_path: Path) -> List[Tuple[Path, int]]:
        """
        Select key files based on priority patterns.

        Returns list of (path, priority) tuples, sorted by priority.
        """
        files = []

        for pattern in PRIORITY_PATTERNS:
            if "**" in pattern:
                matches = list(repo_path.glob(pattern))
            else:
                match = repo_path / pattern
                matches = [match] if match.exists() else []

            for fpath in matches:
                if not fpath.is_file():
                    continue

                rel = str(fpath.relative_to(repo_path))

                # Skip patterns
                if any(fnmatch(rel, pat) for pat in SKIP_PATTERNS):
                    continue

                # Skip generated/low-density code
                if self._is_generated_code(fpath):
                    continue

                # Priority based on pattern order
                priority = PRIORITY_PATTERNS.index(pattern)
                files.append((fpath, priority))

        # Sort by priority, dedupe
        seen = set()
        result = []
        for fpath, priority in sorted(files, key=lambda x: x[1]):
            if fpath not in seen:
                seen.add(fpath)
                result.append((fpath, priority))

        return result

    def _is_generated_code(self, fpath: Path) -> bool:
        """
        Check if file is likely generated code (low information density).

        Heuristics:
        - Large files (>100KB) likely data/generated
        - Python files >500 lines with low cyclomatic complexity (<5 avg per function)
        - Files with "generated" or "auto" in path
        """
        # Size check
        try:
            if fpath.stat().st_size > 100_000:  # >100KB
                return True
        except Exception:
            return False

        # Path hints
        path_lower = str(fpath).lower()
        if any(hint in path_lower for hint in ["generated", "auto_", "_pb2.py", ".pb.go"]):
            return True

        # Complexity check for Python files
        if fpath.suffix == ".py":
            try:
                lines = sum(1 for _ in open(fpath, errors="ignore"))
                if lines > 500:
                    # Simple complexity heuristic: count function defs vs lines
                    content = fpath.read_text(errors="ignore")
                    num_functions = content.count("\ndef ") + content.count("\nclass ")
                    if num_functions == 0:
                        return True  # Large file with no functions = likely data
                    avg_lines_per_func = lines / num_functions
                    # If avg function is >100 lines, likely generated (e.g., migrations, data)
                    if avg_lines_per_func > 100:
                        return True
            except Exception:
                pass

        return False

    def _read_file(self, fpath: Path) -> str:
        """Read file with truncation and notebook handling."""
        ext = fpath.suffix.lower()

        # Special handling for notebooks
        if ext == ".ipynb":
            return self._read_notebook(fpath)

        try:
            lines = fpath.read_text(errors="ignore").splitlines()

            if len(lines) > self.max_file_lines:
                truncated = lines[:self.max_file_lines]
                truncated.append(f"\n[... truncated {len(lines) - self.max_file_lines} more lines ...]")
                return "\n".join(truncated)

            return "\n".join(lines)

        except Exception as e:
            return f"[Error reading file: {e}]"

    def _read_notebook(self, fpath: Path) -> str:
        """
        Parse Jupyter notebook to clean Python script format.

        Extracts code and markdown cells, discards metadata/outputs.
        """
        try:
            with open(fpath) as f:
                nb = json.load(f)

            output = []
            for i, cell in enumerate(nb.get("cells", [])):
                cell_type = cell.get("cell_type", "")
                source = "".join(cell.get("source", []))

                if not source.strip():
                    continue

                if cell_type == "code":
                    output.append(f"# %% [Cell {i+1}: CODE]")
                    output.append(source)
                elif cell_type == "markdown":
                    # Convert markdown to comments
                    commented = "\n".join(f"# {line}" for line in source.splitlines())
                    output.append(f"# %% [Cell {i+1}: MARKDOWN]")
                    output.append(commented)

            return "\n\n".join(output)

        except json.JSONDecodeError:
            return "[Error: Invalid notebook JSON]"
        except Exception as e:
            return f"[Error parsing notebook: {e}]"

    def _extract_skeleton(self, fpath: Path) -> Optional[str]:
        """
        Extract function/class signatures for truncated files.

        Provides API surface without full implementation.
        """
        ext = fpath.suffix.lower()

        if ext not in (".py",):
            return None

        try:
            content = fpath.read_text(errors="ignore")
            lines = content.splitlines()

            skeleton = []
            in_docstring = False
            docstring_char = None

            for line in lines:
                stripped = line.strip()

                # Track docstrings
                if '"""' in stripped or "'''" in stripped:
                    char = '"""' if '"""' in stripped else "'''"
                    if in_docstring:
                        if stripped.endswith(char) or stripped == char:
                            in_docstring = False
                    else:
                        in_docstring = stripped.count(char) == 1
                        docstring_char = char
                    continue

                if in_docstring:
                    continue

                # Extract definitions
                if stripped.startswith(("def ", "class ", "async def ")):
                    skeleton.append(line)
                    # Add ... for body
                    indent = len(line) - len(line.lstrip())
                    skeleton.append(" " * (indent + 4) + "...")

                elif stripped.startswith(("@",)):  # Decorators
                    skeleton.append(line)

                elif stripped.startswith(("import ", "from ")):
                    skeleton.append(line)

            return "\n".join(skeleton) if skeleton else None

        except Exception:
            return None

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate."""
        return len(text) // CHARS_PER_TOKEN

    def _get_language(self, ext: str) -> str:
        """Get language name for code fence."""
        mapping = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".jsx": "javascript",
            ".rs": "rust",
            ".go": "go",
            ".r": "r",
            ".R": "r",
            ".md": "markdown",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".toml": "toml",
        }
        return mapping.get(ext.lower(), "")


class Fetcher:
    """
    Main fetcher class for Literature Sentry.

    Handles downloading content from various sources.
    """

    def __init__(self, staging_dir: str = None, brave_api_key: str = None):
        self.staging_dir = Path(staging_dir or "/home/user/work/polymax/ingest_staging")
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.brave_api_key = brave_api_key or os.environ.get("BRAVE_API_KEY")
        self.github_flattener = GitHubFlattener()

    def fetch(self, item: Dict) -> FetchResult:
        """
        Fetch content for a discovered item.

        Args:
            item: Item dict with source, url, etc.

        Returns:
            FetchResult with content or error
        """
        source = item.get("source", "")

        if source == "github":
            return self._fetch_github(item)
        elif source in ("europepmc", "pubmed", "arxiv", "biorxiv"):
            return self._fetch_paper(item)
        elif source == "youtube":
            return self._fetch_youtube(item)
        else:
            return FetchResult(status="failed", error=f"Unknown source: {source}")

    def _fetch_github(self, item: Dict) -> FetchResult:
        """Clone and flatten a GitHub repository."""
        url = item.get("url", item.get("clone_url", ""))
        if not url:
            return FetchResult(status="failed", error="No URL provided")

        # Clone to temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            clone_path = Path(tmpdir) / "repo"

            try:
                result = subprocess.run(
                    ["git", "clone", "--depth", "1", url, str(clone_path)],
                    capture_output=True,
                    text=True,
                    timeout=120
                )

                if result.returncode != 0:
                    return FetchResult(
                        status="failed",
                        error=f"Git clone failed: {result.stderr}"
                    )

                # Flatten
                metadata = {
                    "full_name": item.get("full_name", ""),
                    "stars": item.get("stars", item.get("stargazers_count", "?")),
                    "language": item.get("language", "?"),
                    "last_commit": item.get("last_commit", item.get("pushed_at", "?")),
                }

                content = self.github_flattener.flatten(str(clone_path), metadata)

                # Compute hash
                sha256 = hashlib.sha256(content.encode()).hexdigest()

                return FetchResult(
                    status="success",
                    content=content,
                    file_size=len(content),
                    sha256=sha256,
                    metadata=metadata
                )

            except subprocess.TimeoutExpired:
                return FetchResult(status="failed", error="Git clone timed out")
            except Exception as e:
                return FetchResult(status="failed", error=str(e))

    def _fetch_paper(self, item: Dict) -> FetchResult:
        """
        Fetch a paper PDF.

        Tries:
        1. Open access URL from source
        2. Unpaywall lookup
        3. Brave search fallback
        4. If all fail, mark as paywalled
        """
        doi = item.get("doi")
        url = item.get("oa_url") or item.get("pdf_url") or item.get("url")
        title = item.get("title", "")

        # Try direct URL first
        if url and self._is_pdf_url(url):
            result = self._download_pdf(url, title)
            if result.status == "success":
                return result

        # Try Unpaywall
        if doi:
            unpaywall_url = self._lookup_unpaywall(doi)
            if unpaywall_url:
                result = self._download_pdf(unpaywall_url, title)
                if result.status == "success":
                    return result

        # Try Brave search fallback
        if self.brave_api_key and title:
            brave_url = self._brave_pdf_search(title)
            if brave_url:
                result = self._download_pdf(brave_url, title)
                if result.status == "success":
                    result.metadata["source"] = "brave_fallback"
                    return result

        # All methods failed - mark as paywalled
        return FetchResult(
            status="paywalled",
            error="Could not find open access version",
            metadata={"doi": doi, "title": title}
        )

    def _fetch_youtube(self, item: Dict) -> FetchResult:
        """Fetch YouTube transcript using yt-dlp."""
        url = item.get("url", "")
        if not url:
            return FetchResult(status="failed", error="No URL provided")

        try:
            # Use yt-dlp to get transcript
            result = subprocess.run(
                [
                    "yt-dlp",
                    "--write-auto-sub",
                    "--sub-lang", "en",
                    "--skip-download",
                    "--output", "%(title)s",
                    url
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(self.staging_dir)
            )

            # Find the subtitle file
            for f in self.staging_dir.glob("*.vtt"):
                content = self._parse_vtt(f)
                f.unlink()  # Clean up
                return FetchResult(
                    status="success",
                    content=content,
                    file_size=len(content),
                    sha256=hashlib.sha256(content.encode()).hexdigest()
                )

            return FetchResult(
                status="failed",
                error="No transcript available"
            )

        except subprocess.TimeoutExpired:
            return FetchResult(status="failed", error="yt-dlp timed out")
        except FileNotFoundError:
            return FetchResult(status="failed", error="yt-dlp not installed")
        except Exception as e:
            return FetchResult(status="failed", error=str(e))

    def _download_pdf(self, url: str, title: str) -> FetchResult:
        """Download a PDF file."""
        try:
            import httpx

            response = httpx.get(url, follow_redirects=True, timeout=60)

            if response.status_code != 200:
                return FetchResult(
                    status="failed",
                    error=f"HTTP {response.status_code}"
                )

            # Check if it's actually a PDF
            content_type = response.headers.get("content-type", "")
            if "pdf" not in content_type.lower() and not response.content[:4] == b"%PDF":
                return FetchResult(
                    status="failed",
                    error="Response is not a PDF"
                )

            # Save to staging
            safe_title = re.sub(r'[^\w\s-]', '', title)[:50]
            filename = f"{safe_title}.pdf"
            filepath = self.staging_dir / filename

            filepath.write_bytes(response.content)

            sha256 = hashlib.sha256(response.content).hexdigest()

            return FetchResult(
                status="success",
                file_path=str(filepath),
                file_size=len(response.content),
                sha256=sha256
            )

        except Exception as e:
            return FetchResult(status="failed", error=str(e))

    def _is_pdf_url(self, url: str) -> bool:
        """Check if URL likely points to a PDF."""
        return url.endswith(".pdf") or "pdf" in url.lower()

    def _lookup_unpaywall(self, doi: str) -> Optional[str]:
        """Look up open access URL via Unpaywall."""
        try:
            import httpx

            email = "polymath@example.com"  # Required by Unpaywall
            url = f"https://api.unpaywall.org/v2/{doi}?email={email}"

            response = httpx.get(url, timeout=10)
            if response.status_code != 200:
                return None

            data = response.json()
            best_oa = data.get("best_oa_location", {})
            return best_oa.get("url_for_pdf") or best_oa.get("url")

        except Exception:
            return None

    def _brave_pdf_search(self, title: str) -> Optional[str]:
        """Search Brave for PDF link."""
        if not self.brave_api_key:
            return None

        try:
            import httpx

            query = f"{title} filetype:pdf"
            url = "https://api.search.brave.com/res/v1/web/search"

            response = httpx.get(
                url,
                params={"q": query, "count": 5},
                headers={"X-Subscription-Token": self.brave_api_key},
                timeout=10
            )

            if response.status_code != 200:
                return None

            data = response.json()
            results = data.get("web", {}).get("results", [])

            for result in results:
                url = result.get("url", "")
                if self._is_pdf_url(url):
                    return url

            return None

        except Exception:
            return None

    def _parse_vtt(self, vtt_path: Path) -> str:
        """Parse VTT subtitle file to plain text."""
        try:
            content = vtt_path.read_text()
            lines = []

            for line in content.splitlines():
                # Skip timestamps and WEBVTT header
                if "-->" in line or line.startswith("WEBVTT") or not line.strip():
                    continue
                # Skip position tags
                if line.strip().startswith("<"):
                    continue
                lines.append(line.strip())

            # Dedupe consecutive identical lines
            deduped = []
            for line in lines:
                if not deduped or line != deduped[-1]:
                    deduped.append(line)

            return " ".join(deduped)

        except Exception:
            return ""
