"""Centralized configuration for Polymath.

All hardcoded paths and configuration values should be imported from here.
Supports environment variable overrides for containerization.
"""
import os
from pathlib import Path
from typing import Optional


def _find_project_root() -> Path:
    """Find project root by looking for pyproject.toml or .git."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return current.parent  # Fallback to lib's parent


# Project root - auto-detected or from environment
PROJECT_ROOT = Path(os.environ.get("POLYMATH_ROOT", str(_find_project_root())))

# =============================================================================
# Database Paths
# =============================================================================

CHROMADB_PATH = Path(os.environ.get("CHROMADB_PATH", str(PROJECT_ROOT / "chromadb")))
STAGING_DIR = Path(os.environ.get("STAGING_DIR", str(PROJECT_ROOT / "ingest_staging")))

# =============================================================================
# Database Connections
# =============================================================================

POSTGRES_DSN = os.environ.get(
    "POSTGRES_DSN",
    "dbname=polymath user=polymath host=/var/run/postgresql"
)

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")

# =============================================================================
# Embedding Models
# =============================================================================

# Default to BGE-M3 for SOTA performance (1024-dim)
# Set EMBEDDING_MODEL=all-mpnet-base-v2 for legacy 768-dim embeddings
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
EMBEDDING_DIM = 1024  # BGE-M3 dimension

# Collection names - using isolated collection for BGE-M3 (prevents dimension mismatch)
CHROMA_COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION_NAME", "polymath_bge_m3")
PAPERS_COLLECTION = os.environ.get("PAPERS_COLLECTION", "polymath_bge_m3")
CODE_COLLECTION = os.environ.get("CODE_COLLECTION", "polymath_code_bge_m3")

# Legacy collection names (768-dim, for fallback)
PAPERS_COLLECTION_LEGACY = "polymath_papers"
CODE_COLLECTION_LEGACY = "polymath_code"

# =============================================================================
# Local LLM Configuration (Ollama)
# =============================================================================

LOCAL_LLM_FAST = os.environ.get("LOCAL_LLM_FAST", "qwen3:4b")
LOCAL_LLM_HEAVY = os.environ.get("LOCAL_LLM_HEAVY", "deepseek-r1:8b")

# =============================================================================
# LLM Configuration
# =============================================================================

# Primary mode: Claude subagents handle reasoning in Claude Code sessions
# API mode: Direct Anthropic calls for standalone/batch execution
ANTHROPIC_API_KEY: Optional[str] = os.environ.get("ANTHROPIC_API_KEY")
LLM_MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-4-20250514")
ENABLE_LLM_API = os.environ.get("ENABLE_LLM_API", "false").lower() == "true"

# =============================================================================
# External API Keys
# =============================================================================

BRAVE_API_KEY: Optional[str] = os.environ.get("BRAVE_API_KEY")
GITHUB_TOKEN: Optional[str] = os.environ.get("GITHUB_TOKEN")
OPENALEX_EMAIL = os.environ.get("OPENALEX_EMAIL", "polymath@example.com")

# =============================================================================
# Rate Limits (requests per second)
# =============================================================================

RATE_LIMITS = {
    "europepmc": 10.0,
    "arxiv": 0.33,  # 1 per 3 seconds
    "biorxiv": 1.0,
    "github": 0.5,
    "openalex": 10.0,  # Polite pool with email
}

# =============================================================================
# Utility Functions
# =============================================================================

def ensure_dirs():
    """Create required directories if they don't exist."""
    CHROMADB_PATH.mkdir(parents=True, exist_ok=True)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)


def get_chroma_path() -> str:
    """Get ChromaDB path as string (for ChromaDB client)."""
    return str(CHROMADB_PATH)
