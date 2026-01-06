# Literature Sentry Design

**Date:** 2026-01-04
**Status:** Approved for Implementation

## Overview

The Literature Sentry is an autonomous, quality-focused curator that discovers and ingests the best polymathic resources across domains. It operates on-demand via CLI invocation.

## Goals

1. **Quality over quantity** - Curate excellence, not bulk
2. **Cross-domain discovery** - Bridge concepts from different fields
3. **Autonomous operation** - Make intelligent decisions without asking
4. **Graceful paywall handling** - Flag inaccessible items for manual fetch

## Architecture

```
/literature-sentry "query" [--source arxiv,pubmed,github] [--max 20]
           │
           ▼
┌──────────────────────────────────────────────────────┐
│              SKILL (literature_sentry.md)            │
│  - Parses user query                                 │
│  - Calls lib/sentry/ modules                         │
│  - Presents summary, flags paywalled items           │
└──────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────┐
│              LIBRARY (lib/sentry/)                   │
│  ├── sources/                                        │
│  │   ├── europepmc.py   # Europe PMC (preferred)     │
│  │   ├── arxiv.py       # arXiv API                  │
│  │   ├── biorxiv.py     # bioRxiv API                │
│  │   ├── github.py      # GitHub Search API          │
│  │   ├── youtube.py     # yt-dlp for transcripts     │
│  │   └── brave.py       # Fallback PDF search        │
│  ├── scoring.py         # Quality gate logic         │
│  ├── fetcher.py         # Download + validate        │
│  └── sentry.py          # Orchestrator               │
└──────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────┐
│              STORAGE                                 │
│  - source_items table (Postgres) - discovery ledger  │
│  - needs_user_tickets table - paywalled queue        │
│  - unified_ingest.py - ChromaDB + Neo4j + Postgres   │
└──────────────────────────────────────────────────────┘
```

## Source Coverage (Full Polymathic Sweep)

| Source | Purpose | Quality Gate |
|--------|---------|--------------|
| Europe PMC | Academic papers | Citations, concept bridges, author trust |
| arXiv | Preprints (speed) | Freshness, domain relevance |
| bioRxiv | Bio preprints | Freshness, mesh term overlap |
| GitHub | Code implementations | Stars (log-scaled), velocity, language |
| YouTube | Academic lectures | Whitelisted channels only |
| Brave | Fallback PDF search | Last resort before flagging |

## Quality Scoring System

### Log-Normalized Cross-Domain Scoring

```python
import math

SOURCE_THRESHOLDS = {
    "pubmed": {"citations": (0, 500), "halflife_days": 730},
    "arxiv": {"citations": (0, 50), "halflife_days": 180},
    "github": {"stars": (0, 1000), "halflife_days": 365},
    "youtube": {"views": (500, 50000), "halflife_days": 365}
}

def normalize_log_scale(value: float, source: str, metric: str) -> float:
    """Logarithmic normalization for power-law distributions."""
    low, high = SOURCE_THRESHOLDS[source][metric]
    val_safe = max(value, 1)
    low_safe = max(low, 1)

    log_val = math.log10(val_safe)
    log_low = math.log10(low_safe)
    log_high = math.log10(high)

    norm = (log_val - log_low) / (log_high - log_low)
    return min(1.0, max(0.0, norm))

def freshness_decay(pub_date, source: str) -> float:
    """Exponential decay based on source velocity."""
    days_old = (date.today() - pub_date).days
    halflife = SOURCE_THRESHOLDS[source]["halflife_days"]
    return 2 ** (-days_old / halflife)
```

### Priority Score Weights

| Signal | Weight | Description |
|--------|--------|-------------|
| popularity | 0.20 | Citations/stars/views (log-normalized) |
| freshness | 0.15 | Exponential decay based on source halflife |
| bridge_score | 0.30 | Spans multiple concept domains |
| gap_fill | 0.20 | Mentions underrepresented concepts |
| author_trust | 0.15 | On whitelist or high h-index |

### Hidden Gem Detection

```python
def is_hidden_gem(item: dict, source: str) -> bool:
    """Low popularity but high semantic relevance."""
    pop_score = normalize_log_scale(item.get("citations", 0), source, "citations")
    relevance = item.get("abstract_relevance_score", 0)
    return pop_score < 0.2 and relevance > 0.7
```

## Postgres Schema

### Source-Specific JSONB with Generated Columns

```sql
-- Promote frequently-queried fields to stored columns
ALTER TABLE source_items ADD COLUMN IF NOT EXISTS
    priority_score FLOAT GENERATED ALWAYS AS (
        (meta_json->>'priority_score')::float
    ) STORED;

ALTER TABLE source_items ADD COLUMN IF NOT EXISTS
    concept_domains TEXT[] GENERATED ALWAYS AS (
        ARRAY(SELECT jsonb_array_elements_text(meta_json->'concept_domains'))
    ) STORED;

-- Composite uniqueness for ledger integrity
ALTER TABLE source_items
ADD CONSTRAINT uq_source_external_id UNIQUE (source, external_id);

-- Indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_source_items_priority ON source_items(priority_score DESC);
CREATE INDEX IF NOT EXISTS idx_source_items_concepts ON source_items USING GIN(concept_domains);
```

### JSONB Structure Per Source

**PubMed/Europe PMC:**
```json
{
    "pmid": "12345678",
    "doi": "10.1234/example",
    "citations": 45,
    "journal": "Nature Methods",
    "mesh_terms": ["Spatial Transcriptomics", "Machine Learning"],
    "abstract_embedding_sim": 0.82,
    "priority_score": 0.73,
    "concept_domains": ["spatial_transcriptomics", "neural_network"],
    "is_hidden_gem": false,
    "oa_url": "https://europepmc.org/...",
    "brave_fallback_url": null
}
```

**GitHub:**
```json
{
    "stars": 234,
    "forks": 45,
    "last_commit": "2025-12-15",
    "language": "Python",
    "topics": ["spatial-transcriptomics", "deep-learning"],
    "velocity_score": 0.9,
    "priority_score": 0.68,
    "concept_domains": ["transformer", "spatial_transcriptomics"]
}
```

## GitHub Repo Flattening

### Strategy

1. **README always included** - It's the "abstract" of the repo
2. **Structure tree** - Shows architecture without reading every file
3. **Priority-based file selection** - `model.py` before `test_utils.py`
4. **Token budget** - Hard cap (50k tokens) prevents context explosion
5. **Notebook parsing** - Extract code/markdown from .ipynb, discard metadata
6. **Skeleton fallback** - Function signatures for truncated files

### Priority Files

```python
PRIORITY_FILES = [
    "README.md", "README.rst",
    "setup.py", "pyproject.toml",
    "**/model*.py", "**/train*.py", "**/main*.py",
    "**/*config*.py", "**/*utils*.py",
    "**/*.ipynb",  # Notebooks are critical in AI research
]
```

### Notebook Parser

```python
def _read_notebook(self, fpath) -> str:
    """Converts .ipynb JSON to clean script format."""
    import json
    with open(fpath) as f:
        nb = json.load(f)
    output = []
    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            output.append(f"# %% [CODE]\n{source}")
        elif cell['cell_type'] == 'markdown':
            source = ''.join(cell['source'])
            output.append(f"# %% [MARKDOWN]\n{source}")
    return "\n\n".join(output)
```

## Key Design Decisions

1. **Europe PMC over PubMed** - Better at finding open access links
2. **Brave fallback** - Search for PDF before flagging as paywalled
3. **Log-scale normalization** - Handles power-law distributions fairly
4. **Snake_case concept normalization** - Prevents "tag soup" across sources
5. **Score before fetch** - Quality gate on cheap metadata, not expensive content
6. **Velocity for GitHub** - Recent commits matter more than creation date

## Implementation Order

1. CLI entry point (`lib/sentry/cli.py`)
2. Scoring module (`lib/sentry/scoring.py`)
3. Source connectors (`lib/sentry/sources/*.py`)
4. Fetcher with flattening (`lib/sentry/fetcher.py`)
5. Orchestrator (`lib/sentry/sentry.py`)
6. Skill markdown (`skills/literature_sentry.md`)

## Usage

```bash
# Search and ingest spatial transcriptomics papers
python3 -m lib.sentry.cli "spatial transcriptomics visium" --max 15

# GitHub only, minimum 100 stars
python3 -m lib.sentry.cli "image to gene expression" --source github --min-stars 100

# Show paywalled queue
python3 -m lib.sentry.cli --show-tickets

# Dry run (score but don't ingest)
python3 -m lib.sentry.cli "colibactin pks" --dry-run
```
