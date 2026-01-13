# Senior Engineer Optimization Plan for Polymath 2.0

**Author**: Max Van Belkum
**Institution**: Vanderbilt University MD-PhD Program
**Date**: January 2026

---

## Executive Summary

This document captures a comprehensive architectural review of the Polymath knowledge system. The current architecture is sophisticated but suffers from the **"Three-Body Problem"** (syncing Postgres + Neo4j + ChromaDB) and **"Garbage-In-Garbage-Out"** (poor OCR quality).

To achieve real-time, high-fidelity agentic workflows, we must:
1. Simplify the stack (consolidate to 2 databases)
2. Upgrade data quality (vision-based PDF parsing)
3. Standardize workflows (one-command operations)

---

## Phase 1: Immediate Housekeeping & Risk Reduction

### Problem: Split-Brain Environment

Current state reveals configuration drift risk:
- Two `.env` files: `/home/user/polymath-repo/.env` and `/home/user/work/polymax/.env`
- Two data roots with potential for inconsistency

### Actions

```bash
# 1. Symlink .env to prevent config drift
rm /home/user/work/polymax/.env
ln -s /home/user/polymath-repo/.env /home/user/work/polymax/.env

# 2. Archive and delete legacy ChromaDB (reclaims ~7.7 GB)
# DANGER: Verify this is truly legacy before running
tar -czf /scratch/polymath_archive/legacy_chromadb_backup.tar.gz /home/user/work/polymax/chromadb/
rm -rf /home/user/work/polymax/chromadb/
```

---

## Phase 2: Workflow Automation (One-Command Principle)

### Problem: Fragile 5-Step Manual Migration

Current KB V2 migration is manual and error-prone. If Step 3 fails, manual resume required.

### Solution: Orchestrated Makefile

```makefile
.PHONY: help v2-migration v2-validate clean-legacy

PYTHON := python3
SCRIPTS := scripts
LOG_FILE := migration_v2.log

help:
	@echo "Polymath Operations"
	@echo "  make v2-migration    - Run full V2 migration (Resumable)"
	@echo "  make v2-validate     - Run health checks"
	@echo "  make clean-legacy    - Remove legacy ChromaDB artifacts"

# Orchestrates the 24-50 hour process automatically
v2-migration:
	@echo "Starting V2 Migration (Log: $(LOG_FILE))..."
	@echo "[1/4] Backfilling Concepts..."
	@nohup $(PYTHON) $(SCRIPTS)/backfill_chunk_concepts_llm.py --target both --batch-size 16 --resume >> $(LOG_FILE) 2>&1
	@echo "[2/4] Rebuilding Papers Vector Store..."
	@$(PYTHON) $(SCRIPTS)/rebuild_chroma_bge_m3.py --target passages --collection polymath_bge_m3 --resume >> $(LOG_FILE) 2>&1
	@echo "[3/4] Rebuilding Code Vector Store..."
	@$(PYTHON) $(SCRIPTS)/rebuild_chroma_bge_m3.py --target chunks --collection polymath_code_bge_m3 --resume >> $(LOG_FILE) 2>&1
	@echo "[4/4] Hydrating Knowledge Graph..."
	@$(PYTHON) $(SCRIPTS)/rebuild_neo4j_concepts_v2.py --target both --resume >> $(LOG_FILE) 2>&1
	@echo "Migration Complete. Run 'make v2-validate' to verify."

v2-validate:
	$(PYTHON) $(SCRIPTS)/validate_kb_v2.py

clean-legacy:
	@echo "Deleting legacy ChromaDB..."
	rm -rf /home/user/work/polymax/chromadb
```

---

## Phase 3: Ingestion Bottleneck Optimization

### Problem: 24-50 Hour Concept Extraction Runtime

Current state: Sequential processing with small batch size (`--batch-size 16`) against local LLM.

### Optimization Strategies

| Strategy | Implementation | Expected Speedup |
|----------|----------------|------------------|
| **Hybridize** | Use Gemini batch for bulk, local LLM for sensitive data | 10x |
| **Increase batch size** | RTX 5090 24GB can handle batch_size=64-128 with qwen2.5:3b | 4x |
| **Parallel workers** | Spin up 10-20 API workers for Gemini/OpenAI | 10-20x |

```bash
# Profile GPU utilization first
nvidia-smi -l 1

# If GPU util < 90%, increase batch size
python3 scripts/backfill_chunk_concepts_llm.py --batch-size 64 --resume
```

---

## Phase 4: The "Clean Text" Revolution (PDF Parsing)

### Problem: OCR Garbage

`pdfplumber` and basic OCR cannot handle complex scientific paper layouts:
- Two-column formats
- Tables and figures
- Mathematical formulas

**Example failure**: `e^{i\pi} + 1 = 0` becomes `e i n + 1 = 0`

### Solution: Vision-Based Markdown Conversion

#### Option A: MinerU (Local SOTA - FREE)

```bash
# Already have scripts/ingest_mineru.py
# Benchmark on 10 math-heavy papers first
python3 scripts/ingest_mineru.py /path/to/math_heavy_paper.pdf

# Outputs clean Markdown with LaTeX preserved
```

**Pros**: Free, local, excellent for scientific papers
**Cons**: Requires GPU VRAM

#### Option B: Mathpix (Premium - ~$0.005/page)

```python
# lib/mathpix_ingest.py
import requests

def convert_to_markdown(pdf_path):
    """Perfect LaTeX extraction for math, tables, chemistry."""
    response = requests.post(
        "https://api.mathpix.com/v3/pdf",
        files={"file": open(pdf_path, "rb")},
        headers={"app_id": APP_ID, "app_key": APP_KEY}
    )
    return response.json()['text']  # Returns perfect Markdown
```

**Pros**: Perfect LaTeX extraction
**Cons**: Paid API (~$20 for 4,000 pages)

### Decision Tree

```
Try MinerU on 10 math-heavy papers
    |
    ├── LaTeX clean? → Scale up MinerU (FREE)
    |
    └── LaTeX garbage? → Switch to Mathpix ($20/4K pages)
```

---

## Phase 5: Collapsing the Stack (Three-Body Problem)

### Problem: Three Databases Out of Sync

| Database | Role | Sync Mechanism |
|----------|------|----------------|
| Postgres | Source of Truth | - |
| ChromaDB | Vector Search | `consistency_check.py` |
| Neo4j | Graph | `consistency_check.py` |

**Risk**: Scripts are the only defense against drift.

### Solution: Consolidate Vectors into Postgres with `pgvector`

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Add embedding column to passages
ALTER TABLE passages ADD COLUMN embedding vector(1024);

-- Create HNSW index for fast similarity search
CREATE INDEX ON passages USING hnsw (embedding vector_cosine_ops);
```

**Benefits**:
1. **Atomic commits**: Delete paper → vector gone instantly
2. **Hybrid search in single query**:
```sql
SELECT * FROM passages
WHERE embedding <=> $query_vector
  AND year > 2023
  AND author ILIKE '%Mahmood%'
ORDER BY embedding <=> $query_vector
LIMIT 10;
```
3. **Eliminate ChromaDB**: Removes 16GB+ disk, entire sync logic

### New Architecture

```
BEFORE: Postgres + ChromaDB + Neo4j (3 databases, sync scripts)
AFTER:  Postgres (with pgvector) + Neo4j (2 databases, no sync needed)
```

---

## Phase 6: Hierarchical "Small-to-Big" Indexing

### Problem: Context Window Tradeoff

- Full chunks → overload context
- Small chunks → lose surrounding context

### Solution: Parent-Child Indexing

```
PARENT (Retrieval Unit): Full paragraph/section
    |
    └── CHILD (Search Unit): 2-3 sentence propositions
```

**Workflow**:
1. Embed **children** (dense, precise)
2. When child matches, return **parent** (full context)

### Code Indexing (AST-Based)

```python
# Don't chunk by lines - use tree-sitter
import tree_sitter_python as tspython
from tree_sitter import Parser

parser = Parser(tspython.language())

def extract_functions(code: str):
    """Extract complete functions/classes, not arbitrary line chunks."""
    tree = parser.parse(bytes(code, "utf8"))
    # Walk tree, extract function_definition nodes
    ...
```

**Enrichment**: Generate "Summary Docstring" for each function with cheap LLM (Flash/Haiku). Embed the *summary*, retrieve the *code*.

---

## Phase 7: Zotero Integration (Golden Metadata)

### Problem: Manual Metadata Management

Trying to maintain paper metadata in Postgres manually or via brittle OpenAlex scripts.

### Solution: Zotero as Source of Truth

**Workflow**:
```
1. User adds paper to Zotero (Browser Connector)
   └── Zotero grabs perfect metadata + PDF

2. Polymath Sync (scripts/sync_zotero_v2.py)
   └── Pulls metadata from Zotero
   └── Grabs PDF path from Zotero storage

3. Claude Integration (Zotero MCP Server)
   └── Direct access to library context
   └── "Search for papers about 'diffusion models' in my library"
```

**Benefits**:
- Zotero handles messy part: metadata scraping, PDF renaming, deduplication
- Single source of truth for library
- Natural user interface for paper management

---

## Phase 8: New MCP Toolset Definition

### Goal: Precise Tools for Claude

| Tool Name | Input | Function |
|-----------|-------|----------|
| `search_library` | `query`, `year_min` | Returns **Summaries + Citations** (not full text) |
| `read_section` | `doc_id`, `header_name` | Returns full text of specific section only |
| `get_paper_full_text` | `paper_id` | Returns structured Markdown with sections |
| `get_math_context` | `paper_id` | Returns LaTeX definitions so Claude understands variables |
| `graph_traversal` | `concept`, `hops` | Finds unexpected connections via Neo4j |
| `code_archaeologist` | `query` | Searches code *summaries*, returns file path + function |

---

## Phase 9: Code Structure Refactor

### Problem: Scripts Directory is a "Junkyard"

50+ scripts with overlapping functionality.

### Actions

1. **Consolidate repair scripts** into single `maintenance.py` CLI:
   - `fix_numeric_titles.py`
   - `fix_kg_gaps.py`
   - `fix_citations_postgres.sql`

2. **Standardize config imports**:
   - Every script imports `lib.config`
   - No direct `.env` reading in scripts
   - API key validation in `lib/config.py` or Pydantic settings model

---

## Execution Roadmap

### Day 1: Infrastructure Cleanup

```bash
# 1. Enable pgvector
psql -U polymath -d polymath -c "CREATE EXTENSION IF NOT EXISTS vector;"

# 2. Add embedding column
psql -U polymath -d polymath -c "ALTER TABLE passages ADD COLUMN IF NOT EXISTS embedding vector(1024);"

# 3. Migrate from ChromaDB to Postgres
python3 scripts/migrate_chroma_to_pgvector.py

# Deliverable: One database handling Metadata + Vectors
```

### Day 2: Ingestion Pipeline V2

```bash
# 1. Install MinerU
pip install magic-pdf

# 2. Create ingest_v2.py that outputs Markdown
python3 scripts/ingest_v2.py /path/to/paper.pdf --output-markdown

# Deliverable: High-quality, clean text for next 100 papers
```

### Day 3: The "Brain" (Claude Context)

```bash
# 1. Update search tool to use Postgres Hybrid Search
# 2. Implement "Small-to-Big" retrieval logic

# Deliverable: Claude can search precisely without context overload
```

---

## Summary Checklist

| Priority | Action | Status |
|----------|--------|--------|
| 1 | Stop writing parsers - use MinerU/Mathpix | Pending |
| 2 | Activate Zotero as input UI | Pending |
| 3 | Benchmark MinerU on math-heavy papers | Pending |
| 4 | Enable pgvector in Postgres | Pending |
| 5 | Migrate ChromaDB vectors to Postgres | Pending |
| 6 | Retire ChromaDB entirely | Pending |
| 7 | Implement parent-child indexing | Pending |
| 8 | Create new MCP toolset | Pending |
| 9 | Consolidate scripts into maintenance.py | Pending |
| 10 | Symlink .env files | Pending |

---

## Expected Outcomes

| Metric | Before | After |
|--------|--------|-------|
| **Databases** | 3 (Postgres + ChromaDB + Neo4j) | 2 (Postgres + Neo4j) |
| **Sync scripts** | 5+ | 0 |
| **OCR quality** | ~60% clean | ~95% clean |
| **Ingestion time** | 24-50 hours | 4-8 hours |
| **Disk usage** | ~25GB | ~17GB |
| **Metadata accuracy** | 19% DOI | 95%+ DOI |

This transforms Polymath from a "text dump" into a structured **Research Assistant** that understands the difference between an Abstract and a Proof.
