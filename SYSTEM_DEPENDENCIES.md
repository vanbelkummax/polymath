# Polymath System Dependencies - Complete Map

## Quick Summary
| Category | Location | Size |
|----------|----------|------|
| Main Code | `/home/user/polymath-repo/` | 8.8 GB |
| ChromaDB (active) | `/home/user/polymath-repo/chromadb/` | 8.7 GB |
| ChromaDB (legacy) | `/home/user/work/polymax/chromadb/` | 7.7 GB |
| GitHub Repos | `/home/user/work/polymax/data/github_repos/` | 26 GB |
| HuggingFace Models | `~/.cache/huggingface/hub/` | 8.5 GB |
| PostgreSQL | `/var/lib/postgresql/` | ~2 GB |
| Neo4j | `/var/lib/neo4j/` | ~500 MB |
| **TOTAL** | | **~55 GB** |

---

## 1. ENVIRONMENT VARIABLES (Required)

**File**: `/home/user/polymath-repo/.env` (source of truth; `/home/user/work/polymax/.env` mirrors it)

```bash
# Database
POLYMATH_ROOT=/home/user/polymath-repo
NEO4J_PASSWORD=polymathic2026
POSTGRES_DSN=dbname=polymath user=polymath host=/var/run/postgresql

# API Keys
BRAVE_API_KEY=<your_brave_api_key>
OPENALEX_EMAIL=<your_email_for_polite_pool>
OPENALEX_API_KEY=<optional_key>
HUGGINGFACE_TOKEN=<optional>

# Paths (optional - have defaults)
CHROMADB_PATH=/home/user/polymath-repo/chromadb
# Legacy scripts still read CHROMA_PATH (alias to BGE-M3 store)
CHROMA_PATH=/home/user/polymath-repo/chromadb
# Legacy store (archived): /home/user/work/polymax/chromadb
```

**Load with**: `set -a && source /home/user/polymath-repo/.env && set +a`

Codex MCP config (separate): `/home/user/.codex/config.toml`

---

## 2. DATABASES

### PostgreSQL (Source of Truth)
```
Connection: psql -U polymath -d polymath
Location: /var/lib/postgresql/
Tables:
  - documents: 29,485 rows
  - passages: 545,210 rows
  - code_files: 65,441 rows
  - code_chunks: 416,397 rows
  - concepts: 200 rows
```

### ChromaDB (Vector Store)
```
Location: /home/user/polymath-repo/chromadb/
Collections:
  - polymath_bge_m3: 545,099 paper passages (1024-dim)
  - polymath_code_bge_m3: 415,468 code chunks (1024-dim)
Embedding Model: BAAI/bge-m3
```

### Neo4j (Knowledge Graph)
```
Connection: bolt://localhost:7687
Auth: neo4j / polymathic2026
Location: /var/lib/neo4j/
Nodes:
  - Paper: 31,996
  - Concept: 222
  - Code: 13,651
Edges: 27,395
```

---

## 3. CODE REPOSITORY

**Location**: `/home/user/polymath-repo/`

### Core Libraries (`lib/`)
```
lib/
├── config.py              # Centralized configuration
├── db.py                  # Database wrapper
├── unified_ingest.py      # Main ingestion pipeline
├── hybrid_search_v2.py    # Search across all stores
├── evidence_extractor.py  # NLI-based evidence finding
├── enhanced_pdf_parser.py # PDF parsing with page coords
├── doc_identity.py        # Deterministic UUIDv5
├── code_search.py         # Code-specific search
├── citation_builder.py    # Citation formatting
└── sentry/                # Literature discovery
    ├── sources/
    │   ├── openalex.py
    │   ├── arxiv.py
    │   ├── biorxiv.py
    │   ├── europepmc.py
    │   └── github.py
    └── sentry.py          # Orchestrator
```

### Scripts (`scripts/`)
```
scripts/
├── migrate_knowledge_base.py  # BGE-M3 migration (Postgres → Chroma/Neo4j)
├── hackathon_audit.py         # Coverage audit
├── hydrate_neo4j.py           # Sync Neo4j from Postgres
├── ingest_github_batch.py     # Batch repo ingestion
├── fix_numeric_titles.py      # Data cleanup
└── rebuild_chromadb_v2.py     # ChromaDB rebuild
```

### MCP Servers (`mcp/`) - Claude-specific
```
mcp/
├── polymath_mcp.py           # Main MCP server
├── polymath_v11/
│   ├── server.py             # v11 server
│   └── tools/
│       ├── discovery.py      # deep_hunt, find_gaps
│       └── reasoning.py      # hypothesis generation
└── neo4j_server.py           # Graph queries
```

### Entry Points
```
polymath_cli.py    # CLI tool (argparse)
codex_tools.py     # Codex-compatible wrapper
AGENTS.md          # Codex instructions
/home/user/CLAUDE.md  # Claude instructions (outside repo)
```

---

## 4. INDEXED GITHUB REPOS

**Location**: `/home/user/work/polymax/data/github_repos/`
**Count**: 176 repos
**Size**: 26 GB

### Key Repos by Category
```
# Spatial Biology
spatialdata, spatialdata-io, squidpy, scanpy
STalign, BayesPrism, SpatialDE, starfish
bin2cell, paste, paste2, cell2location, Tangram

# Foundation Models
UNI, CONCH, CLAM, HIPT (mahmoodlab)

# Graph/OT Methods
POT, ott, geomloss, fugw, pygmtools
pytorch_geometric, networkx, graspologic

# Single-cell
scvi-tools, scanpy, seurat, scarches, celltypist

# General ML
transformers, dspy, pytorch_geometric
```

---

## 5. ML MODELS (HuggingFace Cache)

**Location**: `~/.cache/huggingface/hub/`
**Size**: 8.5 GB

```
models--BAAI--bge-m3                           # Main embeddings (1024-dim)
models--cross-encoder--ms-marco-MiniLM-L-6-v2  # Reranking
models--cross-encoder--nli-deberta-v3-base     # Evidence NLI
models--sentence-transformers--all-mpnet-base-v2  # Fallback
```

---

## 6. DOCUMENTATION

```
docs/
├── ACADEMIC_PLAYBOOK.md       # Academic workflows
├── HACKATHON_PLAYBOOK.md      # Hackathon workflows
├── POLYMATH_USER_MANUAL.md    # Full manual
├── hackathon_reports/
│   └── HACKATHON_AUDIT.md     # Coverage audit results
└── plans/                     # Design docs
```

---

## 7. DATA DIRECTORIES

```
/home/user/work/polymax/
├── ingest_staging/           # PDFs awaiting ingestion (2.8 GB)
├── data/github_repos/        # Indexed repos (26 GB)
├── logs/                     # Ingestion logs
├── reports/                  # Generated reports
├── chromadb/                 # Backup ChromaDB (7.7 GB)
└── .env                      # Environment variables
```

---

## 8. SYSTEM SERVICES

```bash
# PostgreSQL
sudo systemctl status postgresql

# Neo4j
sudo systemctl status neo4j

# Check all
psql -U polymath -d polymath -c "SELECT 1"
python -c "from neo4j import GraphDatabase; d=GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j','polymathic2026')); d.verify_connectivity(); print('Neo4j OK')"
```

---

## 9. PYTHON DEPENDENCIES

**File**: `/home/user/polymath-repo/requirements.txt`

Key packages:
```
chromadb>=0.4.0
sentence-transformers>=2.2.0
FlagEmbedding>=1.2.0
neo4j>=5.0.0
psycopg2-binary>=2.9.0
transformers>=4.30.0
torch>=2.0.0
httpx>=0.24.0
pdfplumber>=0.9.0
spacy>=3.5.0
typer>=0.9.0
```

---

## 10. BACKUP LOCATIONS

```
/scratch/polymath_backup_2026_01_06/
├── postgres_polymath_dump.sql.gz
├── chromadb_backup/
└── neo4j_backup/
```

---

## Quick Setup for New Machine

```bash
# 1. Clone repo
git clone https://github.com/vanbelkummax/polymath /home/user/polymath-repo

# 2. Copy data directories
rsync -av /source/work/polymax/ /home/user/work/polymax/

# 3. Install Python deps
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 4. Set environment
cp /home/user/work/polymax/.env.example /home/user/work/polymax/.env
# Edit .env with your API keys

# 5. Start services
sudo systemctl start postgresql neo4j

# 6. Verify
python codex_tools.py stats
```
