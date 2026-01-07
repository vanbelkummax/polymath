# Polymath System - Codex Instructions

## Overview
Polymath is a polymathic research platform with ~545K paper passages and ~415K code chunks indexed.

## Quick Commands

### Search
```bash
# Semantic search
python polymath_cli.py search "spatial transcriptomics" -n 10

# With expansion (includes graph concepts)
python polymath_cli.py search "spatial transcriptomics" -n 10 --expand
```

### Python API (preferred for complex queries)
```python
import sys
sys.path.insert(0, '/home/user/polymath-repo')

from lib.hybrid_search_v2 import HybridSearcherV2
hs = HybridSearcherV2()

# Paper search with filters
results = hs.search_papers("query", n=10, year_min=2023)

# Code search with repo filter
results = hs.search_code("attention mechanism", n=10, org_filter="mahmoodlab")

# Hybrid search (papers + code + graph)
results = hs.hybrid_search("spatial transcriptomics", n=20)
```

### Graph Queries
```bash
# Neo4j Cypher queries
python polymath_cli.py graph "MATCH (p:Paper)-[:MENTIONS]->(c:CONCEPT) RETURN c.name, count(p) ORDER BY count(p) DESC LIMIT 10"
```

### Ingestion
```python
from lib.unified_ingest import TransactionalIngestor
ingestor = TransactionalIngestor()

# Ingest PDF
result = ingestor.ingest_pdf("/path/to/paper.pdf")

# Ingest code
result = ingestor.ingest_code("/path/to/code.py", repo_name="my-repo")
```

### Evidence-Bound Verification
```python
from lib.evidence_extractor import EvidenceExtractor
extractor = EvidenceExtractor()

# Find evidence for a claim
spans = extractor.extract_spans_for_claim("Transformers improve spatial prediction")
```

## Environment Setup
```bash
export NEO4J_PASSWORD=polymathic2026
export PYTHONPATH=/home/user/polymath-repo:$PYTHONPATH
cd /home/user/polymath-repo
```

## Codex MCP (polymath-v11)
- MCP config: `/home/user/.codex/config.toml` (restart Codex after edits)
- Server: `/home/user/polymath-repo/mcp/polymath_v11/server.py`
- Backing stores: Postgres (FTS), ChromaDB (BGE-M3), Neo4j (graph)
- Only `polymath-v11` is enabled by default to avoid startup timeouts
- Secrets are stored in config; do not paste into docs or chat

## Database Locations
- PostgreSQL: `psql -U polymath -d polymath`
- ChromaDB: `/home/user/polymath-repo/chromadb/`
- Neo4j: `bolt://localhost:7687`

## Key Directories
- Papers staging: `/home/user/work/polymax/ingest_staging/`
- GitHub repos: `/home/user/work/polymax/data/github_repos/`
- Logs: `/home/user/work/polymax/logs/`
- Full system map: `/home/user/polymath-repo/SYSTEM_DEPENDENCIES.md`

## Playbooks
- Academic workflows: `docs/ACADEMIC_PLAYBOOK.md`
- Hackathon workflows: `docs/HACKATHON_PLAYBOOK.md`

## Coverage (as of 2026-01-07)
- 29,485 documents
- 545,210 passages
- 416,397 code chunks
- Strong: spatial, visium, xenium, transformer, segmentation
- See: `docs/hackathon_reports/HACKATHON_AUDIT.md`
