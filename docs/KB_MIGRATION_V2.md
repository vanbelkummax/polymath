# Knowledge Base V2 Migration Guide

## Overview

This migration upgrades the Polymath knowledge base with:

1. **Typed Concept Extraction** - LLM-based extraction with confidence scores and concept types
2. **BGE-M3 Embeddings** - SOTA 1024-dim embeddings in new ChromaDB collections
3. **Neo4j Concept Graph V2** - Versioned Chunk→Concept edges with provenance

## Prerequisites

### Environment Variables

```bash
export NEO4J_PASSWORD=polymathic2026
export LOCAL_LLM_FAST=qwen2.5:3b      # Optional: override default
export LOCAL_LLM_HEAVY=deepseek-r1:8b  # Optional: override default
```

### Python Dependencies

Ensure these are installed:
```bash
pip install ollama FlagEmbedding psycopg2-binary neo4j chromadb tqdm
```

### Ollama Models

Pull required models:
```bash
ollama pull qwen2.5:3b
ollama pull deepseek-r1:8b  # Optional but recommended
```

## Migration Steps

### Option 1: Automated (Recommended)

Run the orchestrator to execute all steps:

```bash
python scripts/migrate_kb_v2.py

# With options:
python scripts/migrate_kb_v2.py --limit 1000 --dry-run  # Test with 1000 items
python scripts/migrate_kb_v2.py --only backfill_concepts  # Run specific step
python scripts/migrate_kb_v2.py --skip rebuild_neo4j  # Skip a step
```

### Option 2: Manual (Step-by-Step)

#### Step 1: Backfill Concepts

Extract concepts from existing chunks/passages:

```bash
# Passages (papers)
python scripts/backfill_chunk_concepts_llm.py \
  --target passages \
  --batch-size 16 \
  --resume

# Code chunks
python scripts/backfill_chunk_concepts_llm.py \
  --target chunks \
  --batch-size 16 \
  --resume

# Dry run first (recommended)
python scripts/backfill_chunk_concepts_llm.py \
  --target both \
  --limit 100 \
  --dry-run
```

**Expected time**: ~2-5 sec per chunk (LLM processing)

#### Step 2: Rebuild ChromaDB

Create new BGE-M3 collections:

```bash
# Papers
python scripts/rebuild_chroma_bge_m3.py \
  --target passages \
  --collection polymath_bge_m3 \
  --batch-size 128 \
  --resume

# Code
python scripts/rebuild_chroma_bge_m3.py \
  --target chunks \
  --collection polymath_code_bge_m3 \
  --batch-size 128 \
  --resume
```

**Expected time**: ~147 embeddings/sec on RTX 5090

#### Step 3: Rebuild Neo4j

Create concept graph v2:

```bash
python scripts/rebuild_neo4j_concepts_v2.py \
  --extractor-version llm_v2 \
  --edge-version llm_v2 \
  --target both \
  --resume

# Wipe and start fresh (optional)
python scripts/rebuild_neo4j_concepts_v2.py \
  --wipe-neo4j \
  --target both
```

**Expected time**: ~500-1000 nodes/sec

## Validation

After migration, validate the system:

```bash
python scripts/validate_kb_v2.py
```

Expected output:
```
============================================================
KB V2 VALIDATION
============================================================

✓ Postgres:
  Chunks: 411,729
  Passages: 532,259
  Chunk concepts: 2,058,645
  Passage concepts: 2,661,295

✓ ChromaDB:
  Papers collection (polymath_bge_m3): 532,259
  Code collection (polymath_code_bge_m3): 411,729

✓ Neo4j:
  Concepts: 12,456
  MENTIONS edges: 4,719,940
  Passage nodes: 532,259

============================================================
✓ VALIDATION PASSED
============================================================
```

## Verification Queries

### Postgres: Check Concept Density

```sql
-- Average concepts per passage
SELECT AVG(cnt) FROM (
  SELECT passage_id, COUNT(*) as cnt
  FROM passage_concepts
  WHERE extractor_version = 'llm_v2'
  GROUP BY passage_id
) t;

-- Top concepts by frequency
SELECT concept_name, COUNT(*) as freq
FROM passage_concepts
WHERE extractor_version = 'llm_v2'
GROUP BY concept_name
ORDER BY freq DESC
LIMIT 20;
```

### ChromaDB: Test Hybrid Search

```python
from lib.hybrid_search_v2 import HybridSearcherV2

hs = HybridSearcherV2()
results = hs.search_papers("spatial transcriptomics variational inference", n=5)

for r in results:
    print(f"{r.title[:80]} (score={r.score:.3f})")
```

### Neo4j: Test Bridge Queries

```cypher
// Find concepts bridging spatial biology and math
MATCH (p:Passage)-[m1:MENTIONS]->(c:Concept)
WHERE c.name CONTAINS 'spatial' OR c.name CONTAINS 'gene'
WITH c, COUNT(DISTINCT p) as spatial_papers
WHERE spatial_papers > 5

MATCH (c)<-[m2:MENTIONS]-(p2:Passage)
MATCH (p2)-[m3:MENTIONS]->(c2:Concept)
WHERE c2.type IN ['method', 'algorithm', 'math_object']
  AND c2.name != c.name

RETURN c.name as bio_concept,
       c2.name as math_concept,
       COUNT(DISTINCT p2) as bridge_papers
ORDER BY bridge_papers DESC
LIMIT 10
```

Expected result: Concepts like `spatial_transcriptomics` connected to `variational_inference`, `optimal_transport`, etc.

## Rollback Instructions

If you need to revert:

### ChromaDB Rollback

```python
import chromadb

client = chromadb.PersistentClient("/home/user/polymath-repo/chromadb")

# Switch back to legacy collections
# Update lib/config.py:
# PAPERS_COLLECTION = "polymath_papers"
# CODE_COLLECTION = "polymath_code"
```

### Neo4j Rollback

```cypher
// Delete v2 nodes (keeps old graph intact)
MATCH (n:Artifact) DETACH DELETE n;
MATCH (n:Chunk) DETACH DELETE n;
MATCH (n:Passage) DETACH DELETE n;
MATCH (n:Concept) DETACH DELETE n;
```

### Postgres Rollback

Derived tables can be dropped without data loss:

```sql
DROP TABLE IF EXISTS chunk_concepts CASCADE;
DROP TABLE IF EXISTS passage_concepts CASCADE;
DROP TABLE IF EXISTS kb_migrations CASCADE;
```

Core tables (`artifacts`, `chunks`, `passages`) are unchanged.

## Resumability

All scripts support `--resume` (default) to continue from last checkpoint:

```bash
# Will resume automatically
python scripts/backfill_chunk_concepts_llm.py --resume

# Force restart from beginning
python scripts/backfill_chunk_concepts_llm.py --no-resume
```

Checkpoints are stored in `kb_migrations` table.

## Troubleshooting

### Ollama Connection Errors

```bash
# Check Ollama is running
curl http://localhost:11434/api/version

# Start Ollama if needed
ollama serve

# Test model
ollama run qwen2.5:3b "test"
```

### ChromaDB Dimension Mismatch

If you see dimension errors, ensure config uses BGE-M3:

```python
# lib/config.py
EMBEDDING_MODEL = "BAAI/bge-m3"
PAPERS_COLLECTION = "polymath_bge_m3"
CODE_COLLECTION = "polymath_code_bge_m3"
```

### Neo4j Authentication Errors

```bash
# Verify password
export NEO4J_PASSWORD=polymathic2026

# Test connection
python3 -c "from neo4j import GraphDatabase; GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'polymathic2026')).verify_connectivity()"
```

### Low Concept Extraction Rate

If very few concepts are extracted:

```bash
# Test extractor directly
python lib/local_extractor.py

# Check Ollama logs
journalctl -u ollama -f  # If running as service
```

## Performance Notes

**Hardware used for benchmarks:**
- CPU: 24 cores
- RAM: 196GB
- GPU: RTX 5090 24GB

**Observed rates:**
- LLM extraction: 2-5 sec/chunk (depends on model)
- BGE-M3 embedding: ~147 embeddings/sec (GPU)
- Neo4j writes: ~500-1000 nodes/sec

**Estimated total time for 500K passages + 400K chunks:**
- Concept extraction: ~24-50 hours (can run overnight)
- ChromaDB rebuild: ~1-2 hours
- Neo4j rebuild: ~20-30 minutes

**Optimization tips:**
- Run concept extraction in background: `nohup python scripts/backfill_chunk_concepts_llm.py > backfill.log 2>&1 &`
- Use `--limit` for testing: `--limit 1000`
- Use `--batch-size` to control memory: increase for GPU, decrease for CPU

## Schema Reference

### Postgres Derived Tables

```sql
-- chunk_concepts
CREATE TABLE chunk_concepts (
    chunk_id TEXT NOT NULL,
    concept_name TEXT NOT NULL,
    concept_type TEXT NULL,
    aliases JSONB NULL,
    confidence REAL NULL,
    extractor_model TEXT NOT NULL,
    extractor_version TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (chunk_id, concept_name, extractor_version)
);

-- passage_concepts (same structure for passages)
CREATE TABLE passage_concepts (
    passage_id UUID NOT NULL,
    concept_name TEXT NOT NULL,
    concept_type TEXT NULL,
    aliases JSONB NULL,
    confidence REAL NULL,
    extractor_model TEXT NOT NULL,
    extractor_version TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (passage_id, concept_name, extractor_version)
);

-- kb_migrations (checkpoint tracking)
CREATE TABLE kb_migrations (
    job_name TEXT PRIMARY KEY,
    cursor_position TEXT NULL,
    cursor_type TEXT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    status TEXT NOT NULL,
    notes TEXT NULL,
    items_processed INTEGER DEFAULT 0,
    items_failed INTEGER DEFAULT 0,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    finished_at TIMESTAMPTZ NULL
);
```

### Neo4j Schema V2

```cypher
// Nodes
(:Artifact {artifact_id, title, year, source})
(:Chunk {chunk_id, text_preview})
(:Passage {passage_id, text_preview})
(:Concept {name, type})

// Edges
(:Artifact)-[:HAS_CHUNK]->(:Chunk)
(:Artifact)-[:HAS_PASSAGE]->(:Passage)
(:Chunk)-[:MENTIONS {confidence, extractor_version, edge_version, weight}]->(:Concept)
(:Passage)-[:MENTIONS {confidence, extractor_version, edge_version, weight}]->(:Concept)
```

## Support

For issues:
1. Check logs in migration output
2. Run validation: `python scripts/validate_kb_v2.py`
3. Check checkpoints: `SELECT * FROM kb_migrations ORDER BY updated_at DESC;`
4. Verify dependencies: `pip list | grep -E 'ollama|FlagEmbedding|neo4j|chromadb'`
