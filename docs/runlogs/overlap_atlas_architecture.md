# Polymath Overlap Atlas - Retrieval Architecture

**Generated**: 2026-01-09

## Overview

The Polymath system implements tri-modal search combining SQL concept queries, vector embeddings, and graph traversal. This document summarizes the architecture for the Overlap Atlas project.

## 1. SQL Concept Search (Postgres)

### Tables
- **passage_concepts**: 1.52M rows linking passages to concepts
  - `passage_id` (UUID) → `passages` table
  - `concept_name` (TEXT) - normalized concept
  - `concept_type` (TEXT) - one of 13 types: domain, method, metric, technique, model, field, math_object, dataset, objective, algorithm, prior, extracted, architecture
  - `confidence` (REAL) - 0.0-1.0, mean 0.81
  - `evidence` (JSONB) - source_text, quality.confidence, surface
  - `extractor_version` - "gemini_batch_v1" for current data

### Indexes
- Primary key: (passage_id, concept_name, extractor_version)
- `idx_passage_concepts_name` - fast concept lookup
- `idx_passage_concepts_name_ver` - version-filtered

### Query Patterns
```sql
-- Find passages by concept
SELECT p.passage_id, p.passage_text
FROM passages p
JOIN passage_concepts pc ON p.passage_id = pc.passage_id
WHERE pc.concept_name = 'spatial_transcriptomics'
  AND pc.extractor_version = 'gemini_batch_v1';

-- Co-occurrence query
SELECT pc1.concept_name as concept_a, pc2.concept_name as concept_b, COUNT(*) as cooccur
FROM passage_concepts pc1
JOIN passage_concepts pc2 ON pc1.passage_id = pc2.passage_id
WHERE pc1.concept_name < pc2.concept_name
GROUP BY pc1.concept_name, pc2.concept_name;
```

### Key File
`lib/kb_derived.py` - Schema definitions, upsert/query functions

---

## 2. Vector Search (ChromaDB)

### Collections
- `polymath_bge_m3`: 395K paper passages
- `polymath_code_bge_m3`: 411K code chunks
- **Embedding Model**: BAAI/bge-m3 (1024-dim)

### Class: `HybridSearcherV2`
**File**: `lib/hybrid_search_v2.py`

```python
class HybridSearcherV2:
    def search_papers(query, n=20, year_min=None, concepts=None) -> List[SearchResult]
    def search_code(query, n=20, repo_filter=None, language=None) -> List[SearchResult]
    def lexical_search_passages(query, n=20) -> List[SearchResult]
    def hybrid_search(query, n=20, rerank=True) -> List[SearchResult]
```

### SearchResult Dataclass
```python
@dataclass
class SearchResult:
    id: str           # Document ID
    title: str        # Paper title
    content: str      # Text snippet
    source: str       # 'papers', 'code', 'lexical', 'graph'
    score: float      # 0-1 relevance
    metadata: Dict    # year, concepts, doi, pmid, etc.
```

### RRF Merge
Reciprocal Rank Fusion with k=60 combines multiple result lists:
```python
def _rrf_merge(result_lists, k=60):
    for results in result_lists:
        for rank, result in enumerate(results):
            scores[result.id] += 1 / (k + rank + 1)
```

---

## 3. Graph Search (Neo4j)

### Schema
- **Nodes**: Paper (31,867), CONCEPT (222)
- **Relationships**: MENTIONS, CITES, USES, CO_OCCURS
- **Connection**: bolt://localhost:7687

### Class: `GraphBridgeSearcher`
**File**: `lib/graph_bridge_search.py`

```python
class GraphBridgeSearcher:
    def find_bridge_paths(concept_a, concept_b, max_hops=3) -> List[Dict]
    def expand_concept_neighborhood(concept, n_related=10) -> List[str]
```

### Cypher Patterns
```cypher
-- Find concept bridges
MATCH path = (ca:CONCEPT)-[*1..3]-(bridge:CONCEPT)-[*1..3]-(cb:CONCEPT)
WHERE toLower(ca.name) CONTAINS $concept_a
RETURN bridge, length(path) as hops

-- Concept neighborhood
MATCH (c:CONCEPT)-[r:RELATES_TO|CO_OCCURS]-(related:CONCEPT)
WHERE toLower(c.name) CONTAINS $concept
RETURN related.name ORDER BY r.strength DESC
```

---

## 4. Hybrid Search Pipeline

```
USER QUERY
    │
    ▼
┌─────────────────────────────────────────┐
│          HYBRID SEARCH (RRF)            │
│     lib/hybrid_search_v2.py             │
└───────┬────────┬────────┬───────────────┘
        │        │        │
   ┌────▼────┐┌──▼──┐┌────▼────┐
   │ Vector  ││Lexical│ Graph  │
   │ChromaDB ││FTS    ││Neo4j  │
   │807K vec ││       ││32K    │
   └────┬────┘└──┬──┘└────┬────┘
        │        │        │
        └────────┼────────┘
                 │
           ┌─────▼─────┐
           │ RRF Merge │
           │  k=60     │
           └─────┬─────┘
                 │
           ┌─────▼─────────┐
           │Cross-Encoder  │
           │ Reranking     │
           └─────┬─────────┘
                 │
           ┌─────▼─────┐
           │ Top K     │
           │ Results   │
           └───────────┘
```

---

## 5. For Overlap Atlas: New Components

### atlas_search() Method
Extends HybridSearcherV2 with:
1. **SQL concept branch**: Direct passage_concepts query
2. **Explainability trace**: Which channel contributed what
3. **Evidence extraction**: Pull quotes from evidence JSONB

### Co-occurrence Analysis
New SQL patterns for PMI calculation:
```sql
-- Field-field co-occurrence
WITH field_passages AS (
  SELECT passage_id, concept_name as field
  FROM passage_concepts
  WHERE concept_type = 'domain' AND confidence >= 0.7
)
SELECT f1.field, f2.field, COUNT(DISTINCT f1.passage_id) as cooccur
FROM field_passages f1
JOIN field_passages f2 ON f1.passage_id = f2.passage_id
WHERE f1.field < f2.field
GROUP BY f1.field, f2.field;
```

---

## 6. Data Coverage (Current State)

| Store | Items | Notes |
|-------|-------|-------|
| passage_concepts | 1.52M | 36% of passages labeled |
| ChromaDB papers | 395K | Full coverage |
| ChromaDB code | 411K | Full coverage |
| Neo4j papers | 31,867 | Partial hydration |
| Neo4j concepts | 222 | Core concept nodes |

---

## Key Files Reference

| Component | Path |
|-----------|------|
| Hybrid Search | `lib/hybrid_search_v2.py` |
| Graph Search | `lib/graph_bridge_search.py` |
| Concept Schema | `lib/kb_derived.py` |
| Config | `lib/config.py` |
| Database | `lib/db.py` |
| Reasoning Tools | `mcp/polymath_v11/tools/reasoning.py` |
