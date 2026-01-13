# Polymath 2.0 Architecture

**Author**: Max Van Belkum
**Version**: 2.0.0-alpha
**Last Updated**: 2026-01-13

---

## Table of Contents

1. [Design Philosophy](#design-philosophy)
2. [System Components](#system-components)
3. [Data Flow](#data-flow)
4. [Storage Layer](#storage-layer)
5. [Query Layer](#query-layer)
6. [Discovery Engine](#discovery-engine)
7. [Integration Points](#integration-points)
8. [Security Model](#security-model)
9. [Performance Considerations](#performance-considerations)

---

## Design Philosophy

### Core Principles

1. **Zotero is Source of Truth for Metadata**
   - All PDFs enter through Zotero
   - Metadata (DOI, PMID, authors, venue) validated at entry
   - Eliminates 80% missing DOI problem from v1.0

2. **Passages Have Hierarchy**
   - Every passage knows its parent section and document
   - Enables context expansion during retrieval
   - Supports drill-down from claim to evidence

3. **Mechanisms, Not Labels**
   - Extract HOW methods work, not just WHAT they're called
   - Enable reasoning about cross-domain transfer
   - Support actionable hypothesis generation

4. **Grounded Reasoning**
   - Every claim must cite specific evidence
   - Contradictions flagged automatically
   - Audit trail for all reasoning

5. **Transfer Validation**
   - Cross-domain hypotheses require mechanism matching
   - Same mechanism + compatible data structure + different domain = valid transfer
   - Novelty checking against existing literature

### What We Learned from v1.0

| Problem | Cause | Solution in v2.0 |
|---------|-------|------------------|
| 80% missing DOI | PDF-first workflow | Zotero-first workflow |
| Decontextualized passages | Flat 500-char chunks | Hierarchical 1500-char passages |
| Garbage hypotheses | Label-based similarity | Mechanism-based matching |
| Unverifiable claims | No citation integration | Evidence validation layer |
| Disconnected components | Ad-hoc integration | Unified architecture |

---

## System Components

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           USER INTERFACES                                │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │
│  │   CLI   │  │   Web   │  │   MCP   │  │   API   │  │ Jupyter │       │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘       │
└───────┼────────────┼────────────┼────────────┼────────────┼─────────────┘
        │            │            │            │            │
        └────────────┴────────────┴─────┬──────┴────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          QUERY ORCHESTRATOR                              │
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │
│  │    Query     │  │   Evidence   │  │   Response   │                   │
│  │  Decomposer  │  │  Validator   │  │  Synthesizer │                   │
│  └──────────────┘  └──────────────┘  └──────────────┘                   │
└─────────────────────────────────────────────────────────────────────────┘
                                        │
        ┌───────────────┬───────────────┼───────────────┬─────────────────┐
        │               │               │               │                 │
        ▼               ▼               ▼               ▼                 ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   Semantic   │ │    Graph     │ │   Keyword    │ │    Code      │ │   External   │
│   Search     │ │   Search     │ │   Search     │ │   Search     │ │    APIs      │
│  (ChromaDB)  │ │   (Neo4j)    │ │ (Postgres)   │ │  (Unified)   │ │ (PubMed etc) │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
        │               │               │               │                 │
        └───────────────┴───────────────┴───────┬───────┴─────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         STORAGE LAYER                                    │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                      PostgreSQL (Source of Record)                │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐  │   │
│  │  │ documents  │  │  passages  │  │   code     │  │ migrations │  │   │
│  │  │    _v2     │  │    _v2     │  │  _chunks   │  │            │  │   │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐ │
│  │      ChromaDB      │  │       Neo4j        │  │    File Archive    │ │
│  │  (Vector Index)    │  │  (Knowledge Graph) │  │   (PDFs, Repos)    │ │
│  └────────────────────┘  └────────────────────┘  └────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Technology |
|-----------|---------------|------------|
| CLI | Interactive terminal interface | Python/Click |
| Web | Browser-based UI | FastAPI + HTMX |
| MCP | Claude Code integration | MCP Protocol |
| API | Programmatic access | FastAPI REST |
| Query Orchestrator | Route and combine queries | Python |
| Semantic Search | Vector similarity | ChromaDB + BGE-M3 |
| Graph Search | Relationship traversal | Neo4j + Cypher |
| Keyword Search | Full-text search | PostgreSQL FTS |
| Code Search | Implementation lookup | Custom index |
| External APIs | Literature validation | PubMed, S2, etc. |

---

## Data Flow

### Ingestion Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                                │
└─────────────────────────────────────────────────────────────────────────┘

Step 1: Zotero Export
┌──────────┐     ┌──────────┐     ┌──────────┐
│  Zotero  │────▶│  Export  │────▶│   CSV    │
│ Library  │     │  Plugin  │     │ + PDFs   │
└──────────┘     └──────────┘     └────┬─────┘
                                       │
Step 2: Metadata Validation            ▼
                               ┌──────────────┐
                               │   Validate   │
                               │  - DOI       │
                               │  - PMID      │
                               │  - Authors   │
                               │  - Year      │
                               └──────┬───────┘
                                      │
Step 3: PDF Parsing                   ▼
                               ┌──────────────┐
                               │    Parse     │
                               │  - Sections  │
                               │  - Figures   │
                               │  - Tables    │
                               │  - Refs      │
                               └──────┬───────┘
                                      │
Step 4: Hierarchy Construction        ▼
                               ┌──────────────┐
                               │   Build      │
                               │  Hierarchy   │
                               │  - Doc       │
                               │  - Section   │
                               │  - Paragraph │
                               │  - Sentence  │
                               └──────┬───────┘
                                      │
Step 5: Concept Extraction            ▼
                               ┌──────────────┐
                               │   Extract    │
                               │  - Methods   │
                               │  - Mechanisms│
                               │  - DataTypes │
                               │  - Problems  │
                               └──────┬───────┘
                                      │
Step 6: Storage                       ▼
                    ┌─────────────────┼─────────────────┐
                    │                 │                 │
                    ▼                 ▼                 ▼
             ┌──────────┐     ┌──────────┐     ┌──────────┐
             │ Postgres │     │ ChromaDB │     │  Neo4j   │
             │(passages)│     │(vectors) │     │ (graph)  │
             └──────────┘     └──────────┘     └──────────┘
```

### Query Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          QUERY PIPELINE                                  │
└─────────────────────────────────────────────────────────────────────────┘

Step 1: Query Reception
┌──────────┐
│  User    │
│  Query   │
└────┬─────┘
     │
Step 2: Query Analysis
     ▼
┌──────────────┐
│   Analyze    │
│  - Intent    │
│  - Entities  │
│  - Scope     │
└──────┬───────┘
       │
Step 3: Query Decomposition
       ▼
┌──────────────┐     ┌────────────────────────────────┐
│  Decompose   │────▶│  Sub-queries:                  │
│              │     │  1. Find relevant methods      │
│              │     │  2. Get mechanism details      │
│              │     │  3. Check domain applications  │
│              │     │  4. Validate novelty           │
└──────────────┘     └────────────────────────────────┘
       │
Step 4: Parallel Retrieval
       ▼
┌──────────────────────────────────────────────────────┐
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐     │
│  │Semantic│  │ Graph  │  │Keyword │  │External│     │
│  │ Search │  │ Search │  │ Search │  │  APIs  │     │
│  └───┬────┘  └───┬────┘  └───┬────┘  └───┬────┘     │
│      │           │           │           │          │
│      └───────────┴─────┬─────┴───────────┘          │
│                        │                            │
│                        ▼                            │
│               ┌──────────────┐                      │
│               │   Reciprocal │                      │
│               │ Rank Fusion  │                      │
│               └──────────────┘                      │
└──────────────────────────────────────────────────────┘
       │
Step 5: Context Expansion
       ▼
┌──────────────┐
│   Expand     │
│  - Parents   │
│  - Siblings  │
│  - Children  │
└──────┬───────┘
       │
Step 6: Evidence Validation
       ▼
┌──────────────┐
│  Validate    │
│  - NLI check │
│  - Citation  │
│  - Conflicts │
└──────┬───────┘
       │
Step 7: Response Synthesis
       ▼
┌──────────────┐     ┌────────────────────────────────┐
│  Synthesize  │────▶│  Response with:                │
│              │     │  - Answer                      │
│              │     │  - Citations [1,2,3]           │
│              │     │  - Confidence score            │
│              │     │  - Reasoning trace             │
└──────────────┘     └────────────────────────────────┘
```

---

## Storage Layer

### PostgreSQL Schema (Source of Record)

See [schemas/postgres_schema.sql](../schemas/postgres_schema.sql) for full schema.

**Key Tables**:

```sql
-- Documents with Zotero metadata
documents_v2 (
    doc_id UUID PRIMARY KEY,
    zotero_key VARCHAR(50) UNIQUE,  -- Canonical identifier
    doi VARCHAR(100),
    pmid VARCHAR(20),
    title TEXT NOT NULL,
    authors JSONB,
    year INT,
    venue TEXT,
    abstract TEXT,
    ...
)

-- Hierarchical passages
passages_v2 (
    passage_id UUID PRIMARY KEY,
    doc_id UUID REFERENCES documents_v2,
    parent_id UUID REFERENCES passages_v2,  -- Hierarchy
    level INT,  -- 0=doc, 1=section, 2=para, 3=sentence
    position INT,  -- Order within parent
    passage_text TEXT,
    passage_type VARCHAR(50),  -- abstract, methods, results...
    context_before TEXT,
    context_after TEXT,
    ...
)

-- Extracted concepts with mechanisms
passage_concepts_v2 (
    passage_id UUID REFERENCES passages_v2,
    concept_name VARCHAR(255),
    concept_type VARCHAR(50),  -- method, mechanism, data_structure, problem
    mechanism_description TEXT,  -- NEW: How it works
    data_structure TEXT,  -- NEW: What it operates on
    objective TEXT,  -- NEW: What it optimizes
    confidence FLOAT,
    ...
)
```

### ChromaDB Configuration

```python
# Collection configuration
collection_config = {
    "name": "polymath_v2_bge_m3",
    "embedding_model": "BAAI/bge-m3",
    "embedding_dim": 1024,
    "distance_metric": "cosine",
    "hnsw_config": {
        "M": 32,
        "ef_construction": 200,
        "ef_search": 100
    }
}
```

### Neo4j Graph Schema

See [schemas/neo4j_schema.cypher](../schemas/neo4j_schema.cypher) for full schema.

**Node Types**:

```cypher
// Method nodes
(:Method {
    name: String,
    aliases: [String],
    paper_count: Int,
    first_mentioned: Int
})

// Mechanism nodes (NEW)
(:Mechanism {
    name: String,
    description: String,
    properties: [String],
    mathematical_form: String
})

// Data structure nodes (NEW)
(:DataStructure {
    name: String,
    description: String,
    features: [String],
    examples: [String]
})

// Problem nodes
(:Problem {
    name: String,
    domain: String,
    data_characteristics: [String],
    success_metrics: [String]
})

// Domain nodes
(:Domain {
    name: String,
    parent: String,
    data_types: [String],
    key_challenges: [String]
})
```

**Edge Types**:

```cypher
(:Method)-[:IMPLEMENTS]->(:Mechanism)
(:Mechanism)-[:OPERATES_ON]->(:DataStructure)
(:Mechanism)-[:OPTIMIZES]->(:Objective)
(:DataStructure)-[:APPEARS_IN]->(:Domain)
(:Problem)-[:REQUIRES]->(:Mechanism)
(:Method)-[:APPLIED_TO]->(:Problem)
```

---

## Query Layer

### Query Types

| Type | Description | Primary Store | Example |
|------|-------------|---------------|---------|
| Semantic | Find conceptually similar | ChromaDB | "papers about spatial gene prediction" |
| Exact | Find specific entities | PostgreSQL | "papers by Ken Lau from 2023" |
| Graph | Traverse relationships | Neo4j | "methods that use distributional matching" |
| Cross-modal | Link code to papers | Unified | "implementations of optimal transport for ST" |
| Discovery | Find transfer opportunities | All | "methods from CV applicable to pathology" |

### Hybrid Search with RRF

```python
def hybrid_search(query: str, n: int = 20) -> List[Result]:
    """
    Combine multiple search modalities using Reciprocal Rank Fusion.
    """
    # 1. Get results from each store
    semantic_results = chromadb_search(query, n=100)
    keyword_results = postgres_fts_search(query, n=100)
    graph_results = neo4j_concept_search(query, n=100)

    # 2. Compute RRF scores
    k = 60  # RRF constant
    scores = defaultdict(float)

    for rank, result in enumerate(semantic_results):
        scores[result.id] += 1 / (k + rank + 1)

    for rank, result in enumerate(keyword_results):
        scores[result.id] += 1 / (k + rank + 1)

    for rank, result in enumerate(graph_results):
        scores[result.id] += 1 / (k + rank + 1)

    # 3. Sort by combined score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # 4. Expand context and return
    return [expand_context(doc_id) for doc_id, score in ranked[:n]]
```

---

## Discovery Engine

### Gap Detection Query

```cypher
// Find methods with mechanisms not yet applied to target domain
MATCH (m:Method)-[:IMPLEMENTS]->(mech:Mechanism)
      -[:OPERATES_ON]->(ds:DataStructure)
WHERE ds.name IN ['point_cloud', 'spatial_distribution', 'graph_structure']

// Check if NOT applied to spatial transcriptomics
AND NOT EXISTS {
    MATCH (m)-[:APPLIED_TO]->(p:Problem)-[:IN_DOMAIN]->(d:Domain)
    WHERE d.name = 'spatial_transcriptomics'
}

// Return with mechanism details
RETURN m.name AS method,
       mech.name AS mechanism,
       mech.description AS how_it_works,
       ds.name AS operates_on,
       collect(DISTINCT d2.name) AS current_domains
ORDER BY m.paper_count DESC
LIMIT 50
```

### Transfer Validation

```python
def validate_transfer(
    source_method: str,
    target_domain: str
) -> TransferValidation:
    """
    Validate whether a cross-domain transfer makes mechanistic sense.
    """
    # 1. Get method's mechanism
    mechanism = get_mechanism(source_method)

    # 2. Get target domain's data characteristics
    target_data = get_domain_data_types(target_domain)

    # 3. Check compatibility
    compatibility = check_data_compatibility(
        mechanism.operates_on,
        target_data
    )

    # 4. Check novelty
    novelty = check_novelty(
        method=source_method,
        domain=target_domain,
        sources=['pubmed', 'semantic_scholar', 'internal']
    )

    # 5. Generate validation report
    return TransferValidation(
        is_valid=compatibility.score > 0.7 and novelty.score > 0.5,
        compatibility=compatibility,
        novelty=novelty,
        reasoning=generate_reasoning(mechanism, target_data)
    )
```

---

## Integration Points

### External APIs

| API | Purpose | Rate Limit | Auth |
|-----|---------|------------|------|
| PubMed | Literature validation | 10 req/s | API key |
| Semantic Scholar | Citation data | 100 req/5min | API key |
| OpenAlex | Open access metadata | 100 req/s | None |
| Unpaywall | PDF access | 100 req/s | Email |
| CrossRef | DOI resolution | 50 req/s | None |

### MCP Server Integration

```python
# MCP tools exposed
tools = [
    "semantic_search",      # Vector similarity search
    "graph_search",         # Neo4j traversal
    "find_gaps",            # Gap detection
    "validate_transfer",    # Transfer validation
    "generate_hypothesis",  # Hypothesis generation
    "get_citations",        # Citation retrieval
    "ingest_paper",         # Paper ingestion
    "ingest_repo",          # Repository ingestion
]
```

---

## Security Model

### Authentication

- Local deployment: No authentication (single-user)
- Shared deployment: API key authentication
- Sensitive data: Never stored (API keys in environment)

### Data Handling

- PDFs: Stored locally, not transmitted
- Embeddings: Local ChromaDB, not cloud
- API calls: Rate-limited, logged
- User queries: Logged for audit (can be disabled)

---

## Performance Considerations

### Indexing Performance

| Operation | Target | Current v1.0 |
|-----------|--------|--------------|
| PDF ingest | 30 sec/paper | 45 sec/paper |
| Embedding generation | 100 passages/sec | 50 passages/sec |
| Graph insertion | 1000 nodes/sec | 500 nodes/sec |

### Query Performance

| Query Type | Target Latency | Current v1.0 |
|------------|----------------|--------------|
| Semantic search (n=20) | <500ms | 800ms |
| Graph traversal (2-hop) | <200ms | 400ms |
| Hybrid search | <1s | 2s |
| Full discovery pipeline | <10s | 30s |

### Optimization Strategies

1. **Batch embedding generation** - Process passages in batches of 32
2. **HNSW index tuning** - Optimize M and ef parameters for recall/speed
3. **Query caching** - Cache frequent queries (LRU, 1hr TTL)
4. **Parallel retrieval** - Execute search modalities concurrently
5. **Incremental indexing** - Only re-index changed documents
