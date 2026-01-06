# Code Repository Ingestion Design

**Date:** 2026-01-05
**Status:** Draft
**Goal:** Ingest 53+ GitHub repos (11GB) for polymathic cross-referencing

## Use Cases

1. **Coding Reference** - "How does mahmoodlab/UNI implement attention pooling?"
2. **Hypothesis Generation** - "What techniques from GNNs could apply to spatial transcriptomics?"
3. **Implementation Assist** - "Show me examples of contrastive learning in pathology"
4. **Cross-Pollination** - "Find similar patterns across different domains"

## Current State

```
/home/user/work/polymax/data/github_repos/  # 53+ repos, 11GB
├── mahmoodlab/          # UNI, CONCH, CLAM, HIPT
├── theislab/            # squidpy, cell2location, scvi-tools
├── Ken-Lau-Lab/         # spatial_CRC_atlas, dropkick
├── stanfordnlp/dspy/    # Prompt optimization
├── pytorch_geometric/   # GNNs
└── ...
```

## Schema Design

### New Tables

```sql
-- Code files (one row per file)
CREATE TABLE code_files (
    file_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repo_name TEXT NOT NULL,           -- 'mahmoodlab/UNI'
    file_path TEXT NOT NULL,           -- 'models/attention.py'
    language TEXT,                     -- 'python', 'r', 'javascript'
    file_hash TEXT,                    -- SHA256 for dedup
    last_commit TEXT,                  -- Git commit hash
    loc INTEGER,                       -- Lines of code
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(repo_name, file_path)
);

-- Code chunks (semantic units)
CREATE TABLE code_chunks (
    chunk_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_id UUID REFERENCES code_files(file_id),
    chunk_type TEXT,                   -- 'function', 'class', 'module', 'block'
    name TEXT,                         -- 'AttentionPooling', 'forward', etc.
    start_line INTEGER,
    end_line INTEGER,
    content TEXT NOT NULL,
    docstring TEXT,                    -- Extracted docstring
    signature TEXT,                    -- Function/class signature
    imports TEXT[],                    -- Dependencies
    concepts TEXT[],                   -- Auto-tagged concepts
    embedding_id TEXT,                 -- ChromaDB reference
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Cross-references (code <-> papers <-> concepts)
CREATE TABLE code_references (
    id SERIAL PRIMARY KEY,
    chunk_id UUID REFERENCES code_chunks(chunk_id),
    ref_type TEXT,                     -- 'cites_paper', 'implements_concept', 'similar_to'
    ref_id UUID,                       -- doc_id or concept_id or chunk_id
    confidence FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_code_chunks_repo ON code_chunks((file_id));
CREATE INDEX idx_code_chunks_type ON code_chunks(chunk_type);
CREATE INDEX idx_code_chunks_name ON code_chunks(name);
CREATE INDEX idx_code_files_repo ON code_files(repo_name);
CREATE INDEX idx_code_files_lang ON code_files(language);
```

## Ingestion Pipeline

### Phase 1: File Discovery & Metadata

```
For each repo:
  1. Walk directory tree
  2. Filter by extension (.py, .R, .js, .ts, .java, .go, .rs)
  3. Skip: tests/, __pycache__/, node_modules/, .git/
  4. Extract: file hash, LOC, language
  5. Insert into code_files
```

### Phase 2: Semantic Chunking

```
For each file:
  1. Parse AST (tree-sitter or language-specific)
  2. Extract:
     - Classes (with methods)
     - Functions (standalone)
     - Module docstrings
     - Import statements
  3. For each chunk:
     - Extract signature
     - Extract docstring
     - Identify concepts (regex + LLM)
  4. Insert into code_chunks
```

### Phase 3: Embedding & Indexing

```
For each chunk:
  1. Generate embedding (code-specific model: CodeBERT or StarCoder)
  2. Store in ChromaDB collection: 'polymath_code'
  3. Update chunk with embedding_id
```

### Phase 4: Cross-Referencing

```
For each chunk:
  1. Extract paper references (DOIs, arXiv IDs in comments)
  2. Match concepts to existing concept graph
  3. Find similar chunks across repos (embedding similarity)
  4. Insert into code_references
```

## Chunking Strategy

### Python
```python
# tree-sitter or ast module
- Module docstring → chunk_type='module'
- Class definition → chunk_type='class' (includes all methods)
- Standalone function → chunk_type='function'
- Large functions (>50 lines) → split into logical blocks
```

### R
```r
# treesitter.r or custom parser
- Function definitions
- S4/R6 class definitions
- Package documentation
```

### Markdown/Docs
```
- README sections
- Tutorial code blocks
- API documentation
```

## Concept Auto-Tagging

Match against existing concept list + domain patterns:

```python
CONCEPT_PATTERNS = {
    'attention': r'\b(attention|self[-_]?attention|cross[-_]?attention|mha|multihead)\b',
    'contrastive': r'\b(contrastive|simclr|moco|clip|infonce)\b',
    'transformer': r'\b(transformer|bert|gpt|vit|encoder[-_]?decoder)\b',
    'gnn': r'\b(gnn|gcn|gat|message[-_]?passing|node[-_]?embedding)\b',
    'vae': r'\b(vae|variational|elbo|kl[-_]?divergence|reparameterization)\b',
    'spatial': r'\b(spatial|visium|10x|spot|tile|patch)\b',
    # ... 50+ patterns from concept table
}
```

## Priority Order

### Tier 1: Your Research Stack (ingest first)
1. `mahmoodlab/*` - UNI, CONCH, CLAM, HIPT
2. `theislab/*` - squidpy, cell2location
3. `Ken-Lau-Lab/*` - spatial_CRC_atlas

### Tier 2: ML/AI Foundations
4. `stanfordnlp/dspy` - Prompt engineering
5. `pytorch_geometric` - GNNs
6. `huggingface/transformers` - Reference implementations

### Tier 3: Polymathic Bridges
7. `giotto-ai/giotto-tda` - Topological data analysis
8. `infer-actively/pymdp` - Active inference
9. `papers-we-love/*` - Classic implementations

### Tier 4: Utilities & Examples
10. Remaining repos

## Search Interface

```python
# Example queries the system should support:

# 1. Find implementation
search_code("attention pooling implementation", repo="mahmoodlab/*")

# 2. Find by concept
search_code(concepts=["contrastive_learning", "pathology"])

# 3. Find similar code
find_similar_chunks(chunk_id="...")

# 4. Cross-reference paper -> code
find_implementations(paper_doi="10.1038/...")

# 5. Polymathic bridge
find_bridges(concept1="compressed_sensing", concept2="gene_expression")
```

## Implementation Steps

### Step 1: Schema (10 min)
- [ ] Create tables
- [ ] Add indexes

### Step 2: File Scanner (30 min)
- [ ] `lib/code_scanner.py` - walks repos, extracts metadata
- [ ] Language detection
- [ ] Dedup by hash

### Step 3: Python Parser (1 hr)
- [ ] AST-based chunking
- [ ] Docstring extraction
- [ ] Import analysis

### Step 4: Embedding Pipeline (30 min)
- [ ] CodeBERT or all-mpnet-base-v2 (simpler)
- [ ] New ChromaDB collection

### Step 5: Concept Tagger (30 min)
- [ ] Regex patterns
- [ ] Optional LLM refinement

### Step 6: MCP Integration (30 min)
- [ ] `search_code()` tool
- [ ] `find_implementations()` tool

### Step 7: CLI Commands (20 min)
- [ ] `polymath_cli.py code search "query"`
- [ ] `polymath_cli.py code ingest /path/to/repo`

## Incremental Ingestion

```bash
# Ingest single repo
python3 -m lib.code_ingest mahmoodlab/UNI

# Ingest tier
python3 -m lib.code_ingest --tier 1

# Ingest all (background)
python3 -m lib.code_ingest --all --workers 4 &
```

## Storage Estimates

| Component | Size |
|-----------|------|
| code_files rows | ~50K files |
| code_chunks rows | ~200K chunks |
| ChromaDB embeddings | ~500MB |
| Total DB growth | ~1GB |

## Success Criteria

1. Can search "attention mechanism" and get ranked results from UNI, HIPT, transformers
2. Can find code that implements concepts from papers
3. Can discover similar implementations across repos
4. Ingestion completes in <1 hour for all 53 repos
