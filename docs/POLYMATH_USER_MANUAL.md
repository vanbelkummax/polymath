# Polymath: A Polymathic Research Intelligence System

## User Manual v1.0

**Author**: Max Van Belkum
**Institution**: Vanderbilt University, Huo Lab
**Last Updated**: 2026-01-06

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Data Inventory](#3-data-inventory)
4. [Tools & Capabilities](#4-tools--capabilities)
5. [Inner Workings](#5-inner-workings)
6. [Usage Examples](#6-usage-examples)
7. [Future Directions](#7-future-directions)
8. [Academic Playbook](#8-academic-playbook)
9. [Hackathon Playbook](#9-hackathon-playbook)

---

## 1. Executive Summary

### What is Polymath?

Polymath is a **self-growing research intelligence system** designed for polymathic discovery—finding unexpected connections across disparate scientific domains. Unlike traditional literature databases that organize knowledge in silos, Polymath explicitly models **cross-domain bridges**: the fertile intersections where spatial transcriptomics meets information theory, where thermodynamics informs deep learning, where cybernetics illuminates biological systems.

### The Core Insight

The most transformative scientific breakthroughs often come from **analogical reasoning across fields**:
- Shannon's information theory revolutionized genetics (DNA as code)
- Hopfield networks brought statistical physics to neural computation
- Compressed sensing transformed MRI and single-cell sequencing

Polymath is built to systematically surface these cross-domain opportunities.

### Key Statistics

| Component | Count | Description |
|-----------|-------|-------------|
| **Papers** | 30,283 | Full-text indexed scientific papers |
| **Passages** | 532,259 | Searchable text chunks (396K citable) |
| **Code Chunks** | 411,729 | Functions, classes, methods from 90 organizations |
| **Concepts** | 222 | Cross-domain concept nodes |
| **Graph Edges** | 26,820 | Semantic relationships |
| **GitHub Repos** | 154 | Curated code repositories (23GB) |

### Design Philosophy

1. **Quality over quantity**: Curated, full-text papers rather than metadata-only databases
2. **Cross-domain by design**: Every paper tagged with concepts spanning physics, biology, ML, systems theory
3. **Code + Papers together**: Implementations live alongside the papers that describe them
4. **Evidence-bound citations**: Claims traced to page-level source text (no hallucinations)
5. **Self-improving**: The system identifies its own coverage gaps and proposes expansions

---

## 2. System Architecture

### 2.1 Three-Database Architecture

Polymath uses a polyglot persistence architecture where each database excels at different query patterns:

```
┌─────────────────────────────────────────────────────┐
│                    USER QUERY                        │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│           HYBRID SEARCH (RRF Fusion)                │
│         Reciprocal Rank Fusion across all stores    │
└─────────────────────┬───────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
┌─────────────┐ ┌──────────┐ ┌────────────┐
│  ChromaDB   │ │ Postgres │ │   Neo4j    │
│  960K vecs  │ │  (FTS)   │ │  (graph)   │
│  2 colls    │ │  SoR     │ │ 32K nodes  │
└─────────────┘ └──────────┘ └────────────┘
```

#### PostgreSQL (System of Record)
- **Role**: Ground truth for all data, full-text search, metadata storage
- **Tables**: artifacts, passages, code_chunks, concepts, documents
- **Why**: ACID compliance, complex queries, joins, aggregations

#### ChromaDB (Vector Store)
- **Role**: Semantic similarity search via embeddings
- **Collections**:
  - `polymath_bge_m3`: 545,099 passage embeddings
  - `polymath_code_bge_m3`: 415,468 code chunk embeddings
- **Why**: Sub-second semantic search, filtering by year/concept/language

#### Neo4j (Knowledge Graph)
- **Role**: Multi-hop reasoning, concept co-occurrence, relationship traversal
- **Nodes**: Paper (31,867), CONCEPT (222), Code, Repo
- **Edges**: MENTIONS, CO_OCCURS, USES, CITES
- **Why**: "Papers that mention X and Y but not Z", concept bridging

### 2.2 Hybrid Search Pipeline

Every query runs through three parallel search paths:

```python
def hybrid_search(query, n=20):
    # 1. Semantic search (ChromaDB)
    semantic_results = chromadb.query(embed(query), n=50)

    # 2. Lexical search (Postgres FTS)
    lexical_results = postgres.fts_query(query, n=50)

    # 3. Graph traversal (Neo4j)
    concepts = extract_concepts(query)
    graph_results = neo4j.concept_search(concepts, n=50)

    # Reciprocal Rank Fusion
    return rrf_merge([semantic, lexical, graph], k=60)[:n]
```

**Reciprocal Rank Fusion (RRF)** combines rankings without requiring score calibration:
```
RRF_score(doc) = Σ 1/(k + rank_in_list_i)
```

### 2.3 Evidence-Bound Citation System

Every claim can be traced to source text:

```
Question → Decompose (claims) → Verify (NLI) → Synthesize (citations)
```

**Key components**:
- `lib/doc_identity.py`: Deterministic UUIDv5 document fingerprinting
- `lib/enhanced_pdf_parser.py`: Page-local text extraction with character offsets
- `lib/evidence_extractor.py`: Batched Natural Language Inference (NLI) scoring
- `lib/citation_builder.py`: Multi-format citation generation

**Citability gate**: Only passages with valid `page_num >= 0` can be cited.

---

## 3. Data Inventory

### 3.1 Papers by Domain

| Domain | Papers | Description |
|--------|--------|-------------|
| Medical Imaging | 7,469 | MRI, CT, X-ray analysis, segmentation |
| Genomics | 1,339 | Gene expression, sequencing, variants |
| Cancer/Oncology | 750 | Tumor biology, treatment response |
| Systems Theory | 714 | Cybernetics, feedback, emergence |
| Deep Learning/AI | 587 | Transformers, attention, neural nets |
| Graph/Network Science | 481 | Network analysis, GNNs, connectivity |
| Computational Pathology | 254 | Histology, H&E analysis, digital path |
| Optimization/OR | 244 | Operations research, scheduling |
| Structural Biology | 219 | Protein structure, molecular modeling |
| Computer Vision | 211 | Detection, segmentation, classification |
| Spatial Transcriptomics | 110 | ST methods, Visium, spatial patterns |
| Single-Cell Biology | 106 | scRNA-seq, cell typing, trajectories |
| Information Theory | 94 | Entropy, mutual information, coding |
| Probabilistic Methods | 75 | Bayesian inference, uncertainty |
| Microbiome | 43 | Bacteria, metagenomics, E. coli |
| Causal Inference | 29 | Counterfactuals, interventions |
| Colorectal Research | 29 | CRC, colon biology |
| IBD/GI Pathology | 20 | Inflammatory bowel disease |
| Other/Cross-Domain | 17,509 | Polymathic, interdisciplinary |

### 3.2 Concept Distribution (Top 25)

These concepts are automatically detected and linked:

| Concept | Papers | Domain Bridge |
|---------|--------|---------------|
| classification | 1,321 | ML → Biology |
| clustering | 1,270 | ML → Biology |
| regression | 860 | Statistics → All |
| diffusion | 799 | Physics → ML → Biology |
| emergence | 698 | Systems → Complexity |
| segmentation | 644 | CV → Medical |
| gene_expression | 566 | Biology core |
| entropy | 543 | Physics → Information |
| bayesian | 491 | Stats → All |
| transformer | 457 | ML → NLP → Biology |
| single_cell | 407 | Biology core |
| thermodynamics | 372 | Physics → Biology |
| cell_type | 350 | Biology core |
| feedback | 312 | Systems → Control |
| fourier | 227 | Signal → Imaging |
| attention | 222 | ML → Interpretation |
| cybernetics | 171 | Systems theory |
| free_energy | 168 | Physics → Neuro |
| spatial_transcriptomics | 146 | Biology core |

### 3.3 Code Repository Collection

**Total**: 411,729 code chunks from 90 organizations

#### By Organization (Top 20)

| Organization | Chunks | Focus Area |
|--------------|--------|------------|
| python | 62,127 | Standard library |
| huggingface | 55,430 | Transformers, NLP |
| sympy | 31,276 | Symbolic math |
| vllm-project | 22,931 | LLM inference |
| langgenius | 22,412 | LLM applications |
| run-llama | 20,777 | LLM frameworks |
| sgl-project | 20,745 | Structured generation |
| voxel51 | 12,415 | Computer vision |
| langchain-ai | 11,572 | LLM chains |
| cupy | 11,451 | GPU arrays |
| scikit-learn | 11,312 | Classical ML |
| comfyanonymous | 8,280 | Diffusion models |
| napari | 7,634 | Image visualization |
| networkx | 7,598 | Graph algorithms |
| kornia | 7,324 | Computer vision |
| pyg-team | 7,206 | Graph neural networks |
| deepchem | 6,878 | Chemistry ML |
| scverse | 6,430 | Single-cell ecosystem |
| rapidsai | 5,614 | GPU data science |

#### Strategic Tiers

**Tier 1: Vanderbilt Ecosystem**
- `Ken-Lau-Lab/` (319 chunks) - spatial_CRC_atlas, dropkick
- `hrlblab/` (1,306 chunks) - Map3D, PathSeg
- `MASILab/` (2,599 chunks) - SLANTbrainSeg, PreQual

**Tier 2: Virtual Sequencing Stack (H&E → ST)**
- `mahmoodlab/` (1,730 chunks) - UNI, CONCH, CLAM, HIPT
- `theislab/` (978 chunks) - squidpy, cell2location, scvi-tools
- `owkin/` (342 chunks) - HistoSSLscaling

**Tier 3: Polymath Stack**
- `stanfordnlp/` (1,886 chunks) - DSPy prompt optimization
- `infer-actively/` (432 chunks) - pymdp active inference
- `giotto-ai/` (959 chunks) - Topological data analysis

#### By Language

| Language | Chunks | Percentage |
|----------|--------|------------|
| Python | 389,000+ | 94.5% |
| TypeScript | 11,000+ | 2.7% |
| Markdown | 6,700+ | 1.6% |
| C/C++ | 4,400+ | 1.1% |
| Rust | 2,400+ | 0.6% |

---

## 4. Tools & Capabilities

### 4.1 MCP Server Tools (Polymath v11)

The system exposes tools via the Model Context Protocol (MCP):

#### Discovery Tools

| Tool | Maturity | Description |
|------|----------|-------------|
| `deep_hunt` | ✅ Stable | Multi-source literature search (corpus + Europe PMC + arXiv + bioRxiv + GitHub) |
| `find_gaps` | ✅ Stable | Detect research gaps: orphan concepts, time gaps, method gaps |
| `watch_competitor` | 🔸 Alpha | Track lab/author publications and methods |
| `trend_radar` | 🔶 Beta | Analyze publication velocity and emerging methods |
| `find_datasets` | 🔸 Alpha | Hunt for datasets across GEO, SRA, Zenodo |

#### Reasoning Tools

| Tool | Maturity | Description |
|------|----------|-------------|
| `generate_hypothesis` | ✅ Stable | Cross-domain hypothesis generation with novelty scoring |
| `validate_hypothesis` | 🔶 Beta | Validate claims against corpus with NLI |
| `find_analogy` | 🔶 Beta | Find analogous solutions from unexpected domains |
| `serendipity` | ✅ Stable | Surface unexpected concept bridges |

#### Self-Improvement Tools

| Tool | Maturity | Description |
|------|----------|-------------|
| `collection_health` | ✅ Stable | Assess recency, coverage gaps, citability |
| `expand_collection` | 🔶 Beta | Propose expansion targets by strategy |

### 4.2 Literature Sentry (Autonomous Curator)

Autonomous discovery and ingestion system:

```bash
# Full polymathic sweep
python3 -m lib.sentry.cli "spatial transcriptomics visium" --max 15

# GitHub only with star threshold
python3 -m lib.sentry.cli "attention mechanism" --source github --min-stars 100

# Show paywalled items needing manual download
python3 -m lib.sentry.cli --show-tickets
```

**Sources searched**:
- Europe PMC (preferred over PubMed)
- arXiv (preprints)
- bioRxiv/medRxiv (bio preprints)
- GitHub (code implementations)
- Brave Search (fallback for PDFs)

**Quality scoring**:
- Log-normalized citation/star counts
- Cross-domain bridge bonus (0.30 weight)
- Hidden gem detection (low popularity + high relevance)
- Lab whitelist: mahmoodlab, Ken-Lau-Lab, theislab, hrlblab

### 4.3 Hybrid Search API

```python
from lib.hybrid_search_v2 import HybridSearcherV2
hs = HybridSearcherV2()

# Papers with filtering
results = hs.search_papers(
    "attention mechanism in pathology",
    n=10,
    year_min=2023,
    concepts=["transformer", "segmentation"]
)

# Code with org/language filtering
results = hs.search_code(
    "multiple instance learning pooling",
    n=10,
    org_filter="mahmoodlab",
    language="python"
)

# Hybrid (papers + code + lexical, RRF merged)
results = hs.hybrid_search(
    "predict gene expression from H&E",
    n=20,
    rerank=True  # Cross-encoder reranking
)
```

### 4.4 CLI Interface

```bash
CLI="python3 /home/user/work/polymax/polymath_cli.py"

# System stats
$CLI stats

# Semantic search
$CLI search "compressed sensing spatial transcriptomics" -n 10

# Neo4j graph query
$CLI graph "MATCH (p:Paper)-[:MENTIONS]->(c:CONCEPT {name: 'entropy'}) RETURN p.title LIMIT 5"

# Cross-domain hypothesis generation
$CLI hypothesis

# Bridge two concepts
$CLI crossref "diffusion" "gene_expression"

# Web search (Brave)
$CLI web "Visium HD colorectal cancer 2025"
```

---

## 5. Inner Workings

### 5.1 Document Ingestion Pipeline

```
PDF → Parse → Chunk → Embed → Index → Link
```

**Step 1: Parse**
- Triple-fallback: PyMuPDF → pdfplumber → OCR (Tesseract)
- Page-local coordinates preserved for citations
- Deterministic doc_id via UUIDv5(SHA256 of content)

**Step 2: Chunk**
- Semantic chunking by paragraph/section
- ~500 token chunks with 50 token overlap
- Code: AST-aware chunking (function/class boundaries)

**Step 3: Embed**
- Model: `sentence-transformers/all-MiniLM-L6-v2` (384 dims)
- Batch processing: 147 embeddings/sec on RTX 5090
- Stored in ChromaDB with metadata filters

**Step 4: Index**
- Postgres: Full record with FTS tsvector
- ChromaDB: Embedding + metadata (year, concepts, doi)
- Neo4j: Concept extraction and linking

**Step 5: Link**
- Pattern matching for 35+ cross-domain concepts
- CO_OCCURS edges between concepts in same document
- CITES edges from reference extraction

### 5.2 Concept Extraction

Regex patterns detect cross-domain concepts:

```python
CONCEPT_PATTERNS = {
    # Signal Processing
    "compressed_sensing": r"compress(?:ed|ive)?\s*sens(?:ing|ed)",
    "wavelet": r"wavelet",
    "fourier": r"fourier",

    # Physics
    "entropy": r"entropy",
    "free_energy": r"free\s*energy",
    "diffusion": r"diffusion",

    # Causality
    "causal_inference": r"causal\s*inference",
    "counterfactual": r"counterfactual",

    # Systems
    "feedback": r"feedback\s*(?:loop|control|system)",
    "cybernetics": r"cybernetic",
    "emergence": r"emergenc(?:e|t)",

    # Cognitive
    "predictive_coding": r"predictive\s*coding",
    "active_inference": r"active\s*inference",

    # ML/AI
    "transformer": r"transformer",
    "attention": r"attention\s*mechanism",
    "graph_neural_network": r"graph\s*(?:neural\s*)?network|GNN",

    # Biology
    "spatial_transcriptomics": r"spatial\s*transcript",
    "single_cell": r"single[\s-]*cell",
    "gene_expression": r"gene\s*expression",
}
```

### 5.3 Novelty Scoring Algorithm

Hypothesis novelty is computed from corpus statistics:

```python
def compute_novelty(source_concepts, target_domain, corpus_stats):
    # 1. Domain distance (0.0-0.4)
    # Cross-domain bridges score higher
    domain_distance = 1 - jaccard(source_domain, target_domain)

    # 2. Corpus coverage (0.0-0.4)
    # Fewer existing papers = more novel
    existing_papers = count_papers_with_concepts(source + target)
    coverage_score = 1 - log(existing_papers + 1) / log(max_papers)

    # 3. Combination specificity (0.0-0.2)
    # Rare source concepts add novelty
    rarity = 1 - avg(concept_frequency(c) for c in source_concepts)

    return 0.4 * domain_distance + 0.4 * coverage_score + 0.2 * rarity
```

**Range**: [0.30, 0.95] — no hypothesis is 0% or 100% novel

### 5.4 Evidence Extraction Pipeline

For claim verification:

```python
def verify_claim(claim, corpus):
    # 1. Retrieve candidate passages
    candidates = hybrid_search(claim, n=50)

    # 2. Batched NLI scoring
    pairs = [(claim, passage.text) for passage in candidates]
    scores = nli_model.predict(pairs)  # [entailment, neutral, contradiction]

    # 3. Filter by entailment threshold
    evidence = [
        (passage, score)
        for passage, score in zip(candidates, scores)
        if score['entailment'] > 0.7 and passage.page_num >= 0
    ]

    # 4. Build citations with page references
    return format_citations(evidence)
```

---

## 6. Usage Examples

### 6.1 Literature Review: Spatial Transcriptomics Methods

```bash
# Find foundational papers
python3 -c "
from lib.hybrid_search_v2 import HybridSearcherV2
hs = HybridSearcherV2()
results = hs.search_papers('spatial transcriptomics methodology', n=15)
for r in results:
    print(f'{r.title[:80]}... ({r.year})')
"

# Find implementations
python3 -c "
from lib.hybrid_search_v2 import HybridSearcherV2
hs = HybridSearcherV2()
results = hs.search_code('spatial transcriptomics', n=10, org_filter='theislab')
for r in results:
    print(f'{r.repo_name}/{r.name}')
"
```

### 6.2 Cross-Domain Hypothesis Generation

```python
# Via MCP tool
mcp__polymath-v11__generate_hypothesis(
    research_area="predicting gene expression from H&E images"
)

# Returns hypotheses like:
# "Apply compressed sensing theory to sparse spatial transcriptomics data"
# Novelty: 0.72, Feasibility: 0.65, Testability: 0.80
```

### 6.3 Finding Analogous Solutions

```python
# Problem: "How to impute missing spots in spatial transcriptomics data?"
mcp__polymath-v11__find_analogy(
    problem="impute missing values in spatially structured data"
)

# Returns analogies from:
# - Image inpainting (computer vision)
# - Kriging (geostatistics)
# - Matrix completion (compressed sensing)
# - Diffusion models (generative AI)
```

### 6.4 Gap Detection for Grant Writing

```python
mcp__polymath-v11__find_gaps(topic="colibactin colorectal cancer")

# Returns:
# - Time gap: No papers since 2023 on X
# - Method gap: Deep learning not applied to Y
# - Orphan concept: Z mentioned but never linked to W
```

### 6.5 Building a Paper's Background Section

```python
from lib.pqe_response_generator import PQEResponseGenerator

pqe = PQEResponseGenerator()
response = pqe.generate(
    question="What is the current state of H&E to spatial transcriptomics prediction?"
)

# Returns verified claims with page-level citations:
# "UNI achieves 0.87 correlation on Visium data [Chen et al. 2024, p.5]"
```

---

## 7. Future Directions

### 7.1 Active Inference Framework (Planned)

Replace reactive search with **epistemic foraging**:

```
Current: User asks → System retrieves
Future:  System predicts what user needs → Proactive suggestions
```

Based on Karl Friston's Free Energy Principle:
- Model the researcher's beliefs and goals
- Minimize expected free energy by suggesting high-value reads
- Balance exploration (new domains) vs. exploitation (depth in known areas)

### 7.2 Visual Knowledge Graph (Planned)

Interactive 3D visualization of the concept graph:
- Force-directed layout showing concept clusters
- Click to explore papers at each node
- Time slider to see field evolution
- Highlight cross-domain bridges

### 7.3 Code Synthesis (Planned)

From paper methods sections to runnable code:

```
Paper: "We applied UMAP for dimensionality reduction followed by Leiden clustering"
       ↓ (Code synthesis)
Generated:
    import scanpy as sc
    sc.pp.neighbors(adata, n_neighbors=15)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=1.0)
```

### 7.4 Multi-Agent Research Teams (Planned)

Specialized agents collaborating on complex tasks:
- **Librarian**: Deep literature search
- **Critic**: Hypothesis validation
- **Synthesizer**: Writing and summarization
- **Experimenter**: Code generation and testing

### 7.5 Expansion Targets

**High-priority domains to add**:
- Category theory for science (structure-preserving maps)
- Topological data analysis classics
- Causal discovery algorithms
- Active learning / experimental design
- Biosemiotics and biological information

**Code repositories to ingest**:
- 78 paper-referenced repos in `/home/user/work/polymax/data/repos_to_clone.txt`
- Additional: POT (optimal transport), GUDHI (TDA), DoWhy (causal)

### 7.6 External API Integrations (Planned)

| API | Purpose | Status |
|-----|---------|--------|
| GEO/SRA | Dataset discovery | Alpha |
| PubMed/NCBI | Author tracking | Planned |
| Semantic Scholar | Citation graphs | Planned |
| OpenAlex | Affiliation data | Planned |
| Zotero | User library sync | Planned |

---

## 8. Academic Playbook

For academic usage patterns and optimization guidance, see:

- `docs/ACADEMIC_PLAYBOOK.md`

This playbook covers evidence-bound writing, systematic reviews,
method replication, cross-domain hypothesis generation, and QA.


## 9. Hackathon Playbook

For the spatial multimodal hackathon workflow, see:

- `docs/HACKATHON_PLAYBOOK.md`

This playbook includes a coverage audit, gap detection, and rapid ingestion
steps tailored for competitive hackathon use.

## Appendix A: File Locations

| Component | Path |
|-----------|------|
| System home (code) | `/home/user/polymath-repo/` |
| ChromaDB | `/home/user/polymath-repo/chromadb/` |
| Hybrid search | `/home/user/polymath-repo/lib/hybrid_search_v2.py` |
| MCP server | `/home/user/polymath-repo/mcp/polymath_v11/server.py` |
| Literature Sentry | `/home/user/polymath-repo/lib/sentry/` |
| Evidence system | `/home/user/polymath-repo/lib/evidence_extractor.py` |
| GitHub repos | `/home/user/work/polymax/data/github_repos/` |
| Ingestion staging | `/home/user/work/polymax/ingest_staging/` |
| Skills | `/home/user/polymath-repo/skills/` |

## Appendix B: Database Connections

| Database | Connection |
|----------|------------|
| PostgreSQL | `psql -U polymath -d polymath` |
| Neo4j | `bolt://localhost:7687` (neo4j/polymathic2026) |
| ChromaDB | `chromadb.PersistentClient('/home/user/polymath-repo/chromadb')` |

## Appendix C: Environment Variables

```bash
POLYMATH_ROOT=/home/user/polymath-repo
NEO4J_PASSWORD=polymathic2026
BRAVE_API_KEY=<your_key>
POSTGRES_DSN=dbname=polymath user=polymath host=/var/run/postgresql
CHROMADB_PATH=/home/user/polymath-repo/chromadb
CHROMA_PATH=/home/user/polymath-repo/chromadb
EMBEDDING_MODEL=BAAI/bge-m3
PAPERS_COLLECTION=polymath_bge_m3
CODE_COLLECTION=polymath_code_bge_m3

# Load all:
set -a && source /home/user/polymath-repo/.env && set +a
```

---

## Changelog

### v1.0 (2026-01-06)
- Initial release
- Neo4j fully synced (31,867 papers)
- 21 integration tests passing
- Evidence-bound citation system production-ready

---

*"The test of a first-rate intelligence is the ability to hold two opposed ideas in mind at the same time and still retain the ability to function."* — F. Scott Fitzgerald

Polymath is built for minds that refuse to be constrained by disciplinary boundaries.
