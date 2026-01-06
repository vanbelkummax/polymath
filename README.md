# Polymath

**A Polymathic Research Intelligence System**

Polymath is a self-growing research intelligence system designed for polymathic discovery—finding unexpected connections across disparate scientific domains. Unlike traditional literature databases that organize knowledge in silos, Polymath explicitly models **cross-domain bridges**: the fertile intersections where spatial transcriptomics meets information theory, where thermodynamics informs deep learning, where cybernetics illuminates biological systems.

## Key Features

- **807K+ Searchable Items**: 395K paper passages + 411K code chunks
- **Cross-Domain Concept Graph**: 222 concepts linked across 31K+ papers
- **Hybrid Search**: Combines semantic (ChromaDB), lexical (Postgres FTS), and graph (Neo4j) search
- **Evidence-Bound Citations**: Claims traced to page-level source text
- **Self-Improving**: Identifies coverage gaps and proposes expansions

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    USER QUERY                        │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│           HYBRID SEARCH (RRF Fusion)                │
└─────────────────────┬───────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
┌─────────────┐ ┌──────────┐ ┌────────────┐
│  ChromaDB   │ │ Postgres │ │   Neo4j    │
│  807K vecs  │ │  (FTS)   │ │  (graph)   │
│  2 colls    │ │  SoR     │ │ 32K nodes  │
└─────────────┘ └──────────┘ └────────────┘
```

## Installation

### Prerequisites

- Python 3.10+
- PostgreSQL 14+
- Neo4j 5.x
- CUDA-capable GPU (recommended for embeddings)

### Setup

```bash
# Clone the repository
git clone https://github.com/vanbelkummax/polymath.git
cd polymath

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .

# Copy and configure environment
cp .env.example .env
# Edit .env with your database credentials

# Initialize databases
python scripts/init_databases.py
```

### Database Setup

#### PostgreSQL
```bash
createdb polymath
psql -d polymath -f schema/postgres_schema.sql
```

#### Neo4j
```bash
# Start Neo4j and set password
neo4j start
cypher-shell -u neo4j -p neo4j "ALTER CURRENT USER SET PASSWORD FROM 'neo4j' TO 'your_password'"
```

#### ChromaDB
ChromaDB is file-based and will be created automatically on first run.

## Quick Start

### Search Papers
```python
from lib.hybrid_search_v2 import HybridSearcherV2

hs = HybridSearcherV2()

# Semantic search
results = hs.search_papers("attention mechanism in pathology", n=10)

# With filters
results = hs.search_papers(
    "spatial transcriptomics",
    n=10,
    year_min=2023,
    concepts=["transformer"]
)
```

### Search Code
```python
# Find implementations
results = hs.search_code(
    "multiple instance learning",
    n=10,
    org_filter="mahmoodlab",
    language="python"
)
```

### Hybrid Search (Papers + Code)
```python
results = hs.hybrid_search(
    "predict gene expression from H&E",
    n=20,
    rerank=True  # Cross-encoder reranking
)
```

### CLI
```bash
# System stats
python polymath_cli.py stats

# Semantic search
python polymath_cli.py search "compressed sensing spatial transcriptomics" -n 10

# Neo4j graph query
python polymath_cli.py graph "MATCH (p:Paper)-[:MENTIONS]->(c:CONCEPT) RETURN c.name, count(p) ORDER BY count(p) DESC LIMIT 10"

# Generate hypotheses
python polymath_cli.py hypothesis
```

## MCP Server Tools

Polymath exposes tools via the Model Context Protocol (MCP):

### Discovery Tools
| Tool | Description |
|------|-------------|
| `deep_hunt` | Multi-source literature search |
| `find_gaps` | Detect research gaps |
| `trend_radar` | Analyze publication trends |
| `find_datasets` | Hunt for datasets |

### Reasoning Tools
| Tool | Description |
|------|-------------|
| `generate_hypothesis` | Cross-domain hypothesis generation |
| `validate_hypothesis` | Validate claims against corpus |
| `find_analogy` | Find analogous solutions |
| `serendipity` | Surface unexpected connections |

### Self-Improvement Tools
| Tool | Description |
|------|-------------|
| `collection_health` | Assess corpus health |
| `expand_collection` | Propose expansion targets |

## Ingestion

### Papers
```bash
# Single PDF
python lib/unified_ingest.py path/to/paper.pdf

# Batch ingestion
python lib/unified_ingest.py path/to/pdfs/ --move

# With enhanced parsing
python lib/unified_ingest.py paper.pdf --enhanced-parser
```

### Code Repositories
```bash
# Ingest a GitHub repo
python lib/code_ingest.py https://github.com/org/repo

# Batch ingest
python scripts/ingest_github_batch.py
```

## Literature Sentry (Autonomous Curator)

```bash
# Full polymathic sweep
python -m lib.sentry.cli "spatial transcriptomics visium" --max 15

# GitHub only
python -m lib.sentry.cli "attention mechanism" --source github --min-stars 100

# Show paywalled items
python -m lib.sentry.cli --show-tickets
```

## Project Structure

```
polymath/
├── lib/                    # Core libraries
│   ├── hybrid_search_v2.py # Hybrid search engine
│   ├── evidence_extractor.py # NLI-based verification
│   ├── code_ingest.py      # Code repository ingestion
│   ├── unified_ingest.py   # Paper ingestion
│   └── sentry/             # Literature Sentry
├── mcp/                    # MCP servers
│   └── polymath_v11/       # Main MCP server
├── scripts/                # Utility scripts
├── tests/                  # Integration tests
├── docs/                   # Documentation
├── schema/                 # Database schemas
└── skills/                 # Skill definitions
```

## Cross-Domain Concepts

Polymath automatically detects and links these concepts:

**Signal Processing**: compressed_sensing, sparse_coding, wavelet, fourier
**Physics**: entropy, free_energy, thermodynamics, diffusion
**Causality**: causal_inference, counterfactual
**Systems**: feedback, control_theory, emergence, cybernetics
**Cognitive**: predictive_coding, bayesian, active_inference
**ML/AI**: transformer, attention, contrastive_learning, graph_neural_network
**Biology**: spatial_transcriptomics, single_cell, gene_expression

## Testing

```bash
# Run integration tests
pytest tests/test_integration.py -v

# Run all tests
pytest tests/ -v
```

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `POSTGRES_URL` | PostgreSQL connection string |
| `NEO4J_URI` | Neo4j bolt URI |
| `NEO4J_PASSWORD` | Neo4j password |
| `CHROMADB_PATH` | Path to ChromaDB directory |
| `BRAVE_API_KEY` | Brave Search API key (optional) |

## Documentation

- [User Manual](docs/POLYMATH_USER_MANUAL.md) - Comprehensive guide
- [Literature Sentry Design](docs/plans/2026-01-04-literature-sentry-design.md)
- [Code Ingestion Design](docs/plans/2026-01-05-code-repo-ingestion-design.md)

## Author

**Max Van Belkum**
- Vanderbilt University, Huo Lab
- MD-PhD Student
- Research: Spatial transcriptomics, Computational pathology, pks+ E. coli in CRC

## License

MIT License - See [LICENSE](LICENSE) for details.

## Citation

If you use Polymath in your research, please cite:

```bibtex
@software{polymath2026,
  author = {Van Belkum, Max},
  title = {Polymath: A Polymathic Research Intelligence System},
  year = {2026},
  url = {https://github.com/vanbelkummax/polymath}
}
```

---

*"The test of a first-rate intelligence is the ability to hold two opposed ideas in mind at the same time and still retain the ability to function."* — F. Scott Fitzgerald
