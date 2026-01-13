# Polymath 2.0: Cross-Domain Knowledge Discovery Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: In Development](https://img.shields.io/badge/Status-In%20Development-blue.svg)]()

**Author**: Max Van Belkum
**Institution**: Vanderbilt University MD-PhD Program
**Contact**: max.van.belkum@vanderbilt.edu

---

## Executive Summary

Polymath 2.0 is a **ground-up redesign** of a knowledge management and discovery platform designed to enable **cross-domain scientific insight**. The platform integrates scientific literature, code repositories, and structured knowledge graphs to facilitate:

1. **Literature Management** - Zotero-integrated workflow with full metadata integrity
2. **Knowledge Retrieval** - Hierarchical passage storage with context-aware search
3. **Cross-Domain Discovery** - Mechanism-based transfer learning between fields
4. **Hypothesis Generation** - Actionable research proposals with validation
5. **Code-Literature Integration** - Unified semantic space linking papers to implementations

### The Problem We're Solving

Current knowledge management systems capture **labels** (what things are called) but not **mechanisms** (how things work). This makes cross-domain discovery nearly impossible because:

- "Optimal transport" in geospatial analysis and "optimal transport" in biology share the SAME mechanism
- But label-based systems can't reason about WHY the transfer makes sense
- Result: Either garbage hypotheses or missed opportunities

### Our Solution

Polymath 2.0 introduces a **mechanism-centric knowledge graph** that captures:

```
METHOD → MECHANISM → DATA_STRUCTURE → DOMAIN
```

This enables queries like:
> "Find methods from ANY domain that use distributional matching on point clouds,
> which haven't been applied to spatial transcriptomics yet"

---

## Platform Goals

### Primary Goals

| Goal | Description | Success Metric |
|------|-------------|----------------|
| **Metadata Integrity** | Every document has DOI, authors, venue, year | 100% DOI coverage |
| **Contextual Retrieval** | Passages maintain document context | 1500+ char avg passage |
| **Mechanism Extraction** | Capture HOW methods work, not just names | 80%+ extraction accuracy |
| **Grounded Reasoning** | Every claim cites specific evidence | 95% citation accuracy |
| **Cross-Domain Transfer** | Valid mechanism-based hypothesis generation | 50+ actionable hypotheses |
| **Code Integration** | Link papers to implementations | 1000+ code-paper links |

### Secondary Goals

- **Audit Trail**: Every query logged with reasoning trace
- **Contradiction Detection**: Flag conflicting evidence across papers
- **Gap Identification**: Automatically identify under-explored research areas
- **Reproducibility**: All workflows documented for independent verification

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        POLYMATH 2.0 ARCHITECTURE                         │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────┐                              ┌──────────────┐
│    ZOTERO    │  ─── CSV Export ───────────▶│   INGEST     │
│  (PDFs +     │                              │   PIPELINE   │
│  Metadata)   │                              └──────┬───────┘
└──────────────┘                                     │
                                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      PRIMARY DATA STORES                                 │
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐│
│  │  PostgreSQL  │  │   ChromaDB   │  │    Neo4j     │  │  Code Index  ││
│  │   (SoR)      │  │  (Vectors)   │  │   (Graph)    │  │  (Repos)     ││
│  │              │  │              │  │              │  │              ││
│  │ - Documents  │  │ - BGE-M3     │  │ - Methods    │  │ - Functions  ││
│  │ - Passages   │  │ - 1024-dim   │  │ - Mechanisms │  │ - Classes    ││
│  │ - Metadata   │  │ - HNSW idx   │  │ - Problems   │  │ - Modules    ││
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘│
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        QUERY LAYER                                       │
│                                                                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐        │
│  │   Query    │─▶│  Retrieve  │─▶│  Validate  │─▶│ Synthesize │        │
│  │ Decompose  │  │ + Expand   │  │ + Ground   │  │  + Cite    │        │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    CROSS-DOMAIN DISCOVERY ENGINE                         │
│                                                                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐        │
│  │    Gap     │─▶│ Mechanism  │─▶│  Transfer  │─▶│ Hypothesis │        │
│  │ Detection  │  │  Matching  │  │ Validation │  │ Generation │        │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
polymath-2.0/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── docs/
│   ├── ARCHITECTURE.md          # Detailed system architecture
│   ├── BUILDING.md              # Build instructions
│   ├── USING.md                 # User guide
│   ├── MAINTAINING.md           # Maintenance and growth guide
│   ├── CORPUS.md                # Existing corpus and expansion plans
│   └── ROADMAP.md               # Implementation timeline
├── designs/
│   ├── ui_search.md             # Search interface mockups
│   ├── ui_discovery.md          # Discovery workflow mockups
│   ├── ui_ingest.md             # Ingestion interface mockups
│   └── ui_audit.md              # Audit/admin interface mockups
├── schemas/
│   ├── postgres_schema.sql      # PostgreSQL schema definitions
│   ├── neo4j_schema.cypher      # Neo4j graph schema
│   └── api_schema.yaml          # API specification (OpenAPI)
├── scripts/
│   ├── ingest/                  # Ingestion pipeline scripts
│   ├── extract/                 # Concept/mechanism extraction
│   ├── search/                  # Search and retrieval
│   ├── discovery/               # Cross-domain discovery
│   └── maintenance/             # Maintenance utilities
├── tests/
│   └── ...                      # Test suites
└── GROUND_UP_REDESIGN.md        # Full redesign specification
```

---

## Current Corpus (Polymath 1.0)

The existing corpus contains substantial indexed content:

| Store | Content | Count |
|-------|---------|-------|
| PostgreSQL | Passages | 748,000 |
| PostgreSQL | Code chunks | 575,000 |
| ChromaDB | Paper embeddings | 750,000 |
| Neo4j | Concepts | 765,000 |
| Neo4j | Concept relationships | 1,200,000 |
| Archive | PDF files | 3,158 |
| Archive | Code repositories | 243 |

### Corpus Domains

- **Primary**: Spatial transcriptomics, computational pathology, single-cell genomics
- **Supporting**: Machine learning, computer vision, statistics, optimization
- **Cross-domain**: Geospatial analysis, signal processing, operations research

See [docs/CORPUS.md](docs/CORPUS.md) for detailed corpus statistics and expansion plans.

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-4)
- Zotero export pipeline
- Hierarchical passage schema
- Context-aware retrieval

### Phase 2: Mechanism Extraction (Weeks 5-8)
- Mechanism/data structure/objective extraction
- Graph schema implementation
- Quality validation

### Phase 3: Agentic Layer (Weeks 9-12)
- Query decomposition
- Evidence validation
- Citation integration

### Phase 4: Discovery Engine (Weeks 13-16)
- Gap detection
- Transfer validation
- Hypothesis generation

### Phase 5: Code Integration (Weeks 17-20)
- Code concept extraction
- Unified semantic space
- Cross-modal retrieval

See [docs/ROADMAP.md](docs/ROADMAP.md) for detailed timeline.

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/vanbelkummax/polymath-2.0.git
cd polymath-2.0

# Install dependencies
pip install -r requirements.txt

# Initialize databases
./scripts/maintenance/init_databases.sh

# Run initial ingest (requires Zotero export)
python scripts/ingest/from_zotero.py --csv /path/to/zotero_export.csv

# Start the CLI
python polymath_cli.py
```

---

## For Auditors

This repository is designed for systematic audit by coding experts. Key resources:

1. **[GROUND_UP_REDESIGN.md](GROUND_UP_REDESIGN.md)** - Complete architectural specification
2. **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Technical architecture with rationale
3. **[schemas/](schemas/)** - All database schemas with comments
4. **[scripts/](scripts/)** - Implementation code with docstrings
5. **[tests/](tests/)** - Test coverage for all components

### Audit Checklist

- [ ] Architecture review (docs/ARCHITECTURE.md)
- [ ] Schema validation (schemas/)
- [ ] Code quality assessment (scripts/)
- [ ] Test coverage verification (tests/)
- [ ] Security review (authentication, data handling)
- [ ] Performance analysis (indexing, query optimization)

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## Acknowledgments

This project builds on experience from the Polymath 1.0 system and incorporates insights from:
- CogFlow cognitive architecture patterns
- HuBMAP multi-scale data integration approaches
- Modern knowledge graph design principles
