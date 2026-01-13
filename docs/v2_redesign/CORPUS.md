# Polymath 2.0 Corpus Documentation

**Author**: Max Van Belkum
**Last Updated**: 2026-01-13

---

## Current Corpus Statistics (from Polymath 1.0)

### Document Stores

| Store | Type | Count | Notes |
|-------|------|-------|-------|
| PostgreSQL - passages | Text chunks | 748,000 | Avg 500 chars (too short) |
| PostgreSQL - code_chunks | Code snippets | 575,000 | From 243 repositories |
| PostgreSQL - documents | Paper records | ~12,000 | 80% missing DOI |
| ChromaDB | Vector embeddings | 750,000 | BGE-M3 1024-dim |
| Neo4j - concepts | Concept nodes | 765,000 | No mechanism layer |
| Neo4j - relationships | Concept edges | 1,200,000 | Label-based similarity |
| File archive - PDFs | PDF files | 3,158 | /scratch/polymath_archive/ |
| File archive - repos | Git repositories | 243 | Mixed quality |

### Concept Type Distribution

```
┌────────────────────────────────────────────────────────────────┐
│                   CONCEPT TYPES (4.8M total)                    │
├────────────────────────────────────────────────────────────────┤
│ domain      ████████████████████████████████████████  2,400,000 │
│ method      █████████████████                           544,000 │
│ technique   ████████████████                            420,000 │
│ entity      ███████████████                             380,000 │
│ metric      ██████████                                  250,000 │
│ tool        ████████                                    200,000 │
│ dataset     ██████                                      150,000 │
│ model       █████                                       125,000 │
│ other       ████████████████████                        331,000 │
│ mechanism   [NOT EXTRACTED]                                   0 │
└────────────────────────────────────────────────────────────────┘
```

**Critical Gap**: No "mechanism" concept type exists. This is the root cause of BridgeMine's garbage hypotheses.

### Passage Length Distribution

```
┌────────────────────────────────────────────────────────────────┐
│                  PASSAGE LENGTHS (748K total)                   │
├────────────────────────────────────────────────────────────────┤
│ 0-200 chars      ██████████                            100,000  │
│ 200-500 chars    ██████████████████████████            221,000  │
│ 500-1000 chars   ██████████████████████████████████████ 497,000 │
│ 1000-1500 chars  █                                          35  │
│ 1500+ chars      ████                                    4,000  │
└────────────────────────────────────────────────────────────────┘

Target for v2.0: 1500-3000 chars (section-level passages)
```

### Metadata Completeness

| Field | Coverage | Notes |
|-------|----------|-------|
| title | 100% | Required for ingestion |
| title_hash | 100% | Computed from title |
| doc_id | 100% | UUID generated |
| doi | 15% | **CRITICAL GAP** |
| pmid | <1% | Rarely captured |
| authors | 60% | Often incomplete |
| year | 70% | From filename or text |
| venue | 40% | Inconsistent format |
| abstract | 50% | PDF parsing dependent |

**Root Cause**: PDF-first workflow bypasses metadata sources. Zotero-first fixes this.

---

## Corpus Domains

### Primary Domains (Research Focus)

| Domain | Papers | Passages | Priority |
|--------|--------|----------|----------|
| Spatial Transcriptomics | ~800 | ~65,000 | Highest |
| Computational Pathology | ~600 | ~48,000 | Highest |
| Single-cell Genomics | ~500 | ~40,000 | High |
| Deep Learning (Vision) | ~400 | ~32,000 | High |
| Image Analysis | ~350 | ~28,000 | Medium |

### Supporting Domains

| Domain | Papers | Passages | Value |
|--------|--------|----------|-------|
| Statistics/Optimization | ~200 | ~16,000 | Methods transfer |
| Graph Neural Networks | ~150 | ~12,000 | Spatial modeling |
| Computer Vision | ~300 | ~24,000 | Feature extraction |
| Bioinformatics | ~250 | ~20,000 | Pipeline methods |
| Natural Language Processing | ~100 | ~8,000 | Text processing |

### Cross-Domain Sources (Underexplored)

| Domain | Papers | Potential |
|--------|--------|-----------|
| Geospatial Analysis | ~50 | High - spatial methods |
| Operations Research | ~30 | High - optimization |
| Signal Processing | ~40 | Medium - denoising |
| Materials Science | ~20 | Medium - texture analysis |
| Remote Sensing | ~25 | High - multi-resolution |

---

## Code Repository Inventory

### Indexed Repositories (243 total)

**High-Quality (Well-documented, tests)**
```
├── spatial-omics/
│   ├── squidpy (indexed)
│   ├── spatialdata (indexed)
│   ├── scanpy (indexed)
│   └── anndata (indexed)
├── pathology/
│   ├── CLAM (indexed)
│   ├── HIPT (indexed)
│   └── hover_net (indexed)
└── ml-frameworks/
    ├── pytorch-geometric (partial)
    └── huggingface-transformers (partial)
```

**Priority Unindexed (High Value)**
```
├── spatial-prediction/
│   ├── Img2ST           # Core to research
│   ├── HisToGene        # Comparison baseline
│   ├── STAGATE          # Graph methods
│   ├── SpaGCN           # Spatial clustering
│   └── hist2st-benchmark
├── foundation-models/
│   ├── UNI              # Pathology foundation
│   ├── CONCH            # Vision-language
│   ├── scGPT            # Single-cell
│   └── Geneformer       # Gene expression
└── preprocessing/
    ├── stardist         # Segmentation
    └── cellpose         # Cell detection
```

### Code Quality Assessment

| Quality Tier | Repos | Characteristics |
|--------------|-------|-----------------|
| Tier 1 (Excellent) | 45 | Docs, tests, maintained |
| Tier 2 (Good) | 80 | Some docs, working |
| Tier 3 (Fair) | 70 | Minimal docs, functional |
| Tier 4 (Poor) | 48 | No docs, may not run |

---

## Expansion Plans

### Phase 1: Metadata Enrichment (Weeks 1-2)

**Goal**: Fix metadata gaps in existing corpus

| Task | Target | Method |
|------|--------|--------|
| DOI resolution | 100% coverage | CrossRef API |
| PMID lookup | 70% coverage | PubMed API |
| Author normalization | Consistent format | OpenAlex |
| Venue standardization | Controlled vocab | Manual + API |

**Process**:
```
For each document without DOI:
1. Search CrossRef by title
2. Validate match (Jaccard > 0.9)
3. Update document record
4. Log for manual review if uncertain
```

### Phase 2: Zotero Migration (Weeks 3-4)

**Goal**: Move all PDFs to Zotero, re-export with clean metadata

| Task | Volume | Effort |
|------|--------|--------|
| Import existing PDFs to Zotero | 3,158 | Semi-automated |
| Validate/fix metadata | 3,158 | Manual review |
| Export CSV + PDF paths | 1x | Automated |
| Re-ingest with hierarchy | 3,158 | Automated |

### Phase 3: Domain Expansion (Weeks 5-8)

**Target Domains for Cross-Domain Discovery**:

| Domain | Target Papers | Sources | Priority |
|--------|---------------|---------|----------|
| Geospatial Analysis | 200 | ACM DL, IEEE | High |
| Operations Research | 150 | INFORMS, OR journals | High |
| Signal Processing | 100 | IEEE SP, ICASSP | Medium |
| Remote Sensing | 100 | ISPRS, RSE | Medium |
| Materials Science | 50 | Acta Mat, Comp Mat | Low |

**Selection Criteria**:
1. Methods applicable to spatial data
2. Optimization techniques
3. Multi-scale analysis
4. Point cloud / distribution methods

### Phase 4: Code Repository Expansion (Weeks 9-12)

**Priority Queue**:

```python
priority_repos = [
    # Spatial transcriptomics (core)
    ("Img2ST", "https://github.com/...", "highest"),
    ("HisToGene", "https://github.com/...", "high"),
    ("STAGATE", "https://github.com/...", "high"),
    ("SpaGCN", "https://github.com/...", "high"),

    # Foundation models
    ("UNI", "https://github.com/...", "high"),
    ("CONCH", "https://github.com/...", "high"),
    ("scGPT", "https://github.com/...", "medium"),
    ("Geneformer", "https://github.com/...", "medium"),

    # Methods libraries
    ("POT", "https://github.com/...", "high"),  # Optimal transport
    ("geomstats", "https://github.com/...", "medium"),
    ("pytorch-geometric", "https://github.com/...", "medium"),
]
```

---

## Quality Metrics

### Current Quality Issues

| Issue | Prevalence | Impact | Fix |
|-------|------------|--------|-----|
| Missing DOI | 80% | Cannot cite | Zotero-first |
| Short passages | 70% | Lost context | Re-chunk |
| OCR errors | 15% | Bad text | Enhanced parser |
| Duplicate papers | 5% | Inflated counts | Deduplication |
| Incomplete authors | 40% | Bad attribution | API enrichment |

### Target Quality Metrics (v2.0)

| Metric | Current | Target | Validation |
|--------|---------|--------|------------|
| DOI coverage | 15% | 100% | CrossRef check |
| Passage context | 500 chars | 1500 chars | Length analysis |
| OCR quality | 85% | 98% | Spell check |
| Author completeness | 60% | 95% | ORCID lookup |
| Mechanism extraction | 0% | 80% | Manual sample |

---

## Maintenance Procedures

### Daily

- Monitor ingestion queue
- Check for failed jobs
- Review error logs

### Weekly

- Run deduplication check
- Update corpus statistics
- Verify index consistency

### Monthly

- Full backup to archive
- Quality audit (sample 100 papers)
- Update domain statistics
- Review expansion priorities

### Quarterly

- Major corpus expansion
- Re-index with improved extractors
- Performance benchmarking
- User feedback review

---

## Appendix: Corpus Sources

### Academic Sources

| Source | Type | Access | Papers |
|--------|------|--------|--------|
| PubMed Central | Open access | API | ~2,000 |
| bioRxiv | Preprints | API | ~500 |
| arXiv | Preprints | API | ~300 |
| Semantic Scholar | Metadata | API | All |
| Unpaywall | PDF access | API | ~1,500 |

### Institutional Sources

| Source | Type | Papers |
|--------|------|--------|
| Vanderbilt Libraries | Licensed | ~200 |
| Author copies | Direct | ~100 |
| Conference proceedings | Purchased | ~50 |

### Code Sources

| Source | Type | Repos |
|--------|------|-------|
| GitHub | Public | 230 |
| GitLab | Public | 8 |
| Bitbucket | Public | 5 |
