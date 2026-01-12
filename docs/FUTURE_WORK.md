# Polymath Knowledge Base: Future Work & Gap Analysis

**Last Audit**: 2026-01-12
**Status**: Production-ready for search, gaps in indexing coverage

---

## Executive Summary

| Store | Status | Coverage | Gap |
|-------|--------|----------|-----|
| **Postgres** | Complete | 765K passages, 4.87M concepts | 27K metadata-only docs (citation nodes) |
| **ChromaDB** | Good | 743K paper embeddings, 564K code | ~22K passages unsynced |
| **Neo4j** | Complete | 764K passages, 1.2M concepts | Matches Postgres |
| **Code Repos** | 91% | 287 indexed | 26 unindexed (high-value) |

**Bottom Line**: Core system works. Priority is indexing 26 high-value code repos and filling PDF gaps.

---

## 1. Database Gap Analysis

### 1.1 Postgres (Master Data)

```
Total documents:     30,647
├─ With passages:     3,598 (properly ingested)
└─ Metadata-only:    27,049 (citation nodes, failed ingests)

Total passages:      765,565
Total concepts:      4,868,910 (99.73% coverage)
```

**Metadata-Only Documents Breakdown**:
| Category | Count | Action |
|----------|-------|--------|
| No identifiers (citation nodes) | 27,044 | Keep as-is (graph placeholders) |
| Has DOI only | 4 | Could attempt PDF retrieval |
| Has PMID | 1 | Could attempt PDF retrieval |

**Assessment**: The 27K metadata-only documents are mostly citation nodes created during reference extraction. They serve as graph placeholders and don't need full text. Only 5 have identifiers that could be resolved to PDFs.

### 1.2 ChromaDB (Vectors)

```
Papers collection:    743,337
Code collection:      564,128
Postgres passages:    765,565
GAP:                 ~22,000 passages
```

**Issue**: `vector_synced_at` tracking column not consistently used (only 16 docs marked synced).

**Fix**: Run sync verification:
```bash
python3 -c "
from lib.hybrid_search_v2 import verify_chromadb_sync
verify_chromadb_sync()  # Will identify unsynced passages
"
```

### 1.3 Neo4j (Graph)

```
Concepts:        1,181,577
Passages:          764,864 (matches Postgres)
Papers:             32,723
Code:               14,368
Repos:                  65
```

**Status**: Graph is current. Uses incremental MERGE pattern.

---

## 2. Code Repository Gaps (HIGH PRIORITY)

### 2.1 Summary

```
Repos on disk:     313
Repos indexed:     287
Unindexed:          26 (including critical research repos)
```

### 2.2 Priority 1: Spatial Transcriptomics (User's Core Research)

| Repo | Location | Files | Priority |
|------|----------|-------|----------|
| **squidpy** | `/home/user/work/polymax/data/github_repos/squidpy` | 99 | CRITICAL |
| **spatialdata** | `/home/user/work/polymax/data/github_repos/spatialdata` | 106 | CRITICAL |
| **hist2st-benchmark** | `/home/user/hist2st-benchmark` | 23 | CRITICAL |
| **HisToGene** | `/home/user/work/polymax/data/github_repos/HisToGene` | 5 | HIGH |
| **STAGATE** | `/home/user/work/polymax/data/github_repos/STAGATE` | 6 | HIGH |
| **SpaGCN** | `/home/user/work/polymax/data/github_repos/SpaGCN` | 17 | HIGH |
| **spatial_CRC_atlas** | (Ken Lau lab) | - | HIGH |
| **STalign** | - | - | MEDIUM |
| **SEDR** | - | - | MEDIUM |
| **paste/paste2** | - | - | MEDIUM |
| **cell2location** | - | - | MEDIUM |
| **Tangram** | - | - | MEDIUM |

**Command to index**:
```bash
python3 /home/user/polymath-repo/lib/unified_ingest.py /home/user/work/polymax/data/github_repos/squidpy --type code
```

### 2.3 Priority 2: Foundation Models / Pathology

| Repo | Location | Files | Notes |
|------|----------|-------|-------|
| **HIPT** | `/home/user/work/comp_path_core/repos/HIPT` | 45 | Mahmood lab |
| **CLAM** | `/home/user/work/comp_path_core/repos/CLAM` | 33 | Mahmood lab |
| **UNI** | `/home/user/work/polymax/data/github_repos/UNI` | 17 | Foundation model |
| **CONCH** | `/home/user/work/polymax/data/github_repos/CONCH` | 14 | Contrastive |
| **prov-gigapath** | - | - | Microsoft |

### 2.4 Priority 3: Single-Cell Foundation Models

| Repo | Location | Files | Notes |
|------|----------|-------|-------|
| **scGPT** | `/home/user/work/singlecell_foundation/repos/scGPT` | 38 | Critical |
| **Geneformer** | `/home/user/work/singlecell_foundation/repos/Geneformer` | 22 | Critical |
| **scBERT** | `/home/user/work/singlecell_foundation/repos/scBERT` | - | |
| **cellpose** | - | - | Segmentation |
| **cellrank** | - | - | Trajectory |

---

## 3. Resource Acquisition Pipeline

### 3.1 Autonomous Acquisition (Can Do Automatically)

| Source | Method | Script |
|--------|--------|--------|
| arXiv preprints | arxiv API | `scripts/acquire_arxiv.py` (TODO) |
| GitHub repos | git clone | `scripts/acquire_repos.py` (TODO) |
| PubMed Central (open access) | NCBI E-utils | `scripts/acquire_pmc.py` (TODO) |
| Unpaywall (open access PDFs) | Unpaywall API | `scripts/acquire_unpaywall.py` (TODO) |
| bioRxiv/medRxiv | Cold Spring Harbor | `scripts/acquire_biorxiv.py` (TODO) |

### 3.2 Manual Acquisition Required (Paywalled)

Track in: `/home/user/work/polymax/ingest_staging/tracking/PAYWALLED_WISHLIST.md`

**High Priority Papers**:
- López de Prado: "Advances in Financial Machine Learning" (Wiley)
- Hull: "Options, Futures, and Other Derivatives" (Pearson)
- Shreve: "Stochastic Calculus for Finance" (Springer)
- Hwang lab papers behind paywall

**Acquisition Options**:
1. Vanderbilt library access
2. Author preprints on personal websites
3. ResearchGate requests
4. Interlibrary loan

### 3.3 Gap Detection System (TODO)

Build `lib/gap_detector.py`:
```python
def detect_resource_gaps(query: str) -> dict:
    """
    1. Run hybrid search
    2. Extract cited but missing papers
    3. Check if we have PDFs for DOIs/PMIDs
    4. Return acquisition recommendations
    """
    pass

def request_acquisition(resource: dict, method: str = "auto"):
    """
    1. If open access: acquire automatically
    2. If paywalled: add to PAYWALLED_WISHLIST.md
    3. Notify user of what was found/not found
    """
    pass
```

---

## 4. Production Readiness Checklist

### 4.1 Data Completeness
- [x] Passage-level concept extraction (99.73%)
- [x] ChromaDB embeddings (~97%)
- [x] Neo4j graph sync
- [ ] Index 26 priority code repos
- [ ] Verify ChromaDB sync for 22K passages
- [ ] Backfill 5 metadata-only docs with identifiers

### 4.2 System Reliability
- [x] 16-worker Gemini backfill configuration documented
- [x] ChromaDB corruption recovery documented
- [x] Worker stuck state recovery documented
- [ ] Automated health checks (`scripts/health_check.py`)
- [ ] Sync tracking column usage consistency

### 4.3 Search Quality
- [x] Hybrid search (RRF fusion) working
- [x] Reranking with cross-encoder
- [ ] Eval suite for search quality (TODO: `tests/test_search_quality.py`)
- [ ] Coverage testing against known queries

### 4.4 Integration
- [ ] Subagent patterns for multi-DB queries
- [ ] Skill wrappers for local-first search
- [ ] MCP tool composition
- [ ] Self-improvement loop

---

## 5. Immediate Action Items

**Full Session Prompt**: `/home/user/polymath-repo/docs/NEXT_SESSION_FULLSTACK_INDEXING.md`

### Session 1: Full-Stack Indexing (~3-4 hours)

Two parallel tracks:

| Track | Goal | Expected Yield |
|-------|------|----------------|
| **A: Metadata → Passages** | Convert 50 priority citation nodes to full docs | +10-20K passages |
| **B: Code Repos** | Index 26 unindexed repos | +50-100K code chunks |

**Track A** targets papers by: Hwang, Landman, Lau, Sarkar + conceptually connected citations
**Track B** priorities: squidpy, spatialdata, HIPT, CLAM, scGPT, Geneformer, etc.

### Session 2: ChromaDB Sync Verification (~1 hour)
```bash
# Verify all passages are in ChromaDB
python3 scripts/verify_chromadb_sync.py  # TODO: Create this script

# Update sync tracking columns
python3 scripts/update_sync_timestamps.py  # TODO: Create this script
```

### Session 3: Build Gap Detection System (~3 hours)
1. Create `lib/gap_detector.py`
2. Create `scripts/acquire_*.py` for each source
3. Integrate with Literature Sentry

---

## 6. Long-Term Roadmap

### Phase 1: Complete Indexing (1-2 sessions)
- Index all 26 unindexed repos
- Verify ChromaDB sync
- Fix sync tracking columns

### Phase 2: Acquisition Pipeline (2-3 sessions)
- Build autonomous acquisition for open access
- Create paywalled tracking system
- Integrate with Sentry for new paper discovery

### Phase 3: Self-Improvement Loop (3-5 sessions)
- Gap detection on every query
- Auto-acquisition when possible
- User notification for paywalled items
- CLAUDE.md auto-update mechanism

### Phase 4: Advanced Integration (ongoing)
- Subagent orchestration
- Skill-aware search routing
- MCP tool composition
- Agentic research workflows

---

## Appendix: Quick Commands

```bash
# Check system status
python3 /home/user/polymath-repo/polymath_cli.py stats

# Index a repo
python3 /home/user/polymath-repo/lib/unified_ingest.py /path/to/repo --type code

# Index a PDF
python3 /home/user/polymath-repo/lib/unified_ingest.py /path/to/paper.pdf

# Run hybrid search
python3 -c "from lib.hybrid_search_v2 import HybridSearcherV2; hs=HybridSearcherV2(); print(hs.search_papers('query', n=10))"

# Check ChromaDB counts
python3 -c "import chromadb; c=chromadb.PersistentClient('/home/user/polymath-repo/chromadb'); print(f'papers: {c.get_collection(\"polymath_bge_m3\").count():,}, code: {c.get_collection(\"polymath_code_bge_m3\").count():,}')"
```
