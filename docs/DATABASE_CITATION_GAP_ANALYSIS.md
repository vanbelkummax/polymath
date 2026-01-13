# Polymath Database: Citation Metadata Gap Analysis

**Date**: 2026-01-12
**Purpose**: Document for external review - seeking solutions for incomplete citation metadata

---

## Executive Summary

We have a research knowledge base with **748K passages** from **3,617 scientific papers**. The system works well for semantic search and concept extraction, but **80.8% of documents lack DOIs**, making proper academic citation difficult.

**The core problem**: Most PDFs were batch-ingested using a parser that extracted text but not bibliographic identifiers (DOI, PMID). We now need to retroactively link these documents to their canonical identifiers.

---

## 1. Database Architecture

### Storage Layer (3 databases, kept in sync)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           POLYMATH KNOWLEDGE BASE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │   PostgreSQL    │    │    ChromaDB     │    │     Neo4j       │         │
│  │  (Master Data)  │───▶│   (Vectors)     │───▶│    (Graph)      │         │
│  ├─────────────────┤    ├─────────────────┤    ├─────────────────┤         │
│  │ documents: 30K  │    │ BGE-M3 1024-dim │    │ Paper: 32K      │         │
│  │ passages: 748K  │    │ embeddings: 1.3M│    │ Concept: 1.2M   │         │
│  │ concepts: 4.8M  │    │                 │    │ relationships:  │         │
│  │ code_chunks:576K│    │                 │    │   5.7M          │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Core Tables

**documents** (metadata about each paper)
```sql
doc_id          UUID PRIMARY KEY
doi             TEXT UNIQUE      -- ⚠️ Only 19.2% populated
pmid            TEXT UNIQUE      -- ⚠️ Only 0.9% populated
title           TEXT NOT NULL    -- ✓ 100% populated
title_hash      TEXT UNIQUE      -- For deduplication
authors         TEXT[]           -- ✓ 98.5% populated
year            INTEGER          -- ✓ 98.7% populated
venue           TEXT
openalex_id     TEXT             -- For API enrichment
```

**passages** (text chunks for semantic search)
```sql
passage_id      UUID PRIMARY KEY
doc_id          UUID REFERENCES documents
passage_text    TEXT NOT NULL
page_num        INTEGER
section         TEXT
quality_score   FLOAT DEFAULT 1.0
```

**passage_concepts** (AI-extracted concepts per passage)
```sql
passage_id      UUID REFERENCES passages
concept_name    TEXT
concept_type    TEXT  -- METHOD, FINDING, ENTITY, etc.
confidence      FLOAT
```

### Intended Purpose

1. **Semantic Search**: Find relevant passages across 3,617 papers using vector similarity
2. **Concept Graph**: Navigate relationships between methods, findings, entities
3. **Citation Support**: Generate proper academic citations for retrieved content
4. **Cross-Domain Discovery**: Find analogies between fields (e.g., "attention mechanisms" in both ML and neuroscience)

---

## 2. The Problem: Missing Identifiers

### Current Metadata Coverage (Documents with Passages)

| Field | Has Value | Missing | Coverage |
|-------|-----------|---------|----------|
| **DOI** | 695 | 2,922 | **19.2%** |
| **PMID** | 33 | 3,584 | **0.9%** |
| Year | 3,570 | 47 | 98.7% |
| Authors | 3,562 | 55 | 98.5% |
| Title | 3,617 | 0 | 100% |

### Root Cause: Parser Version Gap

| Parser | Documents | Has DOI | DOI % | Notes |
|--------|-----------|---------|-------|-------|
| `batch_populate_v1` | 1,893 | 19 | **1.0%** | Bulk import, no DOI extraction |
| `pdfplumber_v1` | 1,051 | 628 | **59.8%** | Newer parser with DOI regex |
| `legacy` | 413 | 45 | 10.9% | Old imports |
| NULL | 224 | 3 | 1.3% | Unknown source |
| `ocr_text_v1` | 28 | 0 | 0.0% | OCR'd scans |
| `text_v1` | 8 | 0 | 0.0% | Plain text |

**52% of the corpus** was ingested with `batch_populate_v1`, which extracted text but ignored DOI metadata embedded in PDFs.

### Why This Matters

Without DOIs, we cannot:
- Generate proper academic citations (APA, Vancouver, etc.)
- Link to canonical versions on publisher sites
- Deduplicate against external databases
- Enable citation network analysis
- Meet academic integrity standards for AI-assisted research

---

## 3. Solutions Attempted

### 3.1 OpenAlex Title Matching (Partial Success)

**Script**: `scripts/enrich_metadata_openalex.py`

**Approach**:
1. Query OpenAlex API with paper title
2. Compare returned titles using SequenceMatcher (threshold: 0.85)
3. If match found, extract DOI/PMID/OpenAlex ID

**Results**:
- ~19% match rate on valid titles
- Added ~100 DOIs
- **Limitation**: Many titles are OCR-corrupted or truncated

**Issues**:
- OCR garbage titles fail validation
- API rate limits (0.2s delay required)
- Some papers not in OpenAlex (preprints, gray literature)

### 3.2 Zotero Library Cross-Reference (Partial Success)

**Script**: `scripts/enrich_from_zotero.py`

**Approach**:
1. Export Zotero library to CSV (3,175 entries)
2. Build prefix-hash index (first 4 words of normalized title)
3. Match against Postgres documents
4. Update DOI/PMID/Year/Authors

**Results**:
- 42 papers enriched (41 new DOIs)
- Identified 1,387 papers in Zotero NOT in Polymath
- **Limitation**: Only matches papers already in personal library

### 3.3 GROBID Re-extraction (Not Yet Attempted)

**Potential Approach**:
1. Re-run GROBID on original PDFs with `consolidateCitations=1`
2. GROBID can extract DOI from PDF metadata, watermarks, headers
3. Update documents table with extracted identifiers

**Blockers**:
- Need to map doc_id back to original PDF files
- Some PDFs may no longer be accessible
- GROBID extraction is slow (~5 sec/paper)

### 3.4 CrossRef Title Search (Not Yet Attempted)

**Potential Approach**:
1. Query CrossRef API with title + first author + year
2. Higher precision than OpenAlex for published papers
3. Rate limit: 50 requests/second with polite pool

---

## 4. Current State

### What Works Well

✅ **Semantic Search**: 748K passages fully searchable via BGE-M3 vectors
✅ **Concept Extraction**: 4.8M concepts linked to passages (99.7% coverage)
✅ **Graph Navigation**: Neo4j enables concept→paper→concept traversal
✅ **Title/Author/Year**: 98%+ coverage for basic metadata

### What's Broken

❌ **DOI Coverage**: Only 19.2% (695 of 3,617 documents)
❌ **PMID Coverage**: Only 0.9% (33 documents)
❌ **Citation Generation**: Cannot produce proper citations for 80% of content
❌ **Deduplication**: Risk of duplicate entries without canonical IDs

### Enrichment Progress

| Date | Action | DOIs Before | DOIs After | Δ |
|------|--------|-------------|------------|---|
| Baseline | - | 577 | - | - |
| 2026-01-12 | OpenAlex enrichment | 577 | 654 | +77 |
| 2026-01-12 | Zotero enrichment | 654 | 695 | +41 |
| **Current** | - | - | **695** | **+118 total** |

---

## 5. Options for Moving Forward

### Option A: Accept Incomplete Citations (Pragmatic)

**Approach**: Use what we have, generate best-effort citations

For papers WITH DOI:
```
Smith et al. (2023). Title here. Journal Name. https://doi.org/10.1234/example
```

For papers WITHOUT DOI:
```
Smith et al. (2023). Title here. [Retrieved from Polymath KB, doc_id: abc123]
```

**Pros**:
- No additional work required
- 98% have title/author/year (usable for informal citation)

**Cons**:
- Not suitable for formal academic publication
- Cannot verify sources externally

### Option B: Aggressive API Enrichment

**Approach**: Query multiple APIs with fuzzy matching

1. **CrossRef** (best for journal articles)
2. **OpenAlex** (broad coverage, includes preprints)
3. **Semantic Scholar** (good for CS/ML papers)
4. **PubMed/NCBI** (biomedical papers)

**Expected yield**: Maybe 40-60% coverage with combined APIs

**Effort**: ~2-4 hours of API calls + validation

### Option C: PDF Re-Processing

**Approach**: Re-extract metadata from original PDFs

1. Map doc_id → original PDF path (if available)
2. Run GROBID with `consolidateCitations=1`
3. Extract DOI from PDF metadata, headers, watermarks
4. Use `pdftotext` + regex as fallback

**Expected yield**: Unknown, depends on PDF quality

**Effort**: Need to locate ~3,000 PDFs, ~4 hours processing

### Option D: Manual Curation (High-Value Subset)

**Approach**: Manually verify DOIs for most-cited/important papers

1. Identify top 500 papers by:
   - Query frequency
   - Concept density
   - Author importance
2. Manual lookup on Google Scholar / CrossRef
3. Update database

**Expected yield**: 100% for curated subset

**Effort**: ~10-20 hours manual work

### Option E: Hybrid Citation Format

**Approach**: Design citation format that degrades gracefully

```python
def generate_citation(doc):
    if doc.doi:
        return f"{doc.authors} ({doc.year}). {doc.title}. {doc.venue}. https://doi.org/{doc.doi}"
    elif doc.pmid:
        return f"{doc.authors} ({doc.year}). {doc.title}. PMID: {doc.pmid}"
    elif doc.title and doc.year:
        return f"{doc.authors} ({doc.year}). {doc.title}. [Unverified - search title on Google Scholar]"
    else:
        return f"[Document {doc.doc_id[:8]} - metadata incomplete]"
```

**Pros**: Always produces something usable
**Cons**: Inconsistent citation quality

---

## 6. Recommendation

**Short-term (this week)**: Implement Option E (graceful degradation) + Option B (API enrichment)

1. Create citation API that handles missing DOIs gracefully
2. Run CrossRef + Semantic Scholar enrichment (estimated +500-1000 DOIs)
3. Accept that some papers will never have DOIs (gray literature, theses, etc.)

**Medium-term (if needed)**: Option C (PDF re-processing) for high-value papers

**The 80/20 insight**: Even with 19% DOI coverage, we have 98% title/author/year coverage. For internal research use, this is often sufficient. DOIs become critical only when publishing or sharing results externally.

---

## 7. Files & Scripts

| File | Purpose |
|------|---------|
| `scripts/enrich_metadata_openalex.py` | OpenAlex title→DOI lookup |
| `scripts/enrich_from_zotero.py` | Zotero CSV cross-reference |
| `scripts/sync_citations_neo4j.py` | Propagate metadata to Neo4j |
| `lib/citation.py` | Citation generation API |
| `lib/doc_identity.py` | Document deduplication logic |

---

## 8. Questions for Reviewers

1. **Is 40-50% DOI coverage acceptable** for a research knowledge base, or should we target higher?

2. **Should we invest in PDF re-processing**, or accept that older imports will have incomplete metadata?

3. **Are there bulk DOI lookup services** we haven't considered? (Unpaywall, BASE, CORE?)

4. **For papers without DOIs** (preprints, theses, reports), what's the best citation format?

5. **Should we flag low-confidence passages** that come from poorly-cited documents?

---

## Appendix: Quick Commands

```bash
# Check current DOI coverage
psql -U polymath -d polymath -c "
  SELECT COUNT(*) FILTER (WHERE doi IS NOT NULL) as has_doi,
         COUNT(*) as total,
         ROUND(100.0 * COUNT(*) FILTER (WHERE doi IS NOT NULL) / COUNT(*), 1) as pct
  FROM documents WHERE doc_id IN (SELECT DISTINCT doc_id FROM passages);"

# Run OpenAlex enrichment
python3 /home/user/polymath-repo/scripts/enrich_metadata_openalex.py --batch 500 --delay 0.2

# Run Zotero enrichment
python3 /home/user/polymath-repo/scripts/enrich_from_zotero.py /path/to/zotero_export.csv

# Check Neo4j sync status
python3 -c "from neo4j import GraphDatabase; d=GraphDatabase.driver('bolt://localhost:7687',auth=('neo4j','polymathic2026')); print(d.execute_query('MATCH (p:Paper) WHERE p.doi IS NOT NULL RETURN count(p)')[0])"
```
