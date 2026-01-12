# Next Session: Full-Stack Knowledge Base Completion

**Created**: 2026-01-12
**Context**: Full database audit completed. See `/home/user/polymath-repo/docs/FUTURE_WORK.md` for gap analysis.
**Working Directory**: `/home/user/polymath-repo` (SOURCE OF TRUTH)

---

## Session Goals

Complete two parallel tracks to fill remaining knowledge base gaps:

| Track | Target | Expected Yield |
|-------|--------|----------------|
| **Track A**: Metadata-to-Passages | Convert 50 priority citation nodes to full documents | +10-20K passages |
| **Track B**: Code Repository Indexing | Index 26 unindexed repos | +50-100K code chunks |

---

## Current State

| Metric | Value |
|--------|-------|
| Postgres passages | 765,565 |
| Concepts extracted | 4,868,910 (99.73%) |
| ChromaDB papers | 743,337 |
| ChromaDB code | 564,128 |
| Metadata-only docs | ~27,000 (mostly citation nodes) |
| Repos on disk | 313 |
| Repos indexed | 287 |
| **Repos to index** | **26** |

---

## Hard Constraints / Safety

- **DO NOT** change database schema
- **DO NOT** hardcode any API keys or passwords; load from `/home/user/polymath-repo/.env` via:
  ```bash
  set -a; source /home/user/polymath-repo/.env; set +a
  ```
- **DO NOT** write or use `/mnt/*` for active work (WSL slow). Use `/home/user/...` or `/scratch/...`
- **Maintain idempotency**: reruns should not duplicate docs. Use existing `title_hash`/`doc_id` collision logic in `unified_ingest.py`
- **Log everything**: Every action must be logged to a run folder under `/home/user/work/polymax/reports/` with timestamp
- **Minimize API usage**: Prefer local metadata + OpenAlex/Semantic Scholar lookups before web scraping

---

# TRACK A: Metadata-Only to Full Passage Documents

**Objective**: Convert targeted "metadata-only" documents (citation nodes with 0 passages) into full passage-level documents by acquiring PDFs and running the standard unified ingestion pipeline.

**Target Faculty**: Tae Hyun Hwang, Bennett Landman, Ken Lau, Hirak Sarkar (plus conceptually connected citations)

## PHASE 0 — Create Run Folder + Log Structure

```bash
cd /home/user/polymath-repo
set -a; source .env; set +a

# Create timestamped run folder
RUN_DIR="/home/user/work/polymax/reports/metadata_to_fullpassage_$(date +%Y_%m_%d_%H%M)"
mkdir -p "$RUN_DIR"
echo "Run folder: $RUN_DIR"

# Initialize tracking files
touch "$RUN_DIR/PLAN.md"
touch "$RUN_DIR/TARGETS.csv"
touch "$RUN_DIR/ACQUISITION.csv"
touch "$RUN_DIR/INGEST_LOG.jsonl"
touch "$RUN_DIR/POSTCHECK.md"

# Initialize CSV headers
echo "rank,doc_id,title,year,doi,pmid,matched_reason,score" > "$RUN_DIR/TARGETS.csv"
echo "doc_id,method,url_or_path,success,notes" > "$RUN_DIR/ACQUISITION.csv"
```

## PHASE 1 — Identify Candidate Metadata-Only Docs

### Step 1A: Query all metadata-only documents

```sql
-- Run in psql -U polymath -d polymath
-- Find docs with 0 passages (metadata-only)
\copy (
  SELECT
    d.doc_id,
    d.title,
    d.year,
    d.doi,
    d.pmid,
    array_to_string(d.authors, '; ') as authors,
    d.title_hash,
    d.created_at
  FROM documents d
  LEFT JOIN passages p ON d.doc_id = p.doc_id
  WHERE p.passage_id IS NULL
    AND d.title IS NOT NULL
  ORDER BY d.year DESC NULLS LAST, d.created_at DESC
) TO '/tmp/metadata_only_docs.csv' WITH CSV HEADER;
```

### Step 1B: Filter to faculty-relevant candidates

Filter by ANY of these criteria:
1. **Author match**: Contains (case-insensitive) `hwang`, `landman`, `lau`, `sarkar`
2. **Affiliation**: Vanderbilt signals if stored
3. **Concept proximity**: Within 2 hops of concepts from faculty papers
4. **Citation connectivity**: High-degree neighbors of faculty docs
5. **Title keywords** (fallback): spatial, multimodal, computational pathology, imaging, registration, foundation model, spatial transcriptomics, single-cell, holotomography, MRI, segmentation

```python
# scripts/identify_metadata_targets.py
"""
Identify high-priority metadata-only documents for PDF acquisition.
Usage: python3 scripts/identify_metadata_targets.py --output /path/to/TARGETS.csv --limit 50
"""
import csv
import re
import psycopg2
from pathlib import Path

FACULTY_PATTERNS = [
    r'\bhwang\b', r'\blandman\b', r'\blau\b', r'\bsarkar\b',
    r'\bvanderbilt\b'
]

TOPIC_KEYWORDS = [
    'spatial transcriptomics', 'single-cell', 'computational pathology',
    'holotomography', 'foundation model', 'image registration',
    'MRI', 'brain imaging', 'segmentation', 'colorectal', 'CRC',
    'spatial biology', 'multimodal', 'deep learning pathology',
    'gene expression prediction', 'H&E', 'histology'
]

def score_document(row):
    """Score a metadata-only doc by relevance to target faculty."""
    score = 0
    reasons = []

    authors = (row['authors'] or '').lower()
    title = (row['title'] or '').lower()

    # Author matches (highest priority)
    for pattern in FACULTY_PATTERNS[:4]:  # hwang, landman, lau, sarkar
        if re.search(pattern, authors):
            score += 100
            reasons.append(f"author:{pattern.strip(r'\b')}")

    # Vanderbilt affiliation
    if 'vanderbilt' in authors:
        score += 50
        reasons.append("affiliation:vanderbilt")

    # Topic keyword matches
    for kw in TOPIC_KEYWORDS:
        if kw.lower() in title:
            score += 20
            reasons.append(f"topic:{kw}")

    # Has DOI/PMID (easier to acquire)
    if row['doi']:
        score += 10
        reasons.append("has_doi")
    if row['pmid']:
        score += 10
        reasons.append("has_pmid")

    # Recency bonus
    if row['year'] and int(row['year']) >= 2020:
        score += 5

    return score, '|'.join(reasons) if reasons else 'none'

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int, default=50)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    conn = psycopg2.connect(dbname='polymath', user='polymath')
    cur = conn.cursor()

    # Get metadata-only docs
    cur.execute("""
        SELECT
            d.doc_id::text,
            d.title,
            d.year,
            d.doi,
            d.pmid,
            array_to_string(d.authors, '; ') as authors,
            d.title_hash
        FROM documents d
        LEFT JOIN passages p ON d.doc_id = p.doc_id
        WHERE p.passage_id IS NULL
          AND d.title IS NOT NULL
        ORDER BY d.year DESC NULLS LAST
    """)

    rows = [dict(zip(['doc_id', 'title', 'year', 'doi', 'pmid', 'authors', 'title_hash'], r))
            for r in cur.fetchall()]

    # Score and rank
    scored = []
    for row in rows:
        score, reason = score_document(row)
        if score > 0:
            scored.append({**row, 'score': score, 'matched_reason': reason})

    scored.sort(key=lambda x: -x['score'])
    top = scored[:args.limit]

    print(f"Found {len(scored)} relevant metadata-only docs")
    print(f"Top {len(top)} selected for acquisition")

    if args.dry_run:
        for i, doc in enumerate(top[:10], 1):
            print(f"  {i}. [{doc['score']}] {doc['title'][:60]}...")
        return

    # Write to CSV
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['rank', 'doc_id', 'title', 'year', 'doi', 'pmid', 'matched_reason', 'score'])
        for i, doc in enumerate(top, 1):
            writer.writerow([i, doc['doc_id'], doc['title'], doc['year'],
                           doc['doi'], doc['pmid'], doc['matched_reason'], doc['score']])

    print(f"Written to {args.output}")
    conn.close()

if __name__ == '__main__':
    main()
```

## PHASE 2 — Expand Target Set via Conceptual Citations

Starting from faculty seed set and top metadata-only candidates:
1. Pull high-centrality citation neighbors (top N by degree) from Neo4j
2. Find concept-bridge papers sharing rare concepts with 2+ faculty clusters
3. Add benchmark/dataset/method "keystone" papers

```python
# Add to scripts/identify_metadata_targets.py or create scripts/expand_targets_graph.py

def expand_via_graph(seed_doc_ids, limit=20):
    """Find conceptually connected papers via Neo4j."""
    from neo4j import GraphDatabase
    import os

    driver = GraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", os.getenv("NEO4J_PASSWORD", "polymathic2026"))
    )

    with driver.session() as session:
        # Find papers sharing concepts with seed docs
        result = session.run("""
            MATCH (seed:Paper)-[:HAS_CONCEPT]->(c:Concept)<-[:HAS_CONCEPT]-(neighbor:Paper)
            WHERE seed.doc_id IN $seed_ids
              AND NOT neighbor.doc_id IN $seed_ids
            WITH neighbor, COUNT(DISTINCT c) as shared_concepts
            ORDER BY shared_concepts DESC
            LIMIT $limit
            RETURN neighbor.doc_id as doc_id,
                   neighbor.title as title,
                   shared_concepts
        """, seed_ids=seed_doc_ids, limit=limit)

        return [dict(r) for r in result]

    driver.close()
```

**Cap**: 50 papers total in this run.

**Prefer**:
- DOI/PMID present
- Year >= 2015 unless clearly foundational
- High connectedness to at least one faculty cluster

## PHASE 3 — Acquire PDFs for Targets

For each target, attempt acquisition in this order:

### 3.1 Local Archive Search

```bash
# Search existing PDF archives
find /scratch/polymath_archive/pdfs/ -name "*.pdf" | head -20
find /home/user/vanderbilt_professors_mcp/data/ -name "*.pdf" | head -20
find /home/user/work/polymax/data/ -name "*.pdf" 2>/dev/null | head -20
```

### 3.2 Open Access API Lookups

```python
# scripts/acquire_pdfs_for_targets.py
"""
Acquire PDFs for metadata-only targets via multiple sources.
Usage: python3 scripts/acquire_pdfs_for_targets.py --targets TARGETS.csv --output-dir /path/to/pdfs/
"""
import csv
import os
import requests
import time
from pathlib import Path

def search_openalex(doi=None, pmid=None, title=None):
    """Query OpenAlex for open access PDF URL."""
    base = "https://api.openalex.org/works"

    if doi:
        url = f"{base}/doi:{doi}"
    elif pmid:
        url = f"{base}/pmid:{pmid}"
    else:
        return None

    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            # Check for OA PDF
            if data.get('open_access', {}).get('oa_url'):
                return data['open_access']['oa_url']
            if data.get('best_oa_location', {}).get('pdf_url'):
                return data['best_oa_location']['pdf_url']
    except Exception as e:
        print(f"OpenAlex error: {e}")
    return None

def search_semantic_scholar(doi=None, pmid=None):
    """Query Semantic Scholar for open access PDF."""
    base = "https://api.semanticscholar.org/graph/v1/paper"

    if doi:
        url = f"{base}/DOI:{doi}?fields=openAccessPdf"
    elif pmid:
        url = f"{base}/PMID:{pmid}?fields=openAccessPdf"
    else:
        return None

    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if data.get('openAccessPdf', {}).get('url'):
                return data['openAccessPdf']['url']
    except Exception as e:
        print(f"S2 error: {e}")
    return None

def search_pmc(pmid):
    """Check if paper is in PubMed Central."""
    if not pmid:
        return None
    try:
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?dbfrom=pubmed&db=pmc&id={pmid}&retmode=json"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            links = data.get('linksets', [{}])[0].get('linksetdbs', [])
            for link in links:
                if link.get('dbto') == 'pmc':
                    pmc_ids = link.get('links', [])
                    if pmc_ids:
                        return f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_ids[0]}/pdf/"
    except Exception as e:
        print(f"PMC error: {e}")
    return None

def download_pdf(url, output_path):
    """Download PDF from URL."""
    try:
        r = requests.get(url, timeout=30, headers={'User-Agent': 'Mozilla/5.0'})
        if r.status_code == 200 and 'application/pdf' in r.headers.get('content-type', ''):
            with open(output_path, 'wb') as f:
                f.write(r.content)
            return True
    except Exception as e:
        print(f"Download error: {e}")
    return False

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--targets', required=True, help='TARGETS.csv file')
    parser.add_argument('--output-dir', required=True, help='Directory for downloaded PDFs')
    parser.add_argument('--acquisition-log', required=True, help='ACQUISITION.csv output')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load targets
    with open(args.targets) as f:
        reader = csv.DictReader(f)
        targets = list(reader)

    results = []

    for target in targets:
        doc_id = target['doc_id']
        doi = target.get('doi')
        pmid = target.get('pmid')
        title = target.get('title', '')

        print(f"Processing: {title[:50]}...")

        # Try each source
        pdf_url = None
        method = None

        # 1. OpenAlex
        pdf_url = search_openalex(doi=doi, pmid=pmid)
        if pdf_url:
            method = 'openalex'

        # 2. Semantic Scholar
        if not pdf_url:
            pdf_url = search_semantic_scholar(doi=doi, pmid=pmid)
            if pdf_url:
                method = 'semantic_scholar'

        # 3. PMC
        if not pdf_url and pmid:
            pdf_url = search_pmc(pmid)
            if pdf_url:
                method = 'pmc'

        # Download if found
        success = False
        notes = ''

        if pdf_url and not args.dry_run:
            safe_name = doc_id[:36] + '.pdf'
            output_path = Path(args.output_dir) / safe_name
            success = download_pdf(pdf_url, output_path)
            if success:
                notes = str(output_path)
            else:
                notes = f"download_failed:{pdf_url}"
        elif pdf_url:
            success = True
            notes = f"would_download:{pdf_url}"
        else:
            notes = "no_open_access_found"

        results.append({
            'doc_id': doc_id,
            'method': method or 'none',
            'url_or_path': pdf_url or '',
            'success': 'yes' if success else 'no',
            'notes': notes
        })

        time.sleep(0.5)  # Rate limit

    # Write acquisition log
    with open(args.acquisition_log, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['doc_id', 'method', 'url_or_path', 'success', 'notes'])
        writer.writeheader()
        writer.writerows(results)

    # Summary
    acquired = sum(1 for r in results if r['success'] == 'yes')
    print(f"\nAcquisition complete: {acquired}/{len(targets)} PDFs found")

if __name__ == '__main__':
    main()
```

**PDF Storage**:
```bash
mkdir -p /home/user/work/polymax/ingest_staging/metadata_only_fill/
```

## PHASE 4 — Ingest into Full Passage-Level Documents

```bash
# For each acquired PDF
cd /home/user/polymath-repo
set -a; source .env; set +a

# Ingest all PDFs in the staging folder
for pdf in /home/user/work/polymax/ingest_staging/metadata_only_fill/*.pdf; do
    echo "Ingesting: $pdf"
    python3 lib/unified_ingest.py "$pdf" 2>&1 | tee -a "$RUN_DIR/ingest_raw.log"

    # Log result
    if [ $? -eq 0 ]; then
        echo "{\"file\": \"$pdf\", \"status\": \"success\", \"timestamp\": \"$(date -Iseconds)\"}" >> "$RUN_DIR/INGEST_LOG.jsonl"
    else
        echo "{\"file\": \"$pdf\", \"status\": \"failed\", \"timestamp\": \"$(date -Iseconds)\"}" >> "$RUN_DIR/INGEST_LOG.jsonl"
    fi
done
```

## PHASE 5 — Post-Ingest Validation

```sql
-- Check passage counts for ingested docs
SELECT
    d.doc_id,
    d.title,
    COUNT(p.passage_id) as passages
FROM documents d
JOIN passages p ON d.doc_id = p.doc_id
WHERE d.doc_id IN (
    -- List of doc_ids from TARGETS.csv that were successfully acquired
)
GROUP BY d.doc_id, d.title
ORDER BY passages DESC;
```

```bash
# Verify ChromaDB sync
python3 -c "
import chromadb
c = chromadb.PersistentClient('/home/user/polymath-repo/chromadb')
print(f'Papers: {c.get_collection(\"polymath_bge_m3\").count():,}')
print(f'Code: {c.get_collection(\"polymath_code_bge_m3\").count():,}')
"
```

## PHASE 6 — Output Report

Create `POSTCHECK.md`:
```markdown
# Metadata-to-Passages Run Report

**Run Date**: YYYY-MM-DD HH:MM
**Run Folder**: /home/user/work/polymax/reports/metadata_to_fullpassage_YYYY_MM_DD_HHMM/

## Summary
| Metric | Count |
|--------|-------|
| Targeted | X |
| PDFs Acquired | Y |
| Successfully Ingested | Z |
| Failed | W |

## Acquisition Sources
| Source | Count |
|--------|-------|
| OpenAlex | X |
| Semantic Scholar | Y |
| PMC | Z |
| Local Archive | W |

## Errors & Notes
- [List any errors encountered]

## Unresolved Papers (for next run)
| Title | DOI | Reason |
|-------|-----|--------|
| ... | ... | paywalled |

## Quality Gates
- [ ] No duplicate documents created
- [ ] All ingested targets have passages > 0
- [ ] Searchable via hybrid search
```

---

# TRACK B: Code Repository Indexing

**Objective**: Index 26 high-priority code repositories that are on disk but not in the database.

## Priority 1: Spatial Transcriptomics (DO FIRST)

These are directly relevant to Img2ST research:

```bash
cd /home/user/polymath-repo
set -a; source .env; set +a

# Create code indexing log
CODE_LOG="/home/user/work/polymax/reports/code_indexing_$(date +%Y_%m_%d_%H%M).log"

# Core spatial analysis
echo "=== Indexing squidpy ===" | tee -a "$CODE_LOG"
python3 lib/unified_ingest.py /home/user/work/polymax/data/github_repos/squidpy --type code 2>&1 | tee -a "$CODE_LOG"

echo "=== Indexing spatialdata ===" | tee -a "$CODE_LOG"
python3 lib/unified_ingest.py /home/user/work/polymax/data/github_repos/spatialdata --type code 2>&1 | tee -a "$CODE_LOG"

echo "=== Indexing hist2st-benchmark ===" | tee -a "$CODE_LOG"
python3 lib/unified_ingest.py /home/user/hist2st-benchmark --type code 2>&1 | tee -a "$CODE_LOG"

# H&E to ST methods
echo "=== Indexing HisToGene ===" | tee -a "$CODE_LOG"
python3 lib/unified_ingest.py /home/user/work/polymax/data/github_repos/HisToGene --type code 2>&1 | tee -a "$CODE_LOG"

echo "=== Indexing STAGATE ===" | tee -a "$CODE_LOG"
python3 lib/unified_ingest.py /home/user/work/polymax/data/github_repos/STAGATE --type code 2>&1 | tee -a "$CODE_LOG"

echo "=== Indexing SpaGCN ===" | tee -a "$CODE_LOG"
python3 lib/unified_ingest.py /home/user/work/polymax/data/github_repos/SpaGCN --type code 2>&1 | tee -a "$CODE_LOG"
```

## Priority 2: Foundation Models

```bash
# Mahmood lab
echo "=== Indexing HIPT ===" | tee -a "$CODE_LOG"
python3 lib/unified_ingest.py /home/user/work/comp_path_core/repos/HIPT --type code 2>&1 | tee -a "$CODE_LOG"

echo "=== Indexing CLAM ===" | tee -a "$CODE_LOG"
python3 lib/unified_ingest.py /home/user/work/comp_path_core/repos/CLAM --type code 2>&1 | tee -a "$CODE_LOG"

# Universal models
echo "=== Indexing UNI ===" | tee -a "$CODE_LOG"
python3 lib/unified_ingest.py /home/user/work/polymax/data/github_repos/UNI --type code 2>&1 | tee -a "$CODE_LOG"

echo "=== Indexing CONCH ===" | tee -a "$CODE_LOG"
python3 lib/unified_ingest.py /home/user/work/polymax/data/github_repos/CONCH --type code 2>&1 | tee -a "$CODE_LOG"
```

## Priority 3: Single-Cell Foundation Models

```bash
echo "=== Indexing scGPT ===" | tee -a "$CODE_LOG"
python3 lib/unified_ingest.py /home/user/work/singlecell_foundation/repos/scGPT --type code 2>&1 | tee -a "$CODE_LOG"

echo "=== Indexing Geneformer ===" | tee -a "$CODE_LOG"
python3 lib/unified_ingest.py /home/user/work/singlecell_foundation/repos/Geneformer --type code 2>&1 | tee -a "$CODE_LOG"

echo "=== Indexing scBERT ===" | tee -a "$CODE_LOG"
python3 lib/unified_ingest.py /home/user/work/singlecell_foundation/repos/scBERT --type code 2>&1 | tee -a "$CODE_LOG"
```

## Verification After Indexing

```bash
# Check new totals
python3 -c "
import chromadb
c = chromadb.PersistentClient('/home/user/polymath-repo/chromadb')
print(f'Code chunks: {c.get_collection(\"polymath_code_bge_m3\").count():,}')
"

# Verify specific repos
psql -U polymath -d polymath -c "
SELECT repo_name, COUNT(*) as files
FROM code_files
WHERE repo_name IN ('squidpy', 'spatialdata', 'HIPT', 'scGPT')
GROUP BY repo_name
ORDER BY files DESC;
"
```

---

## Quality Gates (Both Tracks Must Pass)

### Track A (Metadata → Passages)
- [ ] No duplicate documents created (check by title_hash/doi collisions)
- [ ] For each ingested target: `COUNT(passages) > 0`
- [ ] For each ingested target: searchable via hybrid search by title keywords
- [ ] Logs and CSVs written to run folder

### Track B (Code Repos)
- [ ] All 26 repos indexed without errors
- [ ] Code chunk count increased by 50-100K
- [ ] Searchable via `hs.search_code('query')`

---

## Troubleshooting

### If ingest fails with title hash collision:
```python
# The unified_ingest.py already handles this by looking up existing doc_id
# If still failing, check the error message for specific guidance
```

### If ChromaDB errors:
```bash
# Check disk space first
df -h /
# If corrupted, remove UUID dirs - they rebuild from SQLite
ls /home/user/polymath-repo/chromadb/
```

### If API rate limited:
```python
# Add delay between requests
import time
time.sleep(1)  # Increase as needed
```

---

## Reference Documents

- **Future Work**: `/home/user/polymath-repo/docs/FUTURE_WORK.md`
- **Architecture**: `/home/user/polymath-repo/docs/POLYMATH_V2_ARCHITECTURE.md`
- **CLAUDE.md**: `/home/user/CLAUDE.md`

---

**Expected Duration**: 3-4 hours total
**Expected Results**:
- Track A: +10-20K passages from faculty-relevant papers
- Track B: +50-100K code chunks from priority repos
