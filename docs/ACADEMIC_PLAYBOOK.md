# Polymath Academic Playbook

## Purpose

This playbook captures how to use and optimize the Polymath framework for academic
research. It is designed to turn a large, heterogeneous corpus into citable evidence,
cross-domain synthesis, and reproducible research outputs.

## Academic Uses (What You Can Do)

### 1. Discovery and Scoping
- Rapidly map a new field and identify its dominant subtopics and methods.
- Build a curated reading list with evidence-bound excerpts.
- Surface overlooked or cross-disciplinary connections for novel thesis ideas.

### 2. Systematic and Living Reviews
- Run structured, repeatable searches across papers, code, and the knowledge graph.
- Track changes over time and maintain a living review with new additions.
- Produce gap analyses that show what is well covered vs underexplored.

### 3. Evidence-Bound Writing
- Draft claims that are automatically linked to page-level evidence.
- Reduce citation drift by tethering statements to exact passages.
- Use the verification pipeline to fail-closed on weak or unsupported claims.

### 4. Methods Transfer and Reproducibility
- Find code implementations that correspond directly to specific papers.
- Compare alternative implementations for the same method.
- Extract parameters, datasets, and evaluation protocols for replication.

### 5. Cross-Domain Hypothesis Generation
- Identify concepts that bridge domains (for example, thermodynamics to ML).
- Use analogies to propose experimental designs or new analytical frames.
- Validate novel hypotheses against existing evidence before committing effort.

### 6. Grant and Manuscript Support
- Generate structured background sections with citations.
- Produce preliminary results summaries tied to source documents.
- Highlight novelty by contrasting your idea with prior art from multiple fields.

### 7. Lab Memory and Training
- Onboard new students with validated summaries and canonical reading lists.
- Keep lab meeting prep consistent and evidence-grounded.
- Create reusable, citable internal knowledge assets.

### 8. Teaching and Curriculum Development
- Build topic modules with canonical papers, key excerpts, and code examples.
- Create annotated problem sets with supporting references.
- Teach evidence literacy with citable passages and provenance.

## Optimization Levers (How to Improve Output Quality)

### A. Data Quality and Coverage
- Use Postgres as the system of record; it should contain every document and chunk.
- Normalize metadata (title, authors, year, venue) to improve filtering and recall.
- Deduplicate near-identical PDFs and code forks.
- Prefer enhanced PDF parsing for citation-ready passages.

### B. Embeddings and Vector Store
- Keep a single, consistent embedding model for paper and code collections.
- Monitor distribution drift after large ingestion batches.
- Rebuild collections only from Postgres (never treat ChromaDB as source of truth).

### C. Retrieval and Ranking
- Use hybrid search (semantic + lexical + graph) with RRF for stable ranking.
- Adjust filters (year, concepts, org, language) before increasing top-k.
- Maintain a golden query set to detect regressions in search quality.

### D. Graph Quality
- Expand concept extraction coverage and normalize concept naming.
- Add edges for citations, co-occurrence, and method usage.
- Periodically re-run concept linking when vocabularies shift.

### E. Evidence and Citability
- Require page-level coordinates for any claim used in academic writing.
- Set clear NLI thresholds for claim verification to avoid weak citations.
- Keep an audit trail (doc_id, page_num, offsets) for reproducibility.

### F. Operational Reliability
- Use resumable ingestion and migration scripts for long runs.
- Keep disk usage under 85 percent to avoid ChromaDB corruption.
- Run regular backups and integrity checks.

## Recommended Academic Workflows

### Workflow 1: Literature Mapping Sprint
1. Define a precise scope and keyword set.
2. Run hybrid search with filters and capture the top 200 to 500 passages.
3. Use the graph to cluster papers by concept co-occurrence.
4. Export a reading list with evidence-bound excerpts.

### Workflow 2: Evidence-Bound Writing
1. Draft claims in outline form.
2. Use the evidence pipeline to verify claims against source passages.
3. Replace unsupported claims with verified alternatives or remove them.
4. Export citations in the target format.

### Workflow 3: Method Replication
1. Find the original method paper and its linked code implementation.
2. Extract parameters, datasets, and evaluation metrics from passages.
3. Compare at least two independent implementations for robustness.
4. Record reproducibility notes and deviations.

### Workflow 4: Cross-Domain Hypothesis
1. Run a concept query to find adjacent domains.
2. Pull evidence from both domains and identify shared mechanisms.
3. Draft a testable hypothesis and validate evidence coverage.

### Workflow 5: Lab Onboarding
1. Provide a curated reading list with citations.
2. Add a short lab-specific concept map.
3. Include key code repositories for replication.

## Evaluation and QA
- Use `tests/golden_queries.json` and `scripts/golden_query_eval.py` to track quality.
- Run `scripts/test_polymathic_queries.py` for search stability checks.
- Maintain a weekly spot-check of evidence-bound citations.

## Migration and Model Updates (Why the BGE-M3 Path Helps)
- Better recall and semantic alignment across long or cross-domain queries.
- Unified embeddings across papers and code, reducing search fragmentation.
- Higher quality vector space for concept linkage and graph enrichment.

## Quick Command Recipes

```bash
# Semantic search (CLI)
python polymath_cli.py search "spatial transcriptomics" -n 10 --expand

# Graph query example
python polymath_cli.py graph "MATCH (p:Paper)-[:MENTIONS]->(c:CONCEPT) RETURN c.name, count(p) ORDER BY count(p) DESC LIMIT 10"

# Evidence-bound demo (cite from PDF passages)
python lib/pqe_response_generator.py --demo
```

```python
from lib.hybrid_search_v2 import HybridSearcherV2

hs = HybridSearcherV2()

# Papers with filters
results = hs.search_papers(
    "spatial transcriptomics",
    n=10,
    year_min=2022,
    concepts=["transformer"]
)

# Hybrid search across papers + code
results = hs.hybrid_search("multiple instance learning", n=20)
```

## Maintenance Cadence
- Weekly: run consistency checks and add new priority papers.
- Monthly: update concept extraction and review top retrieval failures.
- Quarterly: audit coverage gaps and adjust collection strategy.
