# Polymath 2.0: Ground-Up Redesign Specification

**Author**: Claude + Max Van Belkum
**Date**: 2026-01-13
**Status**: Architecture Proposal

---

## Executive Summary

Polymath 1.0 is a **well-engineered storage and retrieval system** that successfully indexes 748K passages and 4.8M concepts. However, it **fundamentally fails at its core mission—polymathic discovery**—because it captures **labels and embeddings** but not the **structural/mechanistic knowledge** needed to reason about cross-domain transfer.

**The Core Problem**: BridgeMine generates hypotheses like "Apply optimal transport to cyber_attack_detection in spatial transcriptomics" because it extracts METHOD labels without understanding HOW methods work, WHAT data structures they operate on, or WHY they might transfer.

**The Solution**: Polymath 2.0 is a ground-up redesign with:
1. **Zotero-first workflow** for metadata integrity
2. **Hierarchical passages** with parent-child relationships
3. **Mechanism-centric knowledge graph** (not just labels)
4. **Agentic workflows** with document grounding
5. **Cross-domain reasoning engine** with transfer validation

---

## Part 1: Critical Design Flaws in Polymath 1.0

### Flaw 1: Labels Without Mechanisms

**Current extraction**:
```
Passage: "We use optimal transport for crack detection in concrete structures"
→ Extracted: ["optimal_transport" (METHOD), "crack_detection" (PROBLEM)]
```

**What's missing**:
- What MECHANISM does optimal transport implement? (distributional matching)
- What DATA STRUCTURE does it operate on? (point clouds with weights)
- What OBJECTIVE does it optimize? (minimize transport cost)
- What PROPERTIES make it suitable for transfer? (domain-agnostic, metric guarantees)

**Evidence from DB**: We have 544K "method" labels but **NO "mechanism" concept type**. The extraction schema never asked for mechanisms.

### Flaw 2: Passages Too Short for Context

**Current distribution** (from Postgres):
| Length | Count | Problem |
|--------|-------|---------|
| 200-500 chars | 221K | Single sentence, no context |
| 500-1000 chars | 497K | 1-2 paragraphs, limited context |
| 1000-1500 chars | 35 | Rare |
| > 1500 chars | 4K | Exceptional |

**Impact**: When LLM retrieves a 500-char passage, it lacks:
- The paper's methodology section
- The experimental setup
- The results that validate the claim
- The limitations and caveats

**Result**: Hallucination increases because claims are decontextualized.

### Flaw 3: No Parent-Child Passage Relationships

**Current structure**:
```
Document → [Passage1, Passage2, ..., PassageN]  (flat list)
```

**What's needed**:
```
Document
├── Abstract (summary)
├── Introduction
│   ├── Para 1: Problem statement
│   ├── Para 2: Prior work
│   └── Para 3: Our contribution
├── Methods
│   ├── Section 3.1: Data
│   │   ├── Para: Dataset description
│   │   └── Para: Preprocessing
│   └── Section 3.2: Model
│       ├── Para: Architecture
│       └── Para: Training
└── Results
    └── ...
```

**Why it matters**: When retrieving a claim, we should be able to expand UP to section context and DOWN to supporting details.

### Flaw 4: Metadata Catastrophe

**Current state**:
- 80.8% of documents missing DOI
- 99% missing PMID
- 100% of passages lack proper provenance
- Cannot cite anything properly

**Root cause**: PDFs were ingested without metadata enrichment. No Zotero integration. Title-hash collisions block re-ingestion of papers with existing metadata.

### Flaw 5: SIMILAR_TO Based on Embedding Distance

**Current implementation** (neo4j_typed_graph.py):
```python
embeddings = model.encode(problem_names)
if cosine_similarity(embeddings) > 0.7:
    create_edge(p1 -[:SIMILAR_TO]-> p2)
```

**Failure cases**:
- `crack_detection` ≈ `fraud_detection` ≈ `cancer_detection` (all have "detection")
- `optimal_transport` ≈ `public_transport` (both have "transport")
- `image_reconstruction` ≈ `3d_reconstruction` (both have "reconstruction")

**These are textually similar but mechanistically different!**

### Flaw 6: Components Not Integrated

**Built but not connected**:
- `evidence_extractor.py` - Extracts evidence spans via NLI ✅
- `citation_builder.py` - Builds citations ✅
- `pqe_response_generator.py` - Generates responses ✅
- **BUT**: Response generator doesn't call evidence extractor

**Result**: System has the pieces but doesn't use them together.

---

## Part 2: Polymath 2.0 Architecture Overview

### Design Principles

1. **Zotero is the source of truth for metadata** - All PDFs go through Zotero first
2. **Passages have hierarchy** - Every passage knows its parent section and document
3. **Mechanisms, not labels** - Extract HOW methods work, not just WHAT they're called
4. **Grounded reasoning** - Every claim must cite specific evidence
5. **Transfer validation** - Cross-domain hypotheses require mechanism matching

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     POLYMATH 2.0 ARCHITECTURE                    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   ZOTERO    │────▶│   INGEST    │────▶│  POSTGRES   │
│  (PDFs +    │     │  PIPELINE   │     │   (SoR)     │
│  Metadata)  │     └─────────────┘     └──────┬──────┘
└─────────────┘                                │
                                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    HIERARCHICAL PASSAGE STORE                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ Document │──│ Section  │──│ Paragraph│──│ Sentence │        │
│  │  (root)  │  │  (L1)    │  │   (L2)   │  │   (L3)   │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
└─────────────────────────────────────────────────────────────────┘
                                               │
                    ┌──────────────────────────┼───────────────────┐
                    ▼                          ▼                   ▼
           ┌──────────────┐          ┌──────────────┐     ┌──────────────┐
           │  ChromaDB    │          │    Neo4j     │     │ Code Index   │
           │ (Embeddings) │          │ (Mechanism   │     │ (Impl Links) │
           │              │          │   Graph)     │     │              │
           └──────────────┘          └──────────────┘     └──────────────┘
                    │                          │                   │
                    └──────────────────────────┼───────────────────┘
                                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      AGENTIC REASONING LAYER                     │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐ │
│  │   Query    │──│  Retrieve  │──│  Validate  │──│  Synthesize│ │
│  │ Decompose  │  │ + Expand   │  │ + Ground   │  │ + Cite     │ │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   CROSS-DOMAIN DISCOVERY ENGINE                  │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐ │
│  │ Gap        │──│ Mechanism  │──│ Transfer   │──│ Hypothesis │ │
│  │ Detection  │  │ Matching   │  │ Validation │  │ Generation │ │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Part 3: Zotero-First Workflow

### Why Zotero?

1. **Metadata is already curated** - DOI, PMID, authors, venue, year
2. **PDF management built-in** - File storage, deduplication, organization
3. **Export to CSV** - Clean structured metadata for ingestion
4. **Collections** - Organize by project, topic, reading list
5. **Tagging** - User annotations preserved

### Workflow

```
User finds paper
      │
      ▼
┌─────────────────┐
│ Add to Zotero   │ ◀── Browser extension, DOI lookup, manual entry
│ (with PDF)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Zotero extracts │ ◀── DOI, PMID, authors, venue, year, abstract
│ metadata        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Export CSV      │ ◀── Nightly sync or on-demand
│ + PDF paths     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Polymath 2.0    │
│ Ingest Pipeline │
│ - Parse PDF     │
│ - Link metadata │
│ - Extract hier  │
│ - Extract mech  │
└─────────────────┘
```

### Zotero Export Schema

```csv
zotero_key,doi,pmid,title,authors,year,venue,abstract,pdf_path,collections,tags
ABC123,10.1038/xxx,12345678,"Paper Title","Author1; Author2",2024,"Nature",
  "Abstract text...","/path/to/PDF.pdf","spatial-omics;methods","read;important"
```

### Polymath 2.0 Ingest from Zotero

```python
def ingest_from_zotero_export(csv_path: str):
    """Ingest papers from Zotero CSV export with full metadata."""

    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        # 1. Create document with full metadata
        doc = Document(
            zotero_key=row['zotero_key'],
            doi=row['doi'],
            pmid=row['pmid'],
            title=row['title'],
            authors=parse_authors(row['authors']),
            year=row['year'],
            venue=row['venue'],
            abstract=row['abstract'],
            collections=row['collections'].split(';'),
            tags=row['tags'].split(';')
        )

        # 2. Parse PDF with hierarchy
        pdf_path = row['pdf_path']
        hierarchy = parse_pdf_hierarchical(pdf_path)

        # 3. Store with parent-child relationships
        store_hierarchical_passages(doc, hierarchy)

        # 4. Extract mechanisms (not just labels)
        extract_mechanisms(doc)

        # 5. Link to code implementations if available
        link_implementations(doc)
```

### Benefits

| Before (PDF-first) | After (Zotero-first) |
|-------------------|---------------------|
| 80% missing DOI | 100% have DOI (Zotero enforces) |
| 99% missing PMID | ~70% have PMID (auto-lookup) |
| Title hash collisions | Zotero key is canonical ID |
| No collections | Organized by project |
| No user annotations | Tags preserved |
| Manual PDF management | Automatic sync |

---

## Part 4: Hierarchical Passage Design

### The Problem with Flat Passages

Current system chunks PDFs into ~500-char passages with no structure. When you retrieve a passage, you get:

```
"The model achieves 0.87 accuracy on the test set, outperforming prior methods."
```

But you DON'T get:
- What model? (need to look at Methods section)
- What test set? (need to look at Data section)
- What prior methods? (need to look at Related Work)
- What are the caveats? (need to look at Limitations)

### Hierarchical Passage Schema

```sql
CREATE TABLE passages_v2 (
    passage_id UUID PRIMARY KEY,
    doc_id UUID REFERENCES documents(doc_id),

    -- Hierarchy
    parent_id UUID REFERENCES passages_v2(passage_id),  -- NULL for root
    level INT NOT NULL,  -- 0=document, 1=section, 2=paragraph, 3=sentence
    position INT NOT NULL,  -- Order within parent

    -- Content
    passage_text TEXT NOT NULL,
    passage_type VARCHAR(50),  -- 'abstract', 'intro', 'methods', 'results', 'discussion', 'caption', etc.

    -- Location
    page_num INT,
    page_char_start INT,
    page_char_end INT,

    -- Context window (for retrieval)
    context_before TEXT,  -- 200 chars before
    context_after TEXT,   -- 200 chars after

    -- Metadata
    quality_score FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Index for hierarchy traversal
CREATE INDEX idx_passages_parent ON passages_v2(parent_id);
CREATE INDEX idx_passages_doc_level ON passages_v2(doc_id, level);
```

### Passage Types and Levels

```
Level 0: Document
├── Level 1: Section
│   ├── abstract
│   ├── introduction
│   ├── related_work
│   ├── methods
│   │   └── Level 2: Subsection
│   │       ├── methods.data
│   │       ├── methods.model
│   │       └── methods.training
│   ├── results
│   ├── discussion
│   ├── conclusion
│   └── references
│
└── Level 2: Paragraph (within each section)
    └── Level 3: Sentence (fine-grained claims)
```

### Context Expansion

When retrieving a sentence, automatically expand to include context:

```python
def retrieve_with_context(passage_id: UUID, expand_levels: int = 2) -> PassageWithContext:
    """Retrieve a passage with parent context for grounded reasoning."""

    passage = get_passage(passage_id)

    # Get parent chain
    parents = []
    current = passage
    for _ in range(expand_levels):
        if current.parent_id:
            parent = get_passage(current.parent_id)
            parents.append(parent)
            current = parent

    # Get siblings for local context
    siblings = get_siblings(passage_id)

    return PassageWithContext(
        passage=passage,
        parents=parents,  # Section → Subsection → ...
        siblings=siblings,  # Neighboring paragraphs
        context_window=passage.context_before + passage.passage_text + passage.context_after
    )
```

### Target Passage Lengths

| Level | Target Length | Purpose |
|-------|---------------|---------|
| Document | 5000-10000 chars | Full paper summary |
| Section | 1500-3000 chars | Thematic grouping |
| Paragraph | 500-1500 chars | Coherent idea unit |
| Sentence | 100-300 chars | Atomic claim |

**Key change**: Passages should be **1500-3000 chars** by default (section-level), not 500 chars.

---

## Part 5: Mechanism-Centric Knowledge Graph

### The Core Insight

**Current**: `METHOD -[:SOLVES]-> PROBLEM` (two node types, one edge type)

**Needed**: Multi-layer graph capturing HOW methods work

```
METHOD -[:IMPLEMENTS]-> MECHANISM
MECHANISM -[:OPERATES_ON]-> DATA_STRUCTURE
MECHANISM -[:OPTIMIZES]-> OBJECTIVE
DATA_STRUCTURE -[:APPEARS_IN]-> DOMAIN
PROBLEM -[:REQUIRES]-> MECHANISM
```

### Node Types

```cypher
// 1. METHOD: Named algorithms/techniques
(:Method {
    name: "optimal_transport",
    aliases: ["OT", "Wasserstein distance", "earth mover distance"],
    paper_count: 3839,
    first_mentioned: 1781  // Monge!
})

// 2. MECHANISM: How methods work (the transferable part)
(:Mechanism {
    name: "distributional_matching",
    description: "Find minimum-cost correspondence between probability distributions",
    properties: ["domain_agnostic", "metric_guarantees", "computationally_tractable"],
    mathematical_form: "argmin_{T} sum_{i,j} T_{ij} * C_{ij}"
})

// 3. DATA_STRUCTURE: What methods operate on
(:DataStructure {
    name: "weighted_point_cloud",
    description: "Set of points in R^d with associated weights/masses",
    features: ["spatial_coordinates", "mass_weights", "sparse_sampling"],
    examples: ["spot_coordinates_in_visium", "pixel_locations_in_image", "gps_coordinates"]
})

// 4. OBJECTIVE: What methods optimize
(:Objective {
    name: "minimize_transport_cost",
    description: "Find assignment that minimizes total movement cost",
    metrics: ["wasserstein_distance", "sinkhorn_divergence"],
    constraints: ["mass_conservation", "non_negativity"]
})

// 5. PROBLEM: Application challenges
(:Problem {
    name: "spatial_gene_imputation",
    domain: "spatial_transcriptomics",
    data_characteristics: ["sparse_spots", "high_dimensional_genes", "spatial_structure"],
    success_metrics: ["pearson_r", "mse", "biological_validity"]
})

// 6. DOMAIN: Application areas
(:Domain {
    name: "spatial_transcriptomics",
    parent: "computational_biology",
    data_types: ["visium", "slide_seq", "merfish", "xenium"],
    key_challenges: ["sparsity", "resolution", "batch_effects"]
})
```

### Edge Types

```cypher
// Method implements mechanism
(:Method)-[:IMPLEMENTS {confidence: 0.95}]->(:Mechanism)

// Mechanism operates on data structure
(:Mechanism)-[:OPERATES_ON {requirements: ["weighted", "discrete"]}]->(:DataStructure)

// Mechanism optimizes objective
(:Mechanism)-[:OPTIMIZES {typical_formulation: "LP or entropic regularization"}]->(:Objective)

// Data structure appears in domain
(:DataStructure)-[:APPEARS_IN {examples: ["Visium spots", "MERFISH transcripts"]}]->(:Domain)

// Problem requires mechanism
(:Problem)-[:REQUIRES {why: "Need to align distributions across sections"}]->(:Mechanism)

// Method applied to problem (with evidence)
(:Method)-[:APPLIED_TO {paper_doi: "10.1038/xxx", performance: 0.92}]->(:Problem)
```

### Mechanism Extraction Prompt

```python
MECHANISM_EXTRACTION_PROMPT = """
Analyze this passage and extract structured knowledge about methods and mechanisms.

For each METHOD mentioned, extract:
1. method_name: The named algorithm/technique
2. mechanism: HOW it works (the transferable computational pattern)
3. data_structure: WHAT data objects it operates on
4. objective: WHAT it optimizes/achieves
5. properties: Key properties that enable/limit transfer
6. evidence_quote: Direct quote from passage supporting this

Output JSON:
{
  "methods": [
    {
      "method_name": "optimal_transport",
      "mechanism": {
        "name": "distributional_matching",
        "description": "Finds minimum-cost correspondence between probability distributions",
        "mathematical_form": "argmin_T sum(T_ij * C_ij) subject to marginal constraints"
      },
      "data_structure": {
        "type": "weighted_point_cloud",
        "features": ["spatial_coordinates", "mass_weights"],
        "requirements": ["discrete samples", "defined cost matrix"]
      },
      "objective": {
        "name": "minimize_transport_cost",
        "interpretation": "Total 'work' to move mass from source to target distribution"
      },
      "properties": [
        "domain_agnostic",
        "metric_guarantees",
        "computationally_tractable_with_entropic_regularization"
      ],
      "evidence_quote": "We use optimal transport to align spatial distributions..."
    }
  ]
}

Passage:
{passage_text}
"""
```

### Transfer Validation Query

The key insight: **Valid transfer = Same mechanism + Compatible data structure + Different domain**

```cypher
// Find methods that could transfer to spatial transcriptomics
MATCH (m:Method)-[:IMPLEMENTS]->(mech:Mechanism)-[:OPERATES_ON]->(ds:DataStructure),
      (m)-[:APPLIED_TO]->(p1:Problem)-[:IN_DOMAIN]->(source:Domain)

// Find target problems with compatible data structures
MATCH (ds)-[:APPEARS_IN]->(target:Domain {name: "spatial_transcriptomics"}),
      (p2:Problem)-[:IN_DOMAIN]->(target),
      (p2)-[:REQUIRES]->(mech2:Mechanism)

// Check mechanism compatibility
WHERE mech.name = mech2.name  // Same mechanism needed!
  AND NOT (m)-[:APPLIED_TO]->(p2)  // Not yet applied

RETURN m.name as method,
       mech.name as mechanism,
       ds.name as data_structure,
       source.name as source_domain,
       p2.name as target_problem,
       "Transfer valid: Same mechanism (" + mech.name + ") operates on " +
       ds.name + " which appears in spatial transcriptomics" as rationale
```

### Actionable Hypothesis Generation

Instead of: "Apply optimal transport to cyber_attack_detection"

Generate:
```
HYPOTHESIS: Apply optimal transport for spatial section alignment

MECHANISM: Distributional matching (Wasserstein distance minimization)
DATA STRUCTURE: Weighted point clouds
  - Source: Visium spots with expression weights
  - Target: Aligned spatial coordinates

TRANSFER RATIONALE:
1. Visium spots are naturally point clouds with weights (total UMI counts)
2. Distributional matching aligns these across sections
3. Wasserstein distance respects spatial structure
4. Entropic regularization makes it computationally tractable

EXPERIMENT:
- Dataset: ENACT serial sections (P1, P2, P5)
- Baseline: ICP registration, landmark-based alignment
- Metric: Alignment quality (overlap %, biological structure preservation)
- Compute: 2 GPU-hours per section pair
- Falsifier: If Wasserstein distance doesn't outperform ICP by >5%, reject

PRIOR ART CHECK:
- [x] Searched "optimal transport spatial transcriptomics alignment"
- [x] Found 3 papers using OT for spot deconvolution, but NOT for section alignment
- [x] This is novel application of known mechanism
```

---

## Part 6: Agentic Workflows and LLM Integration

### Design Principle: Grounded Reasoning

Every claim generated by an LLM must be:
1. **Decomposable** into verifiable sub-claims
2. **Retrievable** from the knowledge base
3. **Validated** against source documents
4. **Cited** with specific evidence

### The CogFlow Pattern (from our papers)

Based on the CogFlow paper in our DB: **Perception ⇒ Internalization ⇒ Reasoning**

```
USER QUERY
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ PERCEPTION: Understand what's being asked                   │
│ - Parse query intent                                        │
│ - Identify required knowledge types                         │
│ - Determine if synthesis, retrieval, or discovery task     │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ INTERNALIZATION: Ground in knowledge base                   │
│ - Decompose into sub-queries                                │
│ - Retrieve relevant passages with hierarchy                 │
│ - Expand to section context                                 │
│ - Extract mechanisms from retrieved content                 │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ REASONING: Synthesize with grounding                        │
│ - Generate claims with explicit citations                   │
│ - Validate each claim against retrieved evidence            │
│ - Check for contradictions                                  │
│ - Identify gaps requiring additional retrieval              │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ OUTPUT: Structured response with audit trail                │
│ - Claims with evidence citations                            │
│ - Confidence scores per claim                               │
│ - Identified knowledge gaps                                 │
│ - Suggested follow-up queries                               │
└─────────────────────────────────────────────────────────────┘
```

### Agentic Response Pipeline

```python
class PolymathAgent:
    """Agentic reasoning with document grounding."""

    def __init__(self, retriever: HierarchicalRetriever, validator: EvidenceValidator):
        self.retriever = retriever
        self.validator = validator
        self.reasoning_trace = []

    async def answer(self, query: str) -> GroundedResponse:
        # 1. PERCEPTION: Decompose query
        sub_queries = await self.decompose_query(query)
        self.reasoning_trace.append(("decompose", sub_queries))

        # 2. INTERNALIZATION: Retrieve and expand
        evidence_map = {}
        for sq in sub_queries:
            # Retrieve with hierarchy
            passages = await self.retriever.retrieve_with_context(sq, expand_levels=2)

            # Extract mechanisms from passages
            mechanisms = await self.extract_mechanisms(passages)

            evidence_map[sq] = {
                "passages": passages,
                "mechanisms": mechanisms
            }
        self.reasoning_trace.append(("retrieve", evidence_map))

        # 3. REASONING: Generate grounded claims
        claims = []
        for sq, evidence in evidence_map.items():
            claim = await self.generate_claim(sq, evidence)

            # Validate against evidence
            validation = await self.validator.validate(claim, evidence["passages"])

            if validation.supported:
                claims.append(GroundedClaim(
                    text=claim,
                    evidence=validation.supporting_passages,
                    confidence=validation.confidence,
                    citations=validation.citations
                ))
            else:
                # Mark as ungrounded, flag for human review
                claims.append(UngroundedClaim(
                    text=claim,
                    reason=validation.rejection_reason,
                    suggested_sources=validation.suggested_sources
                ))

        self.reasoning_trace.append(("reason", claims))

        # 4. OUTPUT: Structured response
        return GroundedResponse(
            claims=claims,
            reasoning_trace=self.reasoning_trace,
            knowledge_gaps=self.identify_gaps(evidence_map),
            follow_up_queries=self.suggest_follow_ups(claims)
        )
```

### Evidence Validation

```python
class EvidenceValidator:
    """Validate claims against retrieved passages."""

    def __init__(self, nli_model: str = "microsoft/deberta-v3-large-mnli"):
        self.nli = pipeline("text-classification", model=nli_model)

    async def validate(self, claim: str, passages: List[Passage]) -> ValidationResult:
        supporting = []
        contradicting = []

        for passage in passages:
            # Check entailment
            result = self.nli(f"{passage.text} [SEP] {claim}")

            if result["label"] == "ENTAILMENT" and result["score"] > 0.7:
                supporting.append(PassageEvidence(
                    passage=passage,
                    score=result["score"],
                    citation=self.build_citation(passage)
                ))
            elif result["label"] == "CONTRADICTION" and result["score"] > 0.7:
                contradicting.append(PassageEvidence(
                    passage=passage,
                    score=result["score"],
                    reason="Contradicts claim"
                ))

        if contradicting:
            return ValidationResult(
                supported=False,
                rejection_reason=f"Contradicted by {len(contradicting)} passages",
                contradicting_passages=contradicting
            )

        if not supporting:
            return ValidationResult(
                supported=False,
                rejection_reason="No supporting evidence found",
                suggested_sources=self.suggest_sources(claim)
            )

        return ValidationResult(
            supported=True,
            confidence=np.mean([s.score for s in supporting]),
            supporting_passages=supporting,
            citations=[s.citation for s in supporting]
        )
```

### Citation Format

```python
class Citation:
    """Structured citation with evidence."""

    doc_id: UUID
    passage_id: UUID

    # Bibliographic
    authors: str  # "Smith et al."
    year: int
    title: str
    venue: str
    doi: str

    # Specific location
    page_num: int
    section: str

    # Evidence
    quote: str  # Direct quote supporting claim
    context: str  # Surrounding context

    def format_inline(self) -> str:
        return f"({self.authors}, {self.year})"

    def format_full(self) -> str:
        return f"{self.authors}. ({self.year}). {self.title}. {self.venue}. doi:{self.doi}"

    def format_with_evidence(self) -> str:
        return f'{self.format_inline()} states: "{self.quote}" (p. {self.page_num}, {self.section})'
```

---

## Part 7: Cross-Domain Discovery Engine

### The Gap Detection Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                    CROSS-DOMAIN DISCOVERY ENGINE                  │
└──────────────────────────────────────────────────────────────────┘

Step 1: BUILD MECHANISM GRAPH
        - Extract mechanisms from all papers
        - Link methods → mechanisms → data structures → domains
        - Identify which problems require which mechanisms

Step 2: IDENTIFY GAPS
        - Query: "Mechanisms used in domain A but not domain B"
        - Filter: Data structure compatibility
        - Rank: By mechanism generality and success rate

Step 3: VALIDATE TRANSFER
        - Check: Same mechanism
        - Check: Compatible data structure
        - Check: Objective alignment
        - Check: No fundamental domain barriers

Step 4: GENERATE HYPOTHESIS
        - Specify: Exact mechanism to transfer
        - Specify: Data structure mapping
        - Specify: Evaluation approach
        - Specify: Falsification criteria

Step 5: NOVELTY CHECK
        - Search: PubMed, Semantic Scholar, Polymath
        - If exists: Label as "literature review"
        - If novel: Label as "novel hypothesis"
```

### Gap Detection Query

```cypher
// Find mechanisms successful in other domains but not applied to spatial transcriptomics
MATCH (m:Method)-[:IMPLEMENTS]->(mech:Mechanism)-[:OPERATES_ON]->(ds:DataStructure),
      (m)-[:APPLIED_TO]->(p1:Problem {success: true})-[:IN_DOMAIN]->(source:Domain)
WHERE source.name <> "spatial_transcriptomics"

// Check if data structure is compatible with spatial transcriptomics
MATCH (ds)-[:APPEARS_IN]->(target:Domain {name: "spatial_transcriptomics"})

// Check if there's a problem in spatial transcriptomics requiring this mechanism
MATCH (p2:Problem)-[:IN_DOMAIN]->(target),
      (p2)-[:REQUIRES]->(mech2:Mechanism)
WHERE mech.properties_overlap(mech2) > 0.7

// Ensure method not already applied
AND NOT (m)-[:APPLIED_TO]->(p2)

// Return ranked by mechanism success rate and novelty
RETURN m.name as method,
       mech.name as mechanism,
       ds.name as data_structure,
       source.name as source_domain,
       p2.name as target_problem,
       m.success_rate as prior_success,
       count{(m)-[:APPLIED_TO]->(:Problem)} as applications
ORDER BY prior_success DESC, applications ASC
LIMIT 20
```

### Transfer Validation Checklist

```python
class TransferValidator:
    """Validate cross-domain transfer hypotheses."""

    def validate(self, hypothesis: TransferHypothesis) -> TransferValidation:
        checks = []

        # 1. Mechanism match
        checks.append(self.check_mechanism_match(
            source_mechanism=hypothesis.source_problem.mechanism,
            target_mechanism=hypothesis.target_problem.required_mechanism
        ))

        # 2. Data structure compatibility
        checks.append(self.check_data_structure_compatibility(
            source_ds=hypothesis.method.data_structure,
            target_ds=hypothesis.target_problem.data_characteristics
        ))

        # 3. Objective alignment
        checks.append(self.check_objective_alignment(
            method_objective=hypothesis.method.objective,
            problem_goal=hypothesis.target_problem.success_metrics
        ))

        # 4. No fundamental barriers
        checks.append(self.check_no_barriers(
            source_domain=hypothesis.source_problem.domain,
            target_domain=hypothesis.target_problem.domain
        ))

        # 5. Novelty
        checks.append(self.check_novelty(
            method=hypothesis.method.name,
            target=hypothesis.target_problem.name
        ))

        return TransferValidation(
            checks=checks,
            valid=all(c.passed for c in checks),
            confidence=np.mean([c.confidence for c in checks]),
            rationale=self.generate_rationale(checks)
        )
```

### Actionable Hypothesis Template

```yaml
hypothesis:
  title: "Apply {method} for {target_problem} in spatial transcriptomics"

  transfer_rationale:
    source_domain: {source_domain}
    source_problem: {source_problem}
    success_evidence: "{method} achieved {performance} on {source_problem} ({citation})"

    mechanism:
      name: {mechanism_name}
      description: {mechanism_description}
      key_property: {why_transferable}

    data_structure:
      source: {source_data_structure}
      target: {target_data_structure}
      mapping: {how_to_map}

    objective:
      source: {source_objective}
      target: {target_objective}
      alignment: {objective_alignment}

  experiment:
    dataset: {specific_dataset}
    baseline: {named_baseline_method}
    metric: {specific_metric}
    expected_improvement: {quantified_expectation}
    compute_budget: {time_and_resources}

    falsifier: "If {method} does not outperform {baseline} by >{threshold}% on {metric}, reject hypothesis"

  novelty_check:
    pubmed_query: "{method} AND spatial transcriptomics AND {target_problem}"
    pubmed_results: {num_results}
    semantic_scholar_results: {num_results}
    polymath_results: {num_results}
    novelty_verdict: "{novel|existing|partially_explored}"

  actionability:
    implementation_difficulty: {low|medium|high}
    data_availability: {available|need_to_generate|not_available}
    compute_requirements: {spec}
    time_to_implement: {estimate}

  priority_score: {0-100}
```

---

## Part 8: Code-Literature Integration

### The Problem

Currently, code repos and papers are in separate silos:
- Papers mention "we use ResNet" but don't link to implementations
- Code repos have implementations but no links to papers explaining them
- Can't ask "Where is {method} from {paper} implemented?"

### Unified Semantic Space

```
┌─────────────────────────────────────────────────────────────────┐
│                   UNIFIED SEMANTIC SPACE                         │
│                                                                  │
│  ┌─────────────┐                      ┌─────────────┐           │
│  │   Papers    │                      │    Code     │           │
│  │             │                      │             │           │
│  │ "We propose │◀─── shared concept ──▶│ class OT:   │           │
│  │  optimal    │     "optimal         │   def fit() │           │
│  │  transport" │      transport"      │             │           │
│  └─────────────┘                      └─────────────┘           │
│         │                                    │                  │
│         ▼                                    ▼                  │
│  ┌─────────────┐                      ┌─────────────┐           │
│  │ Concept     │───── same node ─────▶│ Concept     │           │
│  │ Embedding   │      in graph        │ Embedding   │           │
│  └─────────────┘                      └─────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

### Code-Concept Linking

```python
class CodeConceptLinker:
    """Link code implementations to conceptual knowledge."""

    def link_code_to_concepts(self, code_chunk: CodeChunk) -> List[ConceptLink]:
        links = []

        # 1. Extract concepts from code (docstrings, comments, names)
        code_concepts = self.extract_code_concepts(code_chunk)

        # 2. Find matching paper concepts
        for concept in code_concepts:
            paper_mentions = self.find_paper_mentions(concept)

            for mention in paper_mentions:
                # 3. Verify semantic alignment
                if self.verify_alignment(code_chunk, mention):
                    links.append(ConceptLink(
                        code_chunk=code_chunk,
                        paper_passage=mention.passage,
                        concept=concept,
                        link_type=self.classify_link(code_chunk, mention),
                        confidence=mention.similarity
                    ))

        return links

    def classify_link(self, code: CodeChunk, paper: Passage) -> str:
        """Classify the type of code-paper link."""
        if code.is_implementation_of(paper):
            return "IMPLEMENTS"  # Code implements method from paper
        elif paper.describes(code):
            return "DESCRIBES"   # Paper describes this code pattern
        elif code.extends(paper):
            return "EXTENDS"     # Code extends method from paper
        else:
            return "RELATED"     # General conceptual relationship
```

### Queries Enabled

```cypher
// Find implementations of a method from a paper
MATCH (paper:Paper)-[:INTRODUCES]->(method:Method),
      (code:CodeChunk)-[:IMPLEMENTS]->(method)
WHERE paper.doi = "10.1038/xxx"
RETURN code.file_path, code.chunk_text, code.repo

// Find papers explaining a code pattern
MATCH (code:CodeChunk)-[:RELATED_TO]->(concept:Concept),
      (paper:Paper)-[:DISCUSSES]->(concept)
WHERE code.file_path = "squidpy/methods/ot.py"
RETURN paper.title, paper.doi, concept.name

// Find all implementations of a mechanism
MATCH (mech:Mechanism)<-[:IMPLEMENTS]-(method:Method),
      (code:CodeChunk)-[:IMPLEMENTS]->(method)
WHERE mech.name = "distributional_matching"
RETURN method.name, code.repo, code.file_path
```

---

## Part 9: Implementation Roadmap

### Phase 1: Foundation (Weeks 1-4)

**Goal**: Zotero integration + hierarchical passages

| Week | Task | Deliverable |
|------|------|-------------|
| 1 | Zotero export pipeline | `ingest_from_zotero.py` |
| 1 | Passage hierarchy schema | `passages_v2` table |
| 2 | Hierarchical PDF parser | `parse_pdf_hierarchical()` |
| 2 | Context expansion retrieval | `retrieve_with_context()` |
| 3 | Migrate existing passages | All passages have parent_id |
| 3 | Test on 100 papers | Validate hierarchy quality |
| 4 | Zotero sync automation | Nightly sync script |

**Success criteria**:
- 100% of new papers have DOI from Zotero
- Average passage length 1500+ chars
- Parent-child relationships validated

### Phase 2: Mechanism Extraction (Weeks 5-8)

**Goal**: Extract mechanisms, not just labels

| Week | Task | Deliverable |
|------|------|-------------|
| 5 | Design mechanism schema | `mechanism`, `data_structure`, `objective` node types |
| 5 | Create extraction prompt | `mechanism_extraction_prompt.py` |
| 6 | Build extraction pipeline | `extract_mechanisms()` |
| 6 | Process 1000 papers | Initial mechanism graph |
| 7 | Validate mechanism quality | Manual review of 100 samples |
| 7 | Iterate on prompt | Improve extraction accuracy |
| 8 | Full corpus processing | All passages have mechanisms |

**Success criteria**:
- Mechanism extraction accuracy >80%
- All methods linked to mechanisms
- Data structure types standardized

### Phase 3: Agentic Layer (Weeks 9-12)

**Goal**: Grounded reasoning with citations

| Week | Task | Deliverable |
|------|------|-------------|
| 9 | Query decomposition | `decompose_query()` |
| 9 | Evidence validation | `EvidenceValidator` class |
| 10 | Citation builder integration | Citations in all responses |
| 10 | Reasoning trace logging | Audit trail for all queries |
| 11 | Contradiction detection | Alert on conflicting evidence |
| 11 | Gap identification | Flag knowledge gaps |
| 12 | End-to-end testing | 50 query validation |

**Success criteria**:
- Every claim has citation
- Contradiction detection working
- Reasoning traces auditable

### Phase 4: Discovery Engine (Weeks 13-16)

**Goal**: Cross-domain transfer with mechanism matching

| Week | Task | Deliverable |
|------|------|-------------|
| 13 | Gap detection query | `find_transfer_opportunities()` |
| 13 | Transfer validation | `TransferValidator` class |
| 14 | Hypothesis generation | Actionable hypothesis templates |
| 14 | Novelty checking | PubMed + Semantic Scholar integration |
| 15 | Experiment specification | Concrete experiment proposals |
| 15 | Priority scoring | Ranked hypothesis list |
| 16 | End-to-end pipeline | Full discovery workflow |

**Success criteria**:
- 20 actionable hypotheses generated
- All hypotheses pass transfer validation
- Novelty scores accurate

### Phase 5: Code Integration (Weeks 17-20)

**Goal**: Unified code-literature space

| Week | Task | Deliverable |
|------|------|-------------|
| 17 | Code concept extraction | `extract_code_concepts()` |
| 17 | Code-paper linking | `CodeConceptLinker` class |
| 18 | Unified embeddings | Shared semantic space |
| 18 | Cross-modal retrieval | "Find implementations of X" |
| 19 | Index priority repos | squidpy, spatialdata, HIPT, etc. |
| 19 | Link validation | Manual review of links |
| 20 | Full integration | Code + papers in unified graph |

**Success criteria**:
- 10 priority repos indexed
- Code-paper links validated
- Cross-modal queries working

---

## Part 10: Success Metrics

### Quantitative Metrics

| Metric | Polymath 1.0 | Polymath 2.0 Target |
|--------|-------------|-------------------|
| DOI coverage | 15% | 100% |
| Avg passage length | 500 chars | 1500 chars |
| Parent-child relationships | 0% | 100% |
| Mechanism extraction | 0% | 100% |
| Citation accuracy | 0% | 95% |
| Actionable hypotheses | 5-12 | 50+ |
| Transfer validation | N/A | 100% |
| Code-paper links | 0 | 1000+ |

### Qualitative Metrics

1. **Hypothesis actionability**: Can a graduate student implement the proposed experiment in 1 week?
2. **Citation verifiability**: Can every claim be traced to a specific passage?
3. **Transfer validity**: Does the mechanism actually apply to the target domain?
4. **Knowledge completeness**: Are there major gaps in coverage?

### User Value Metrics

1. **Time to insight**: How long to find relevant prior work?
2. **Discovery quality**: Are cross-domain insights genuinely novel?
3. **Trust calibration**: Do confidence scores match actual reliability?
4. **Workflow integration**: Does the system fit into real research workflows?

---

## Appendix A: Schema Definitions

### PostgreSQL Schema

```sql
-- Documents with full Zotero metadata
CREATE TABLE documents_v2 (
    doc_id UUID PRIMARY KEY,
    zotero_key VARCHAR(50) UNIQUE,
    doi VARCHAR(100),
    pmid VARCHAR(20),
    title TEXT NOT NULL,
    title_hash VARCHAR(64),
    authors JSONB,  -- [{name, affiliation, orcid}]
    year INT,
    venue TEXT,
    abstract TEXT,
    collections TEXT[],
    tags TEXT[],
    pdf_path TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Hierarchical passages
CREATE TABLE passages_v2 (
    passage_id UUID PRIMARY KEY,
    doc_id UUID REFERENCES documents_v2(doc_id),
    parent_id UUID REFERENCES passages_v2(passage_id),
    level INT NOT NULL,
    position INT NOT NULL,
    passage_type VARCHAR(50),
    passage_text TEXT NOT NULL,
    context_before TEXT,
    context_after TEXT,
    page_num INT,
    page_char_start INT,
    page_char_end INT,
    quality_score FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Mechanisms extracted from passages
CREATE TABLE mechanisms (
    mechanism_id UUID PRIMARY KEY,
    passage_id UUID REFERENCES passages_v2(passage_id),
    method_name VARCHAR(100),
    mechanism_name VARCHAR(100),
    mechanism_description TEXT,
    data_structure JSONB,
    objective JSONB,
    properties TEXT[],
    evidence_quote TEXT,
    confidence FLOAT,
    extractor_version VARCHAR(20),
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Neo4j Schema

```cypher
// Node constraints
CREATE CONSTRAINT method_name IF NOT EXISTS FOR (m:Method) REQUIRE m.name IS UNIQUE;
CREATE CONSTRAINT mechanism_name IF NOT EXISTS FOR (m:Mechanism) REQUIRE m.name IS UNIQUE;
CREATE CONSTRAINT data_structure_name IF NOT EXISTS FOR (d:DataStructure) REQUIRE d.name IS UNIQUE;
CREATE CONSTRAINT objective_name IF NOT EXISTS FOR (o:Objective) REQUIRE o.name IS UNIQUE;
CREATE CONSTRAINT domain_name IF NOT EXISTS FOR (d:Domain) REQUIRE d.name IS UNIQUE;
CREATE CONSTRAINT problem_name IF NOT EXISTS FOR (p:Problem) REQUIRE p.name IS UNIQUE;

// Indexes
CREATE INDEX method_paper_count IF NOT EXISTS FOR (m:Method) ON (m.paper_count);
CREATE INDEX mechanism_properties IF NOT EXISTS FOR (m:Mechanism) ON (m.properties);
CREATE INDEX problem_domain IF NOT EXISTS FOR (p:Problem) ON (p.domain);
```

---

## Appendix B: Key Papers from Polymath DB

These papers from our own DB informed this redesign:

1. **CogFlow** (2024): Perception ⇒ Internalization ⇒ Reasoning framework
2. **Machine Learning Hybridization** (2024): Cross-domain transfer requires mechanism matching
3. **Inductive Logic Programming** (2013): Relational structures capture dependencies
4. **HuBMAP** (2024): Multi-scale hierarchies for spatial-molecular reasoning
5. **DocDancer** (2026): Agentic document grounding prevents LLM divergence
6. **Cybernetics papers** (2020): Systems theory for knowledge architecture
7. **Principles of Machine Learning** (2017): Hierarchical decomposition for multi-scale reasoning

---

## Appendix C: Migration Path

### From Polymath 1.0 to 2.0

1. **Keep existing data**: Don't delete, migrate
2. **Add new tables**: `documents_v2`, `passages_v2`, `mechanisms`
3. **Parallel operation**: Run both systems during transition
4. **Incremental migration**: Move papers as they're touched
5. **Zotero backfill**: Import existing papers to Zotero, re-export

### Breaking Changes

- Passage IDs will change (new hierarchy)
- Concept extraction schema changes (mechanisms added)
- ChromaDB collection restructured (hierarchical embeddings)
- Neo4j schema completely new

### Backwards Compatibility

- Keep `passages` table as read-only archive
- Redirect queries to `passages_v2` for new searches
- Maintain old MCP tools with deprecation warnings
- Gradual sunset over 6 months

---

## Conclusion

Polymath 2.0 is a ground-up redesign that addresses the fundamental architectural gaps preventing polymathic discovery:

1. **Zotero-first** ensures metadata integrity
2. **Hierarchical passages** preserve context
3. **Mechanism extraction** enables transfer reasoning
4. **Agentic grounding** prevents hallucination
5. **Cross-domain engine** generates actionable hypotheses
6. **Code integration** links theory to implementation

The system moves from "storage and retrieval" to "discovery and reasoning" - the original vision for a polymathic research engine.

---

*This document was generated using Polymath 1.0 to search for papers that could inform its own redesign - a fitting testament to the system's retrieval capabilities, even as it reveals the limitations we must overcome.*
