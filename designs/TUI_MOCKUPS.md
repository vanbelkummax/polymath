# Polymath 2.0 Text User Interface Design

**Author**: Max Van Belkum
**Version**: 2.0.0-alpha
**Last Updated**: 2026-01-13

---

## Design Philosophy

Polymath 2.0's TUI follows these principles:
- **Progressive disclosure**: Simple commands for common tasks, detailed flags for power users
- **Document grounding**: All outputs include source citations with jump-to-source capability
- **Mechanism transparency**: Show HOW methods work, not just labels
- **Agentic collaboration**: LLM-guided workflows with human checkpoints

---

## Main CLI Interface

```bash
$ polymath

Polymath 2.0 - Cross-Domain Knowledge Discovery Platform
═══════════════════════════════════════════════════════════

Corpus: 3,617 papers (748K passages) + 26 code repos (576K chunks)
Status: ✓ Postgres  ✓ ChromaDB  ✓ Neo4j  ⚠ 80% missing DOI

Commands:
  search     Semantic search across papers and code
  discover   Cross-domain transfer opportunities
  bridge     Find methods from domain A applicable to domain B
  ingest     Add papers or code repositories
  validate   Check hypothesis against literature
  stats      System health and corpus statistics

Type 'polymath help <command>' for detailed usage.
```

---

## Search Interface

### Basic Search

```bash
$ polymath search "spatial transcriptomics imputation"

🔍 Searching 748K passages...

📊 Top Results (showing 10 of 247 matches)
─────────────────────────────────────────────────────────

1. ★★★★☆ (95% relevance)
   Tangram: Spatial alignment of single-cell transcriptomics

   Authors: Biancalani et al.
   Year: 2021 | DOI: 10.1038/s41592-021-01264-7

   "We present Tangram, a method for mapping gene expression to spatial
   locations using optimal transport. The algorithm treats gene expression
   as a probability distribution and minimizes the Wasserstein distance..."

   [📄 View full passage] [🔗 Source PDF] [🌐 DOI link]

   Related: optimal_transport (METHOD), wasserstein_distance (MECHANISM),
            point_cloud (DATA_STRUCTURE)

2. ★★★★☆ (92% relevance)
   SpaGE: Spatial Gene Enhancement using scRNA-seq

   Authors: Abdelaal et al.
   Year: 2020 | DOI: 10.1093/nar/gkaa740

   "SpaGE imputes gene expression in spatial transcriptomics by transferring
   information from scRNA-seq using an entropy-regularized optimal transport
   formulation with spatial smoothness constraints..."

   [📄 View full passage] [🔗 Source PDF] [🌐 DOI link]

   Related: optimal_transport (METHOD), entropy_regularization (MECHANISM)

...

Options:
  [n] Next page
  [1-10] View full paper
  [e] Export results
  [r] Refine search
  [q] Quit
```

### Advanced Search with Filters

```bash
$ polymath search "denoising diffusion" \
    --year-min 2023 \
    --domain "spatial_transcriptomics" \
    --show-mechanism \
    -n 20

🔍 Advanced search: denoising diffusion
   Domain filter: spatial_transcriptomics
   Year range: 2023-2026
   Results: 20

📊 Results with Mechanism Analysis
─────────────────────────────────────────────────────────

1. ★★★★★ (98% relevance)
   STDiff: Denoising Diffusion for Spatial Transcriptomics

   Authors: Zhang et al.
   Year: 2024 | DOI: 10.1101/2024.01.12.575123

   📝 Passage:
   "STDiff uses a conditional DDPM to generate high-resolution gene
   expression from low-resolution spatial data. The diffusion process
   operates on the expression matrix X ∈ R^(n×g) where n=spots, g=genes."

   🔧 Mechanism Extracted:
   - METHOD: Denoising Diffusion Probabilistic Model (DDPM)
   - MECHANISM: Iterative denoising via learned score matching
   - DATA_STRUCTURE: 2D spatial grid + gene expression matrix
   - OBJECTIVE: Maximize ELBO, minimize reconstruction error
   - OPERATES_ON: Spatially-indexed expression measurements

   🔗 Transfer potential: High (same data structure as image inpainting)

   [📄 Full paper] [🔗 Source] [🧬 Implementation: GitHub]
```

---

## Discovery Interface (Cross-Domain Transfer)

```bash
$ polymath discover --target "spatial_transcriptomics" \
    --method-family "generative_models" \
    --novelty-threshold 60

🔬 Cross-Domain Discovery Engine
═══════════════════════════════════════════════════════════

Target domain: spatial_transcriptomics
Method family: generative_models
Minimum novelty: 60/100

🔍 Phase 1: Building mechanism graph...
   ✓ Extracted 247 generative methods
   ✓ Mapped 1,893 mechanism instances
   ✓ Found 412 data structure matches

🔍 Phase 2: Gap detection...
   ✓ Identified 67 methods with compatible data structures
   ✓ Filtered to 23 with <30% spatial penetration

🔍 Phase 3: Novelty validation...
   ✓ PubMed search: 23 methods
   ✓ Semantic Scholar validation
   ✓ Ranked by novelty + feasibility

📊 Top 5 Transfer Opportunities
─────────────────────────────────────────────────────────

1. ⭐⭐⭐⭐⭐ Diffusion Models (DDPM)
   Novelty: 78/100 | Feasibility: 85/100 | Priority: HIGH

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Source Domain: Computer Vision (Image Generation)
   Current Spatial Penetration: 7.2% (54/748 papers)

   🔧 Mechanism Match:
   ┌─────────────────────────────────────────────────────┐
   │ CV Domain          →    Spatial Omics Domain        │
   ├─────────────────────────────────────────────────────┤
   │ Pixel grid (H×W)   →    Spot grid (X×Y)            │
   │ RGB channels (3)   →    Gene channels (2K-20K)     │
   │ Missing pixels     →    Unmeasured spots           │
   │ Forward diffusion  →    Forward diffusion           │
   │ U-Net denoiser     →    U-Net denoiser (adapted)   │
   └─────────────────────────────────────────────────────┘

   📝 Concrete Hypothesis:
   Apply DDPMs to impute missing gene expression in sparse spatial
   transcriptomics data. Treat expression as an image where:
   - Pixels = spatial coordinates (x, y)
   - Channels = genes (2K-20K dimensions)
   - Training: noised Visium HD → denoise → original
   - Inference: sparse 10× Genomics → denoise → dense prediction

   📊 Validation:
   - Baseline: Tangram (optimal transport), SpaGE (graph NN)
   - Metrics: MSE, PCC, spatial autocorrelation preservation
   - Dataset: Mouse brain Visium HD (ground truth available)

   🔗 Implementation Resources:
   - Code: 3 repos indexed (diffusion-models, ddpm-pytorch)
   - Papers: 14 foundational + 3 spatial applications

   Prior Art Found: 3 papers (all 2024, limited scope)
   ✓ No comprehensive DDPM-based ST imputation exists

   [📋 Generate experiment card] [💾 Save to workspace] [🔍 Deep dive]

2. ⭐⭐⭐⭐☆ Image Inpainting (Partial Convolutions)
   Novelty: 72/100 | Feasibility: 90/100 | Priority: HIGH

   Source Domain: Computer Vision (Image Restoration)
   Current Spatial Penetration: 4.1% (31/748 papers)

   🔧 Mechanism: Masked convolutions for irregular holes
   📝 Hypothesis: Treat unmeasured spots as "holes" in expression image...

   [📋 Details] [💾 Save] [🔍 Deep dive]

...

Options:
  [1-5] View detailed transfer plan
  [e] Export all to markdown
  [g] Generate experiment cards
  [w] Save to research workspace
  [n] Next 5 opportunities
```

---

## Bridge Interface (Targeted Transfer)

```bash
$ polymath bridge \
    --source "geospatial_analysis" \
    --target "spatial_transcriptomics" \
    --show-mechanism

🌉 Building Cross-Domain Bridge
═══════════════════════════════════════════════════════════

Source: geospatial_analysis (1,247 papers)
Target: spatial_transcriptomics (748 papers)

🔍 Mechanism matching in progress...

╔═══════════════════════════════════════════════════════════╗
║         VALID TRANSFER FOUND: Optimal Transport           ║
╚═══════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 MECHANISM LAYER ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Common Mechanism: DISTRIBUTIONAL_MATCHING
├─ Implementation: Wasserstein distance minimization
├─ Data Structure: Point clouds with weights
└─ Objective: Minimal-cost transportation map

Geospatial Use Case:
┌─────────────────────────────────────────────────────────┐
│ Problem: Align satellite images across time points     │
│ Data:    GPS coordinates (x,y) + pixel intensities      │
│ Method:  Minimize transport cost between distributions │
│ Output:  Alignment transformation                       │
└─────────────────────────────────────────────────────────┘

Spatial Omics Mapping:
┌─────────────────────────────────────────────────────────┐
│ Problem: Align tissue sections across patients         │
│ Data:    Spot coordinates (x,y) + expression vectors    │
│ Method:  Minimize transport cost between distributions │
│ Output:  Section alignment transformation               │
└─────────────────────────────────────────────────────────┘

✓ Data structure match: Both use weighted 2D point clouds
✓ Mechanism compatibility: Distributional matching is domain-agnostic
✓ Evaluation transferable: Transport cost, alignment quality

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PRIOR ART CHECK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PubMed: "optimal transport" AND "spatial transcriptomics"
└─ 74 results (20 from last 2 years)
   ⚠️ Transfer already extensively studied

Semantic Scholar: Top implementations
├─ Tangram (Biancalani 2021) - 847 citations
├─ PASTE (Zeira 2022) - 234 citations
└─ SpaGE (Abdelaal 2020) - 156 citations

VERDICT: ⚠️ WELL-STUDIED (Novelty: 12/100)

While mechanistically valid, this transfer is not novel.
Consider focusing on unexplored variations:
- Unbalanced OT for batch effects
- Multi-marginal OT for 3D reconstruction
- Neural OT for learned cost functions

[📊 View all papers] [🔬 Explore variations] [❌ Reject]
```

---

## Ingest Interface (Adding Content)

```bash
$ polymath ingest --help

Usage: polymath ingest [OPTIONS] SOURCE

Add papers or code repositories to Polymath corpus.

RECOMMENDED WORKFLOW (Zotero-first):
  1. Add items to Zotero (extracts metadata automatically)
  2. Export Zotero library to CSV
  3. Run: polymath ingest zotero-export.csv
  4. PDFs are linked via DOI, metadata is guaranteed accurate

Options:
  --type TYPE         Source type: pdf, repo, zotero, doi
  --enhanced-parser   Use GROBID + fallback parsing (recommended)
  --validate         Validate metadata before ingestion
  --batch           Process multiple items

Examples:
  # Ingest from Zotero export (RECOMMENDED)
  polymath ingest zotero_library.csv --type zotero

  # Single paper with enhanced parser
  polymath ingest paper.pdf --enhanced-parser

  # Code repository
  polymath ingest /path/to/repo --type repo

  # Batch from directory
  polymath ingest /downloads/*.pdf --batch --validate
```

### Zotero Ingestion Flow

```bash
$ polymath ingest ~/Downloads/My_Library.csv --type zotero

📚 Zotero Import Workflow
═══════════════════════════════════════════════════════════

Step 1: Validating CSV export...
✓ Found 3,175 items
✓ Schema validation passed
✓ 2,847 items have DOIs (89.7%)
✓ 1,387 PDFs attached

Step 2: Deduplication check...
✓ 2,648 items already in database (matched by DOI/title_hash)
✓ 527 new items to ingest

Step 3: Priority sorting...
High priority (faculty match + topic relevance): 124 items
Medium priority: 289 items
Low priority: 114 items

Proceed with high-priority ingestion? [Y/n]: y

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 INGESTING HIGH-PRIORITY ITEMS (124 papers)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[1/124] Processing: "Deep learning for spatial transcriptomics"
        Authors: Chen et al. (2023)
        DOI: 10.1038/s41592-023-01234-x

        ✓ Metadata loaded from Zotero
        ✓ PDF found at: ~/Zotero/storage/ABC123/chen_2023.pdf
        ✓ GROBID extraction: 47 passages (23,450 chars)
        ✓ Document-level concepts: 12 methods, 8 problems, 3 mechanisms
        ✓ BGE-M3 embedding: 47 passages → ChromaDB
        ✓ Postgres sync complete

        Status: ✓ COMPLETE (doc_id: a7f3d...)

[2/124] Processing: "Optimal transport for single-cell analysis"
        ...

Progress: ████████░░░░░░░░ 8/124 (6.5%) | ETA: 14 minutes

Options: [p] Pause [s] Skip current [q] Quit and save progress
```

---

## Validation Interface (Hypothesis Checking)

```bash
$ polymath validate "Use DDPMs to impute missing spots in Visium data"

🔬 Hypothesis Validation Engine
═══════════════════════════════════════════════════════════

Hypothesis: "Use DDPMs to impute missing spots in Visium data"

Phase 1: Parsing hypothesis...
✓ Method: Denoising Diffusion Probabilistic Models (DDPM)
✓ Problem: Missing data imputation
✓ Domain: Spatial transcriptomics (Visium)

Phase 2: Literature validation...

┌─────────────────────────────────────────────────────────┐
│ Prior Art Search Results                                │
├─────────────────────────────────────────────────────────┤
│ PubMed: "DDPM" OR "denoising diffusion"                 │
│         AND "spatial transcriptomics"                    │
│ └─ 3 results (all 2024)                                 │
│                                                          │
│ Semantic Scholar: diffusion models + spatial omics      │
│ └─ 7 results                                            │
│                                                          │
│ Polymath Corpus: Local evidence                         │
│ └─ 14 passages mentioning DDPMs in ST context           │
└─────────────────────────────────────────────────────────┘

Phase 3: Mechanism validation...

✓ Data structure match:
  DDPM operates on: 2D grids with channel dimensions
  Visium provides: 2D spot arrays with gene expression vectors
  Compatibility: HIGH (same structure, different semantics)

✓ Method transferability:
  Source domains: Computer vision, audio synthesis
  Common mechanism: Iterative denoising via score matching
  Transfer precedent: Yes (DDPMs applied to tabular, graph data)

Phase 4: Feasibility assessment...

Resources Available:
├─ Code: 3 DDPM implementations in corpus
├─ Baselines: 12 ST imputation methods for comparison
├─ Data: Visium HD datasets with ground truth
└─ Compute: RTX 5090 24GB (sufficient for training)

Estimated Effort: 3-4 weeks full-time
├─ Week 1: Adapt DDPM architecture to gene expression
├─ Week 2: Training + hyperparameter tuning
├─ Week 3: Baseline comparisons
└─ Week 4: Validation on held-out datasets

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 VERDICT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Novelty Score: 72/100
├─ Novel application (few prior works)
├─ Mechanistically sound (data structure match)
└─ Feasible with available resources

Recommendation: ✅ PROCEED

Suggested next steps:
1. Generate detailed experiment card
2. Identify closest baseline (likely Tangram)
3. Acquire Visium HD training data
4. Prototype minimal DDPM architecture

[📋 Generate experiment card] [🔍 Deep dive into prior art]
[💾 Save to research workspace] [🚀 Begin prototyping]
```

---

## Stats Interface

```bash
$ polymath stats

📊 Polymath 2.0 System Status
═══════════════════════════════════════════════════════════

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 CORPUS STATISTICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Documents:          3,617 papers
Passages:           748,325 (avg 206 per paper)
Concepts:           4,818,574 extracted
Code Chunks:        576,818 (from 26 repos)

Metadata Quality:
├─ DOI coverage:    695/3,617 (19.2%) ⚠️
├─ PMID coverage:   33/3,617 (0.9%)   ⚠️
├─ Year coverage:   3,570/3,617 (98.7%) ✓
└─ Authors:         3,562/3,617 (98.5%) ✓

Recent Activity:
├─ Last ingestion:  2026-01-12 18:47:23
├─ Last search:     2026-01-13 09:14:11
└─ Last validation: 2026-01-13 08:52:03

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 DATABASE STATUS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Postgres:           ✓ HEALTHY
├─ Connection:      localhost:5432
├─ Database:        polymath (v2.0.0-alpha)
├─ Tables:          documents, passages, passage_concepts,
│                   code_chunks, chunk_concepts
└─ Disk usage:      14.2 GB

ChromaDB:           ✓ HEALTHY
├─ Location:        /home/user/polymath-2.0/chromadb/
├─ Collections:     polymath_passages (748K),
│                   polymath_code (576K)
├─ Embedding model: BAAI/bge-m3 (1024-dim)
└─ Disk usage:      22.7 GB

Neo4j:              ✓ HEALTHY
├─ Connection:      bolt://localhost:7687
├─ Nodes:           1.2M (Paper: 32K, Passage: 748K,
│                   Concept: 487K, Mechanism: 12K)
├─ Relationships:   2.8M edges
└─ Disk usage:      8.9 GB

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 SYSTEM HEALTH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Disk Space:
├─ Root (/)         127.4 GB / 500 GB (25.5%) ✓
├─ /scratch         892 GB / 2 TB (43.5%)     ✓
└─ /datafast        1.2 TB / 4 TB (30%)       ✓

Services:
├─ Postgres         ✓ Running (PID 1247)
├─ Neo4j            ✓ Running (PID 1893)
└─ API server       ✓ Running (port 8000)

Background Jobs:
├─ Concept backfill: 0 active
├─ Embeddings sync:  0 active
└─ Neo4j sync:       0 active

Warnings:
⚠️ 80.8% of documents missing DOI
⚠️ 22K passages may need vector re-sync
ℹ️ Consider running metadata enrichment

[🔧 Run diagnostics] [🧹 Cleanup] [📈 Detailed metrics]
```

---

## Agentic Workflow Interface

```bash
$ polymath discover --target "spatial_transcriptomics" \
    --agentic \
    --checkpoint

🤖 Agentic Discovery Mode
═══════════════════════════════════════════════════════════

Checkpoint mode enabled - you'll review each step before proceeding.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 STEP 1: PERCEPTION (Understanding the Target Domain)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Agent analyzing spatial_transcriptomics domain...

Domain Characterization:
┌─────────────────────────────────────────────────────────┐
│ Core Problem: Measure gene expression with spatial      │
│               context in tissue sections                 │
│                                                          │
│ Data Modality:                                           │
│ ├─ Input: H&E tissue images (RGB, 2D)                   │
│ ├─ Output: Expression matrix (spots × genes)            │
│ └─ Structure: Irregular 2D point cloud + vectors        │
│                                                          │
│ Current Methods (from corpus):                           │
│ ├─ Optimal transport: 74 papers                         │
│ ├─ Graph neural networks: 52 papers                     │
│ ├─ Deconvolution: 38 papers                             │
│ └─ Image-to-expression models: 29 papers                │
│                                                          │
│ Open Challenges:                                         │
│ ├─ Limited resolution (55μm spots)                      │
│ ├─ High costs ($1000+ per sample)                       │
│ ├─ Batch effects across platforms                       │
│ └─ Missing data imputation                              │
└─────────────────────────────────────────────────────────┘

✓ Domain understanding complete (142 relevant papers analyzed)

Proceed to STEP 2: Internalization? [Y/n]: y

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 STEP 2: INTERNALIZATION (Extracting Transferable Patterns)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Agent building mechanism taxonomy...

Mechanism Families Detected:
┌─────────────────────────────────────────────────────────┐
│ 1. DISTRIBUTIONAL_MATCHING                              │
│    ├─ Instances: 74                                     │
│    ├─ Data: Point clouds with weights                   │
│    └─ Examples: Optimal transport, Wasserstein          │
│                                                          │
│ 2. GRAPH_MESSAGE_PASSING                                │
│    ├─ Instances: 52                                     │
│    ├─ Data: Graphs with node features                   │
│    └─ Examples: GCN, GAT, GraphSAGE                     │
│                                                          │
│ 3. IMAGE_TO_STRUCTURED                                   │
│    ├─ Instances: 29                                     │
│    ├─ Data: Images → structured outputs                 │
│    └─ Examples: CNNs, Vision Transformers               │
│                                                          │
│ 4. ITERATIVE_DENOISING                                   │
│    ├─ Instances: 14 (LOW COVERAGE!)                     │
│    ├─ Data: Corrupted → clean via gradual refinement    │
│    └─ Examples: DDPMs, score matching                   │
│                                                          │
│ 5. SPARSE_RECOVERY                                       │
│    ├─ Instances: 8 (LOW COVERAGE!)                      │
│    ├─ Data: Undersampled → full reconstruction          │
│    └─ Examples: Compressed sensing, L1 minimization     │
└─────────────────────────────────────────────────────────┘

⚠️ Families 4 and 5 have low coverage but compatible data structures!

Proceed to STEP 3: Reasoning? [Y/n]: y

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 STEP 3: REASONING (Cross-Domain Transfer Identification)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Agent querying mechanism graph for transfers...

Transfer Hypothesis 1: ITERATIVE_DENOISING
┌─────────────────────────────────────────────────────────┐
│ Source: Computer Vision (Image Generation)              │
│ Method: Denoising Diffusion Probabilistic Models        │
│                                                          │
│ Why This Transfer Makes Sense:                          │
│ ├─ ✓ Data structure match                               │
│ │   CV: 2D pixel grid (H×W×C)                           │
│ │   ST: 2D spot grid (X×Y×G)                            │
│ │                                                        │
│ ├─ ✓ Problem alignment                                  │
│ │   CV: Generate missing/corrupted pixels               │
│ │   ST: Impute unmeasured gene expression              │
│ │                                                        │
│ ├─ ✓ Mechanism transferability                          │
│ │   Both: Learn score function ∇log p(x)               │
│ │   Both: Iterative refinement x_t → x_0               │
│ │                                                        │
│ └─ ✓ Computational feasibility                          │
│     U-Net architecture scales to 2K-20K gene channels   │
└─────────────────────────────────────────────────────────┘

Grounding Evidence:
├─ [Paper 1] "Denoising Diffusion Probabilistic Models"
│   Ho et al., NeurIPS 2020
│   DOI: 10.48550/arXiv.2006.11239
│   → "The reverse process p_θ(x_{t-1}|x_t) gradually denoises..."
│
├─ [Paper 2] "STDiff: Spatial Transcriptomics via Diffusion"
│   Zhang et al., bioRxiv 2024
│   DOI: 10.1101/2024.01.12.575123
│   → "We adapt DDPM to gene expression imputation..."
│
└─ [Code] diffusion-models/ddpm_pytorch
    GitHub: hojonathanho/diffusion
    → U-Net implementation ready for adaptation

Novelty: 78/100 (only 3 prior ST applications)
Feasibility: 85/100 (architecture proven, data available)

═══════════════════════════════════════════════════════════

Continue generating hypotheses? [Y/n/s=save current]: s

💾 Saving workspace...

Saved to: /home/user/polymath-2.0/workspaces/discovery_20260113_0914/
├─ domain_analysis.json
├─ mechanism_taxonomy.json
├─ hypothesis_01_ddpm.json (includes all grounding evidence)
└─ session_transcript.md

Resume later with: polymath resume discovery_20260113_0914
```

---

## Research Workspace

```bash
$ ls ~/polymath-2.0/workspaces/discovery_20260113_0914/

hypothesis_01_ddpm/
├─ hypothesis.json           # Structured hypothesis
├─ evidence_spans.json       # Source passages with DOIs
├─ mechanism_analysis.json   # Data structure mapping
├─ experiment_card.md        # Ready-to-execute plan
└─ baseline_comparisons.csv  # Identified baselines

$ polymath workspace view hypothesis_01_ddpm

╔═══════════════════════════════════════════════════════════╗
║  EXPERIMENT CARD: DDPMs for Spatial Imputation            ║
╚═══════════════════════════════════════════════════════════╝

Hypothesis:
  Apply Denoising Diffusion Probabilistic Models to impute
  missing gene expression in spatially resolved transcriptomics

Mechanism:
  Iterative denoising via learned reverse diffusion process
  p_θ(x_{t-1}|x_t) from noise schedule β_1...β_T

Data Objects:
  ├─ Input: Sparse expression matrix E ∈ R^{n×g}
  │         n = measured spots, g = genes
  ├─ Target: Dense expression E_full ∈ R^{N×g}
  │          N = all spatial locations
  └─ Training: Visium HD data (55μm → downsampled as oracle)

Method Implementation:
  ├─ Architecture: U-Net with self-attention
  ├─ Forward process: q(x_t|x_{t-1}) = N(√(1-β_t)x_{t-1}, β_t I)
  ├─ Reverse process: p_θ(x_{t-1}|x_t) = N(μ_θ(x_t,t), Σ_θ(x_t,t))
  └─ Training: L = E_t[||ε - ε_θ(x_t,t)||²]

Baselines:
  ├─ Tangram (optimal transport)        - 847 citations
  ├─ SpaGE (graph smoothing)            - 156 citations
  └─ ENACT (optimal transport + GNN)    - Novel 2024

Evaluation:
  ├─ Metrics: MSE, PCC, spatial autocorrelation (Moran's I)
  ├─ Dataset: Mouse brain Visium HD (10× Genomics)
  └─ Validation: Cross-validation + held-out tissues

Compute Requirements:
  ├─ Training: RTX 5090 24GB (48 hours estimated)
  ├─ Memory: <20GB for typical Visium dataset
  └─ Inference: <1min per sample

Timeline:
  Week 1: Adapt DDPM to gene expression (U-Net modification)
  Week 2: Training + hyperparameter tuning
  Week 3: Baseline comparisons
  Week 4: Biological validation + writeup

Grounding Citations:
  [1] Ho et al. (2020) - DDPM foundations
  [2] Zhang et al. (2024) - STDiff (partial implementation)
  [3] Biancalani et al. (2021) - Tangram baseline

[🚀 Begin prototyping] [📝 Export to LaTeX] [💾 Save]
```

---

## Key TUI Principles

1. **Document Grounding**: Every claim links to source with DOI
2. **Mechanism Transparency**: Show HOW methods work, not just labels
3. **Progressive Disclosure**: Simple by default, detailed on demand
4. **Agentic Checkpoints**: Human-in-the-loop for critical decisions
5. **Reproducibility**: All workflows save structured JSON + markdown

---

## Implementation Notes

The TUI is built with:
- `rich` library for styled terminal output
- `questionary` for interactive prompts
- `typer` for CLI routing
- PostgreSQL + ChromaDB + Neo4j backends
- Gemini Flash for agentic workflows (with structured JSON output)

All outputs include:
- Source DOIs (clickable in terminal emulators)
- Export to markdown/LaTeX
- Resumable sessions via workspace files
