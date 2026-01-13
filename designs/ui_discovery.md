# Discovery Interface Design

**Author**: Max Van Belkum
**Version**: 2.0.0-alpha

---

## Gap Detection Interface

```
╔══════════════════════════════════════════════════════════════════════════╗
║                     POLYMATH 2.0 - CROSS-DOMAIN DISCOVERY                 ║
╚══════════════════════════════════════════════════════════════════════════╝

> polymath discover --target-domain "spatial_transcriptomics"

Analyzing knowledge graph... ████████████████████ 100%
Checking novelty... ████████████████████ 100%

┌─────────────────────────────────────────────────────────────────────────┐
│ TRANSFER OPPORTUNITIES                                                  │
│ Target Domain: spatial_transcriptomics                                  │
│ Found: 15 potential transfers | Validated: 8 | Novel: 5                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ [1] ★★★★★ OPTIMAL TRANSPORT → SPATIAL GENE IMPUTATION                  │
│     ────────────────────────────────────────────────────────────────    │
│     Source Domain: geospatial_analysis                                  │
│     Mechanism: distributional_matching                                  │
│     Operates On: weighted_point_cloud                                   │
│     Spatial Penetration: 2.3% (37 / 1,592 papers)                       │
│     Novelty Score: 78/100 ████████░░                                    │
│                                                                         │
│     WHY IT TRANSFERS:                                                   │
│     • Same data structure: Visium spots ≈ GPS coordinates               │
│     • Same objective: Minimize distribution divergence                  │
│     • Proven in: batch correction, domain adaptation                    │
│                                                                         │
│     EXISTING WORK (3 papers):                                           │
│     • CellOT (2021) - single-cell trajectory inference                  │
│     • scOT (2022) - batch correction                                    │
│     • SpatialOT (2023) - section alignment                              │
│                                                                         │
│     GAP: No application to gene imputation from H&E                     │
│                                                                         │
│     [h]ypothesis | [p]apers | [m]echanism | [v]alidate                  │
│                                                                         │
│ ─────────────────────────────────────────────────────────────────────── │
│                                                                         │
│ [2] ★★★★☆ GRAPH WAVELETS → SPATIAL FEATURE AGGREGATION                  │
│     ────────────────────────────────────────────────────────────────    │
│     Source Domain: signal_processing                                    │
│     Mechanism: multi_scale_decomposition                                │
│     Operates On: graph_structure                                        │
│     Spatial Penetration: 0.5% (8 / 1,592 papers)                        │
│     Novelty Score: 92/100 █████████░                                    │
│                                                                         │
│     WHY IT TRANSFERS:                                                   │
│     • Tissue = graph (cells as nodes, proximity as edges)               │
│     • Wavelets capture multi-scale spatial patterns                     │
│     • Proven in: molecular graphs, social networks                      │
│                                                                         │
│     [h]ypothesis | [p]apers | [m]echanism | [v]alidate                  │
│                                                                         │
│ [3] ★★★★☆ POINT PROCESS MODELS → CELL DISTRIBUTION MODELING             │
│     ...                                                                 │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│ [n]ext page | [f]ilter | [e]xport | [q]uit                              │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Hypothesis Generation Interface

```
> h 1

┌─────────────────────────────────────────────────────────────────────────┐
│ HYPOTHESIS GENERATOR                                                    │
│ Transfer: optimal_transport → spatial_gene_imputation                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ ═══════════════════════════════════════════════════════════════════════ │
│ GENERATED HYPOTHESIS                                                    │
│ ═══════════════════════════════════════════════════════════════════════ │
│                                                                         │
│ TITLE:                                                                  │
│ Optimal Transport for H&E-Guided Spatial Gene Expression Imputation    │
│                                                                         │
│ ABSTRACT:                                                               │
│ We propose applying optimal transport (OT) methods, originally          │
│ developed for geospatial point cloud registration, to the problem       │
│ of predicting spatial gene expression from H&E histology images.        │
│ The key insight is that both problems involve aligning probability      │
│ distributions over weighted point sets—in the source domain, GPS        │
│ coordinates with elevation/intensity weights; in the target domain,     │
│ Visium spot coordinates with gene expression weights.                   │
│                                                                         │
│ MECHANISM JUSTIFICATION:                                                │
│ Optimal transport implements distributional matching, which operates    │
│ on weighted point clouds to minimize transport cost. In spatial         │
│ transcriptomics:                                                        │
│ - Points = spot coordinates (x, y)                                      │
│ - Weights = total UMI counts or specific gene expression                │
│ - Objective = align expression distributions across sections/samples    │
│                                                                         │
│ PROPOSED APPROACH:                                                      │
│ 1. Extract morphological features from H&E using pre-trained CNN        │
│ 2. Formulate gene prediction as OT from morphology → expression space   │
│ 3. Use Sinkhorn algorithm for efficient approximate OT                  │
│ 4. Regularize with entropic penalty to handle noise                     │
│                                                                         │
│ EXPECTED OUTCOMES:                                                      │
│ - Improved imputation for sparse Visium data                            │
│ - Better handling of batch effects via OT alignment                     │
│ - Interpretable transport plans showing morphology-gene relationships   │
│                                                                         │
│ EVALUATION METRICS:                                                     │
│ - Pearson correlation with ground truth (held-out genes)                │
│ - MSE on imputed expression values                                      │
│ - Biological validity (marker gene patterns)                            │
│                                                                         │
│ KEY CITATIONS:                                                          │
│ [1] Peyré & Cuturi (2019) - Computational Optimal Transport             │
│ [2] Schiebinger et al. (2019) - Optimal-transport analysis (Waddington)│
│ [3] He et al. (2020) - Img2ST (baseline comparison)                     │
│                                                                         │
│ NOVELTY ASSESSMENT:                                                     │
│ ├── PubMed: 2 related papers (not exact match)                          │
│ ├── Semantic Scholar: 3 related papers                                  │
│ ├── Internal corpus: 5 related passages                                 │
│ └── Overall: MODERATELY NOVEL (score: 78/100)                           │
│                                                                         │
│ CONFIDENCE: HIGH                                                        │
│ Mechanism compatibility verified. Data structure match confirmed.       │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│ [e]dit | [s]ave | [x]port LaTeX | [r]egenerate | [←] back               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Transfer Validation Interface

```
> v 1

┌─────────────────────────────────────────────────────────────────────────┐
│ TRANSFER VALIDATION                                                     │
│ Method: optimal_transport | Target: spatial_gene_imputation             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ ═══════════════════════════════════════════════════════════════════════ │
│ MECHANISM COMPATIBILITY                                         [PASS]  │
│ ═══════════════════════════════════════════════════════════════════════ │
│                                                                         │
│ Mechanism: distributional_matching                                      │
│                                                                         │
│ Source Domain (geospatial):                                             │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ Data: GPS coordinates with elevation/intensity weights               │ │
│ │ Format: {(x_i, y_i, w_i)} where w_i ∈ R+                             │ │
│ │ Objective: Minimize Σ T_ij * d(p_i, q_j)                             │ │
│ │ Constraints: Mass conservation                                       │ │
│ └─────────────────────────────────────────────────────────────────────┘ │
│                           ↕ COMPATIBLE ↕                                │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ Data: Visium spot coordinates with expression weights                │ │
│ │ Format: {(x_i, y_i, g_i)} where g_i ∈ R^d (d genes)                  │ │
│ │ Objective: Minimize Σ T_ij * ||h_i - g_j||                           │ │
│ │ Constraints: Mass conservation                                       │ │
│ └─────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│ Compatibility Score: 0.92 █████████░                                    │
│                                                                         │
│ ═══════════════════════════════════════════════════════════════════════ │
│ NOVELTY CHECK                                                   [PASS]  │
│ ═══════════════════════════════════════════════════════════════════════ │
│                                                                         │
│ Query: "optimal transport" AND "spatial transcriptomics" AND            │
│        "gene imputation"                                                │
│                                                                         │
│ Results:                                                                │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ Source          │ Exact Match │ Related │ Score                     │ │
│ ├─────────────────┼─────────────┼─────────┼───────────────────────────┤ │
│ │ PubMed          │     0       │    2    │ 0.85                      │ │
│ │ Semantic Scholar│     0       │    3    │ 0.82                      │ │
│ │ Internal corpus │     0       │    5    │ 0.78                      │ │
│ │ arXiv/bioRxiv   │     1       │    2    │ 0.65                      │ │
│ └─────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│ Novelty Score: 78/100 (MODERATELY NOVEL)                                │
│ Note: One preprint (bioRxiv 2024) partially addresses this.             │
│                                                                         │
│ ═══════════════════════════════════════════════════════════════════════ │
│ FEASIBILITY CHECK                                               [PASS]  │
│ ═══════════════════════════════════════════════════════════════════════ │
│                                                                         │
│ Required Resources:                                                     │
│ ├── GPU: Yes (RTX 5090 sufficient)                                      │
│ ├── Data: Visium datasets (available via GEO)                           │
│ ├── Libraries: POT (Python Optimal Transport) - MIT license             │
│ └── Time estimate: 2-4 weeks for proof-of-concept                       │
│                                                                         │
│ Implementation Path:                                                    │
│ 1. Adapt POT library for spatial data structure                         │
│ 2. Use pre-trained feature extractor (UNI/HIPT)                         │
│ 3. Benchmark against Img2ST, HisToGene                                  │
│                                                                         │
│ ═══════════════════════════════════════════════════════════════════════ │
│ VALIDATION SUMMARY                                                      │
│ ═══════════════════════════════════════════════════════════════════════ │
│                                                                         │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ ✓ Mechanism Compatible    │ Score: 0.92                             │ │
│ │ ✓ Novel Enough            │ Score: 0.78                             │ │
│ │ ✓ Feasible                │ Resources: Available                    │ │
│ │ ─────────────────────────────────────────────────────────────────── │ │
│ │ OVERALL: RECOMMENDED FOR INVESTIGATION                              │ │
│ │ Priority: HIGH                                                      │ │
│ └─────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│ [h]ypothesis | [e]xperiment spec | [c]ompare prior art | [←] back       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Experiment Specification Interface

```
> e

┌─────────────────────────────────────────────────────────────────────────┐
│ EXPERIMENT SPECIFICATION                                                │
│ Hypothesis: OT for H&E-Guided Spatial Gene Imputation                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ ═══════════════════════════════════════════════════════════════════════ │
│ DATA REQUIREMENTS                                                       │
│ ═══════════════════════════════════════════════════════════════════════ │
│                                                                         │
│ Training Data:                                                          │
│ ├── 10x Visium datasets with paired H&E (n ≥ 20 samples)                │
│ ├── Gene expression matrices (filtered, normalized)                     │
│ ├── H&E images (full resolution, registered)                            │
│ └── Spot coordinates (tissue_positions_list.csv)                        │
│                                                                         │
│ Recommended Datasets:                                                   │
│ [1] Human Breast Cancer (10x Genomics) - 4 samples                      │
│ [2] Mouse Brain (10x Genomics) - 2 samples                              │
│ [3] Human Colorectal Cancer (GSE280318) - 3 samples                     │
│ [4] Spatialomics Atlas (various) - 15+ samples                          │
│                                                                         │
│ ═══════════════════════════════════════════════════════════════════════ │
│ IMPLEMENTATION STEPS                                                    │
│ ═══════════════════════════════════════════════════════════════════════ │
│                                                                         │
│ Week 1: Data Preparation                                                │
│ □ Download datasets from GEO/10x                                        │
│ □ Preprocess H&E images (tile, normalize)                               │
│ □ Extract spot-level features using UNI encoder                         │
│ □ Split into train/val/test (70/15/15)                                  │
│                                                                         │
│ Week 2: OT Implementation                                               │
│ □ Set up POT library environment                                        │
│ □ Implement spatial OT loss function                                    │
│ □ Design cost matrix (morphology → expression)                          │
│ □ Add entropic regularization                                           │
│                                                                         │
│ Week 3: Training & Tuning                                               │
│ □ Train baseline model (no OT)                                          │
│ □ Train OT-augmented model                                              │
│ □ Hyperparameter search (λ_ot, ε_entropy)                               │
│ □ Cross-validation                                                      │
│                                                                         │
│ Week 4: Evaluation                                                      │
│ □ Compare to baselines (Img2ST, HisToGene)                              │
│ □ Compute metrics (Pearson r, MSE, SSIM)                                │
│ □ Biological validation (marker genes)                                  │
│ □ Ablation studies                                                      │
│                                                                         │
│ ═══════════════════════════════════════════════════════════════════════ │
│ EVALUATION PROTOCOL                                                     │
│ ═══════════════════════════════════════════════════════════════════════ │
│                                                                         │
│ Primary Metrics:                                                        │
│ ├── Pearson correlation (per-gene, averaged)                            │
│ ├── MSE on held-out spots                                               │
│ └── SSIM on spatial expression patterns                                 │
│                                                                         │
│ Secondary Metrics:                                                      │
│ ├── Marker gene F1 score                                                │
│ ├── Cell type prediction accuracy                                       │
│ └── Computational cost (FLOPs, runtime)                                 │
│                                                                         │
│ Baselines:                                                              │
│ ├── Img2ST (He et al., 2020)                                            │
│ ├── HisToGene (Pang et al., 2021)                                       │
│ ├── ST-Net (Zhao et al., 2022)                                          │
│ └── Mean expression (naive baseline)                                    │
│                                                                         │
│ ═══════════════════════════════════════════════════════════════════════ │
│ CODE TEMPLATE                                                           │
│ ═══════════════════════════════════════════════════════════════════════ │
│                                                                         │
│ ```python                                                               │
│ import ot                                                               │
│ import torch                                                            │
│                                                                         │
│ def spatial_ot_loss(pred_expr, true_expr, coords, eps=0.1):             │
│     """                                                                 │
│     Compute optimal transport loss between predicted and true           │
│     gene expression distributions.                                      │
│                                                                         │
│     Args:                                                               │
│         pred_expr: (n_spots, n_genes) predicted expression              │
│         true_expr: (n_spots, n_genes) ground truth expression           │
│         coords: (n_spots, 2) spatial coordinates                        │
│         eps: entropic regularization                                    │
│     """                                                                 │
│     # Compute cost matrix based on expression distance                  │
│     C = torch.cdist(pred_expr, true_expr, p=2)                          │
│                                                                         │
│     # Uniform marginals (all spots have equal weight)                   │
│     a = torch.ones(n_spots) / n_spots                                   │
│     b = torch.ones(n_spots) / n_spots                                   │
│                                                                         │
│     # Solve OT with Sinkhorn                                            │
│     T = ot.sinkhorn(a, b, C, eps)                                       │
│                                                                         │
│     # Compute Wasserstein distance                                      │
│     return torch.sum(T * C)                                             │
│ ```                                                                     │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│ [s]ave spec | [x]port markdown | [g]enerate code skeleton | [←] back    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Discovery Dashboard

```
╔══════════════════════════════════════════════════════════════════════════╗
║                     POLYMATH 2.0 - DISCOVERY DASHBOARD                    ║
╚══════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────┐
│ ACTIVE HYPOTHESES                                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ Status: ●Active (3)  ○Investigating (2)  ◐Completed (5)  ○Rejected (1)  │
│                                                                         │
│ ┌───────────────────────────────────────────────────────────────────┐   │
│ │ ID   │ Title                              │ Priority │ Status     │   │
│ ├──────┼────────────────────────────────────┼──────────┼────────────┤   │
│ │ H-01 │ OT for H&E Gene Imputation         │ ★★★★★    │ ●Active    │   │
│ │ H-02 │ Graph Wavelets for ST              │ ★★★★☆    │ ●Active    │   │
│ │ H-03 │ Point Process Cell Modeling        │ ★★★★☆    │ ●Active    │   │
│ │ H-04 │ Attention-based Spot Aggregation   │ ★★★☆☆    │ ○Investigating│ │
│ │ H-05 │ Contrastive Learning for ST        │ ★★★☆☆    │ ○Investigating│ │
│ └───────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│ [1-5] view details | [n]ew search | [r]efresh | [e]xport                │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ DOMAIN COVERAGE                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ spatial_transcriptomics  ████████████████████░░░░░░░░░░  65% explored   │
│ single_cell              ██████████████████████████░░░░  82% explored   │
│ computational_pathology  ████████████████████████░░░░░░  78% explored   │
│ computer_vision          ████████░░░░░░░░░░░░░░░░░░░░░░  25% explored   │
│ geospatial               ████░░░░░░░░░░░░░░░░░░░░░░░░░░  12% explored   │
│ operations_research      ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░   5% explored   │
│                                                                         │
│ Highest potential: geospatial → spatial_transcriptomics                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```
