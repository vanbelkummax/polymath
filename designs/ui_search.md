# Search Interface Design

**Author**: Max Van Belkum
**Version**: 2.0.0-alpha

---

## CLI Search Interface

### Basic Search

```
╔══════════════════════════════════════════════════════════════════════════╗
║                         POLYMATH 2.0 - SEARCH                             ║
╚══════════════════════════════════════════════════════════════════════════╝

> polymath search "optimal transport spatial transcriptomics"

Searching... ████████████████████ 100%

┌─────────────────────────────────────────────────────────────────────────┐
│ RESULTS (20 passages, 0.8s)                                  [RRF Hybrid]│
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ [1] ★★★★★ (0.94) Optimal Transport for Single-Cell Genomics            │
│     DOI: 10.1038/s41592-022-01234-5 | Year: 2022 | Venue: Nat Methods   │
│     Authors: Zhang et al.                                               │
│     Section: Methods > Transport Optimization                           │
│     ─────────────────────────────────────────────────────────────────   │
│     "We formulate the spatial gene expression imputation problem as     │
│     an optimal transport task, where each spatial spot is treated as    │
│     a weighted point in a high-dimensional gene expression space.       │
│     The Wasserstein distance provides a natural metric for comparing    │
│     expression distributions across tissue sections..."                 │
│     ─────────────────────────────────────────────────────────────────   │
│     [e]xpand context | [c]ite | [g]raph | [s]imilar                     │
│                                                                         │
│ [2] ★★★★☆ (0.87) Distributional Matching for Batch Correction           │
│     DOI: 10.1093/bioinformatics/btac456 | Year: 2023                    │
│     Authors: Chen et al.                                                │
│     Section: Introduction > Related Work                                │
│     ─────────────────────────────────────────────────────────────────   │
│     "Optimal transport-based methods have shown promise in aligning     │
│     single-cell distributions across batches, leveraging the            │
│     geometric structure of the data manifold..."                        │
│     ─────────────────────────────────────────────────────────────────   │
│     [e]xpand context | [c]ite | [g]raph | [s]imilar                     │
│                                                                         │
│ [3] ★★★★☆ (0.85) Point Cloud Registration in Spatial Omics              │
│     ...                                                                 │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│ [n]ext page | [f]ilter | [e]xport | [q]uit                              │
└─────────────────────────────────────────────────────────────────────────┘

>
```

### Context Expansion

```
> e 1

┌─────────────────────────────────────────────────────────────────────────┐
│ EXPANDED CONTEXT - Result [1]                                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ 📄 DOCUMENT: Optimal Transport for Single-Cell Genomics                 │
│    DOI: 10.1038/s41592-022-01234-5                                      │
│                                                                         │
│ 📑 SECTION HIERARCHY:                                                   │
│    └── Methods                                                          │
│        └── Transport Optimization  ◀── YOU ARE HERE                     │
│            ├── Problem Formulation                                      │
│            ├── Sinkhorn Algorithm                                       │
│            └── Regularization                                           │
│                                                                         │
│ ═══════════════════════════════════════════════════════════════════════ │
│                                                                         │
│ PARENT SECTION (Methods):                                               │
│ ─────────────────────────────────────────────────────────────────────── │
│ "Our computational pipeline consists of three main components:          │
│ (1) preprocessing and quality control, (2) spatial feature              │
│ extraction using a graph neural network, and (3) optimal transport      │
│ for distribution alignment. We describe each component below..."        │
│                                                                         │
│ ═══════════════════════════════════════════════════════════════════════ │
│                                                                         │
│ CURRENT PASSAGE (Transport Optimization):                               │
│ ─────────────────────────────────────────────────────────────────────── │
│ "We formulate the spatial gene expression imputation problem as         │
│ an optimal transport task, where each spatial spot is treated as        │
│ a weighted point in a high-dimensional gene expression space.           │
│ The Wasserstein distance provides a natural metric for comparing        │
│ expression distributions across tissue sections.                        │
│                                                                         │
│ Formally, given source distribution μ and target distribution ν,        │
│ we seek the transport plan T* that minimizes:                           │
│                                                                         │
│     T* = argmin_T Σ_{i,j} T_{ij} · C_{ij}                               │
│                                                                         │
│ subject to the marginal constraints T·1 = μ and T^T·1 = ν."             │
│                                                                         │
│ ═══════════════════════════════════════════════════════════════════════ │
│                                                                         │
│ CHILD SECTIONS:                                                         │
│ ─────────────────────────────────────────────────────────────────────── │
│ [a] Problem Formulation: "The cost matrix C is computed using..."       │
│ [b] Sinkhorn Algorithm: "For computational efficiency, we use..."       │
│ [c] Regularization: "To prevent overfitting, we add entropic..."        │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│ [a-c] drill down | [↑] parent | [←] back | [c]ite | [g]raph             │
└─────────────────────────────────────────────────────────────────────────┘
```

### Graph View

```
> g 1

┌─────────────────────────────────────────────────────────────────────────┐
│ KNOWLEDGE GRAPH - Optimal Transport                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│                    ┌─────────────────────┐                              │
│                    │  optimal_transport  │                              │
│                    │      (METHOD)       │                              │
│                    └──────────┬──────────┘                              │
│                               │ IMPLEMENTS                              │
│                               ▼                                         │
│                    ┌─────────────────────┐                              │
│                    │distributional_match │                              │
│                    │    (MECHANISM)      │                              │
│                    └──────────┬──────────┘                              │
│                    ┌──────────┼──────────┐                              │
│          OPERATES_ON│                    │OPTIMIZES                     │
│                    ▼                     ▼                              │
│         ┌─────────────────┐   ┌─────────────────┐                       │
│         │weighted_point   │   │minimize_transport│                      │
│         │cloud            │   │cost              │                      │
│         │(DATA_STRUCTURE) │   │(OBJECTIVE)       │                      │
│         └────────┬────────┘   └─────────────────┘                       │
│                  │ APPEARS_IN                                           │
│         ┌────────┼────────┬────────┐                                    │
│         ▼        ▼        ▼        ▼                                    │
│    ┌─────────┐┌─────────┐┌─────────┐┌─────────┐                         │
│    │geospatial│ single  ││computer ││ spatial │                         │
│    │analysis  │  cell   ││ vision  ││transcr. │                         │
│    │(DOMAIN)  │(DOMAIN) ││(DOMAIN) ││(DOMAIN) │                         │
│    └─────────┘└─────────┘└─────────┘└─────────┘                         │
│                                                                         │
│ APPLIED TO PROBLEMS:                                                    │
│ ├── batch_correction (single_cell) - 45 papers                          │
│ ├── domain_adaptation (computer_vision) - 120 papers                    │
│ ├── point_registration (geospatial) - 30 papers                         │
│ └── gene_imputation (spatial_transcriptomics) - 8 papers  ◀── GAP!      │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│ [p]apers using this | [r]elated methods | [t]ransfer check | [←] back   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Citation Generation

```
> c 1

┌─────────────────────────────────────────────────────────────────────────┐
│ CITATION - Result [1]                                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ APA:                                                                    │
│ Zhang, Y., Chen, X., & Wang, L. (2022). Optimal Transport for           │
│ Single-Cell Genomics. Nature Methods, 19(4), 456-463.                   │
│ https://doi.org/10.1038/s41592-022-01234-5                              │
│                                                                         │
│ BibTeX:                                                                 │
│ @article{zhang2022optimal,                                              │
│   title={Optimal Transport for Single-Cell Genomics},                   │
│   author={Zhang, Yue and Chen, Xiao and Wang, Lin},                     │
│   journal={Nature Methods},                                             │
│   volume={19},                                                          │
│   number={4},                                                           │
│   pages={456--463},                                                     │
│   year={2022},                                                          │
│   doi={10.1038/s41592-022-01234-5}                                      │
│ }                                                                       │
│                                                                         │
│ Vancouver:                                                              │
│ Zhang Y, Chen X, Wang L. Optimal Transport for Single-Cell              │
│ Genomics. Nat Methods. 2022;19(4):456-63.                               │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│ [Copied APA to clipboard]                                               │
│ [a]pa | [b]ibtex | [v]ancouver | [←] back                               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Filter Interface

```
> f

┌─────────────────────────────────────────────────────────────────────────┐
│ SEARCH FILTERS                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ CURRENT QUERY: "optimal transport spatial transcriptomics"              │
│                                                                         │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ DATE RANGE                                                          │ │
│ │ [x] All time                                                        │ │
│ │ [ ] Last year (2025-2026)                                           │ │
│ │ [ ] Last 3 years (2023-2026)                                        │ │
│ │ [ ] Custom: ______ to ______                                        │ │
│ └─────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ DOMAINS                                                  [multi-sel]│ │
│ │ [x] spatial_transcriptomics (234 papers)                            │ │
│ │ [x] single_cell (456 papers)                                        │ │
│ │ [ ] computer_vision (789 papers)                                    │ │
│ │ [ ] geospatial (45 papers)                                          │ │
│ │ [ ] operations_research (23 papers)                                 │ │
│ └─────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ SECTION TYPES                                            [multi-sel]│ │
│ │ [x] Methods                                                         │ │
│ │ [x] Results                                                         │ │
│ │ [ ] Introduction                                                    │ │
│ │ [ ] Discussion                                                      │ │
│ │ [ ] Abstract                                                        │ │
│ └─────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ SEARCH MODALITY                                                     │ │
│ │ [x] Hybrid (semantic + keyword + graph)                             │ │
│ │ [ ] Semantic only                                                   │ │
│ │ [ ] Keyword only                                                    │ │
│ │ [ ] Graph only                                                      │ │
│ └─────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ MINIMUM CONFIDENCE: [████████░░] 0.8                                │ │
│ └─────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│ [a]pply filters | [r]eset | [s]ave as preset | [←] back                 │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Saved Searches / Presets

```
> polymath presets

┌─────────────────────────────────────────────────────────────────────────┐
│ SEARCH PRESETS                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ [1] spatial-methods                                                     │
│     Query: ""                                                           │
│     Filters: domain=spatial_transcriptomics, section=methods            │
│     Last used: 2 hours ago                                              │
│                                                                         │
│ [2] recent-pathology                                                    │
│     Query: "computational pathology"                                    │
│     Filters: domain=pathology, year>=2023                               │
│     Last used: 1 day ago                                                │
│                                                                         │
│ [3] cross-domain-transfer                                               │
│     Query: ""                                                           │
│     Filters: domains=ALL, section=methods, mechanism_extracted=true     │
│     Last used: 3 days ago                                               │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│ [1-3] run preset | [n]ew preset | [d]elete | [←] back                   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## MCP Integration (Claude Code)

When used via MCP in Claude Code, the search presents results in a structured format:

```
User: Search for papers about optimal transport in spatial transcriptomics

Claude: I'll search the Polymath knowledge base for relevant papers.

[Using polymath.semantic_search]

Found 12 relevant passages across 8 papers:

**Top Results:**

1. **Zhang et al. (2022)** - Nature Methods [DOI: 10.1038/s41592-022-01234-5]
   > "We formulate the spatial gene expression imputation problem as an
   > optimal transport task, where each spatial spot is treated as a
   > weighted point in a high-dimensional gene expression space..."

   📊 Confidence: 0.94 | Section: Methods > Transport Optimization

2. **Chen et al. (2023)** - Bioinformatics [DOI: 10.1093/bioinformatics/btac456]
   > "Optimal transport-based methods have shown promise in aligning
   > single-cell distributions across batches..."

   📊 Confidence: 0.87 | Section: Introduction > Related Work

Would you like me to:
- Expand the context for any of these results?
- Show the mechanism graph for optimal transport?
- Generate citations in a specific format?
```
