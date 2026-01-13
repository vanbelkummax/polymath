# Ingestion Interface Design

**Author**: Max Van Belkum
**Version**: 2.0.0-alpha

---

## Zotero Sync Interface

```
╔══════════════════════════════════════════════════════════════════════════╗
║                      POLYMATH 2.0 - ZOTERO SYNC                           ║
╚══════════════════════════════════════════════════════════════════════════╝

> polymath ingest --from-zotero

┌─────────────────────────────────────────────────────────────────────────┐
│ ZOTERO SYNC STATUS                                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ Zotero Library: max.vanbelkum@vanderbilt.edu                            │
│ Last Sync: 2026-01-13 14:30:00                                          │
│ Export Path: ~/Zotero/exports/polymath_export.csv                       │
│                                                                         │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ LIBRARY SUMMARY                                                     │ │
│ ├─────────────────────────────────────────────────────────────────────┤ │
│ │ Total Items:         3,456                                          │ │
│ │ With PDF:            3,158 (91%)                                    │ │
│ │ With DOI:            3,400 (98%)                                    │ │
│ │ With PMID:           2,890 (84%)                                    │ │
│ │ Already Indexed:     2,500 (72%)                                    │ │
│ │ Pending Ingest:        658 (28%)                                    │ │
│ └─────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│ Collections to Sync:                                                    │
│ [x] spatial-omics (456 items)                                           │
│ [x] pathology (234 items)                                               │
│ [x] single-cell (567 items)                                             │
│ [x] methods (890 items)                                                 │
│ [ ] reading-list (45 items) - personal, skip                            │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│ [s]ync now | [c]onfigure | [v]iew pending | [l]ogs | [q]uit             │
└─────────────────────────────────────────────────────────────────────────┘

> s

Syncing with Zotero...

Step 1/4: Exporting from Zotero... ████████████████████ Done
Step 2/4: Validating metadata... ████████████████████ Done
Step 3/4: Parsing PDFs... ████████████░░░░░░░░ 60% (395/658)

┌─────────────────────────────────────────────────────────────────────────┐
│ CURRENT: Chen_2024_SpatialOT.pdf                                        │
├─────────────────────────────────────────────────────────────────────────┤
│ Title: Optimal Transport for Spatial Gene Expression                    │
│ DOI: 10.1038/s41592-024-01234-5                                         │
│ Authors: Chen, X.; Wang, L.; Zhang, Y.                                  │
│ Year: 2024                                                              │
│ Venue: Nature Methods                                                   │
│                                                                         │
│ Parsing:                                                                │
│ ├── Sections detected: 8                                                │
│ ├── Figures detected: 5                                                 │
│ ├── Tables detected: 2                                                  │
│ └── References extracted: 45                                            │
│                                                                         │
│ Quality Score: 0.94 ████████████████████░░                              │
└─────────────────────────────────────────────────────────────────────────┘

Estimated time remaining: 12 minutes

[p]ause | [c]ancel | running in background (safe to close)
```

---

## Single Paper Ingest

```
> polymath ingest ~/Downloads/new_paper.pdf

┌─────────────────────────────────────────────────────────────────────────┐
│ SINGLE PAPER INGEST                                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ File: new_paper.pdf (2.4 MB)                                            │
│                                                                         │
│ ═══════════════════════════════════════════════════════════════════════ │
│ STEP 1: METADATA DETECTION                                              │
│ ═══════════════════════════════════════════════════════════════════════ │
│                                                                         │
│ Extracted from PDF:                                                     │
│ ├── Title: "Deep Learning for Spatial Transcriptomics" ✓                │
│ ├── Authors: "Smith, J.; Jones, A."                     ?               │
│ ├── Year: 2024                                          ✓               │
│ └── DOI: [not found]                                    ✗               │
│                                                                         │
│ ⚠️  DOI not found in PDF. Searching external sources...                  │
│                                                                         │
│ CrossRef match (confidence: 0.95):                                      │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ Title: Deep Learning for Spatial Transcriptomics: A Review          │ │
│ │ DOI: 10.1016/j.cels.2024.01.002                                     │ │
│ │ Authors: Smith, John A.; Jones, Alice B.                            │ │
│ │ Venue: Cell Systems                                                 │ │
│ │ Year: 2024                                                          │ │
│ └─────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│ Is this the correct paper? [y]es / [n]o / [e]dit manually               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

> y

┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 2: PDF PARSING                                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ Extracting structure... ████████████████████ Done                       │
│                                                                         │
│ Document Structure:                                                     │
│ ├── Abstract (1 paragraph)                                              │
│ ├── Introduction (4 paragraphs)                                         │
│ ├── Background (6 paragraphs)                                           │
│ │   ├── Spatial Transcriptomics Technologies                            │
│ │   └── Deep Learning Approaches                                        │
│ ├── Methods (8 paragraphs)                                              │
│ │   ├── Data Preprocessing                                              │
│ │   ├── Model Architecture                                              │
│ │   └── Training Procedure                                              │
│ ├── Results (5 paragraphs)                                              │
│ ├── Discussion (4 paragraphs)                                           │
│ └── References (89 citations)                                           │
│                                                                         │
│ Total passages to create: 28                                            │
│ Average passage length: 1,456 chars ✓                                   │
│                                                                         │
│ Figures detected: 7                                                     │
│ Tables detected: 3                                                      │
│                                                                         │
│ Quality Score: 0.91 ████████████████████░░░                             │
│                                                                         │
│ [c]ontinue | [r]eparse | [v]iew structure | [q]uit                      │
└─────────────────────────────────────────────────────────────────────────┘

> c

┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 3: CONCEPT EXTRACTION                                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ Extracting concepts... ████████████████████ Done                        │
│                                                                         │
│ Concepts Extracted:                                                     │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ METHODS (12)                                                        │ │
│ │ ├── convolutional_neural_network                                    │ │
│ │ ├── graph_neural_network                                            │ │
│ │ ├── transformer                                                     │ │
│ │ ├── variational_autoencoder                                         │ │
│ │ └── ... (8 more)                                                    │ │
│ ├─────────────────────────────────────────────────────────────────────┤ │
│ │ MECHANISMS (8)                                                      │ │
│ │ ├── attention_mechanism                                             │ │
│ │ │   └── "computes weighted sum of spatial neighbors"                │ │
│ │ ├── graph_convolution                                               │ │
│ │ │   └── "aggregates features from adjacent nodes"                   │ │
│ │ ├── latent_space_encoding                                           │ │
│ │ │   └── "compresses input to lower-dimensional representation"      │ │
│ │ └── ... (5 more)                                                    │ │
│ ├─────────────────────────────────────────────────────────────────────┤ │
│ │ DATA STRUCTURES (5)                                                 │ │
│ │ ├── spatial_graph                                                   │ │
│ │ ├── expression_matrix                                               │ │
│ │ ├── image_tiles                                                     │ │
│ │ └── ... (2 more)                                                    │ │
│ ├─────────────────────────────────────────────────────────────────────┤ │
│ │ PROBLEMS (6)                                                        │ │
│ │ ├── spatial_gene_prediction                                         │ │
│ │ ├── cell_type_classification                                        │ │
│ │ └── ... (4 more)                                                    │ │
│ └─────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│ [c]ontinue | [e]dit concepts | [v]iew all | [q]uit                      │
└─────────────────────────────────────────────────────────────────────────┘

> c

┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 4: STORAGE                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ Writing to databases...                                                 │
│                                                                         │
│ PostgreSQL:                                                             │
│ ├── documents_v2: 1 record ✓                                            │
│ └── passages_v2: 28 records ✓                                           │
│                                                                         │
│ ChromaDB:                                                               │
│ └── polymath_v2_bge_m3: 28 embeddings ✓                                 │
│                                                                         │
│ Neo4j:                                                                  │
│ ├── :Paper node: 1 ✓                                                    │
│ ├── :Method nodes: 12 (4 new, 8 linked) ✓                               │
│ ├── :Mechanism nodes: 8 (3 new, 5 linked) ✓                             │
│ └── Relationships: 34 ✓                                                 │
│                                                                         │
│ ═══════════════════════════════════════════════════════════════════════ │
│ INGEST COMPLETE                                                         │
│ ═══════════════════════════════════════════════════════════════════════ │
│                                                                         │
│ Document ID: 550e8400-e29b-41d4-a716-446655440000                        │
│ Total passages: 28                                                      │
│ Concepts extracted: 31                                                  │
│ Time elapsed: 45 seconds                                                │
│                                                                         │
│ 💡 Tip: This paper introduces 3 new mechanisms. Consider exploring      │
│    cross-domain transfer opportunities with `polymath discover`.        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Code Repository Ingest

```
> polymath ingest-repo https://github.com/mahmoodlab/HIPT

┌─────────────────────────────────────────────────────────────────────────┐
│ CODE REPOSITORY INGEST                                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ Repository: mahmoodlab/HIPT                                             │
│ URL: https://github.com/mahmoodlab/HIPT                                 │
│ Description: Hierarchical Image Pyramid Transformer                     │
│ Stars: 456 | Forks: 89 | Last commit: 2025-08-15                        │
│                                                                         │
│ ═══════════════════════════════════════════════════════════════════════ │
│ STEP 1: CLONING                                                         │
│ ═══════════════════════════════════════════════════════════════════════ │
│                                                                         │
│ Cloning repository... ████████████████████ Done (234 MB)                │
│                                                                         │
│ ═══════════════════════════════════════════════════════════════════════ │
│ STEP 2: FILE ANALYSIS                                                   │
│ ═══════════════════════════════════════════════════════════════════════ │
│                                                                         │
│ File breakdown:                                                         │
│ ├── Python (.py): 45 files                                              │
│ ├── Notebooks (.ipynb): 8 files                                         │
│ ├── Config (.yaml, .json): 12 files                                     │
│ ├── Documentation (.md): 5 files                                        │
│ └── Other: 23 files                                                     │
│                                                                         │
│ Key files detected:                                                     │
│ ├── models/hipt.py - Main model architecture                            │
│ ├── models/vit.py - Vision Transformer components                       │
│ ├── train.py - Training script                                          │
│ ├── eval.py - Evaluation script                                         │
│ └── utils/preprocessing.py - Data preprocessing                         │
│                                                                         │
│ [c]ontinue | [s]elect files | [v]iew tree | [q]uit                      │
└─────────────────────────────────────────────────────────────────────────┘

> c

┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 3: CODE CHUNKING                                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ Processing files... ████████████████░░░░ 78%                            │
│                                                                         │
│ Current: models/hipt.py                                                 │
│                                                                         │
│ Chunks extracted:                                                       │
│ ├── Classes: 8                                                          │
│ │   ├── HIPT_LR4K (main model)                                          │
│ │   ├── HIPT_LR (region-level)                                          │
│ │   ├── Attention (multi-head)                                          │
│ │   └── ... (5 more)                                                    │
│ ├── Functions: 23                                                       │
│ │   ├── forward_features()                                              │
│ │   ├── get_attention_scores()                                          │
│ │   └── ... (21 more)                                                   │
│ └── Docstrings: 15                                                      │
│                                                                         │
│ ═══════════════════════════════════════════════════════════════════════ │
│ CONCEPT LINKING                                                         │
│ ═══════════════════════════════════════════════════════════════════════ │
│                                                                         │
│ Linked to existing concepts:                                            │
│ ├── vision_transformer → 12 chunks                                      │
│ ├── attention_mechanism → 8 chunks                                      │
│ ├── hierarchical_features → 5 chunks                                    │
│ └── patch_embedding → 4 chunks                                          │
│                                                                         │
│ Linked to papers:                                                       │
│ ├── Chen et al. (2022) "HIPT for WSI Analysis" - DOI:10.1007/xxx        │
│ └── Dosovitskiy et al. (2020) "ViT" - DOI:10.48550/arXiv.2010.11929     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Batch Ingest Queue

```
> polymath ingest-queue status

┌─────────────────────────────────────────────────────────────────────────┐
│ INGEST QUEUE STATUS                                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ Queue: 45 items | Processing: 3 | Completed today: 127 | Failed: 2      │
│                                                                         │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ STATUS │ TYPE │ NAME                              │ PROGRESS        │ │
│ ├────────┼──────┼───────────────────────────────────┼─────────────────┤ │
│ │ ▶ RUN  │ PDF  │ Chen_2024_SpatialOT.pdf          │ ████████░░ 80%  │ │
│ │ ▶ RUN  │ PDF  │ Wang_2024_GraphST.pdf            │ ████░░░░░░ 40%  │ │
│ │ ▶ RUN  │ REPO │ github.com/spatialdata/spatialdata│ ██░░░░░░░░ 20%  │ │
│ │ ○ WAIT │ PDF  │ Smith_2024_DeepST.pdf            │ ░░░░░░░░░░  0%  │ │
│ │ ○ WAIT │ PDF  │ Jones_2024_CellTyping.pdf        │ ░░░░░░░░░░  0%  │ │
│ │ ...    │      │ (40 more items)                   │                 │ │
│ │ ✗ FAIL │ PDF  │ corrupted_scan.pdf               │ OCR failed      │ │
│ │ ✗ FAIL │ PDF  │ password_protected.pdf           │ Access denied   │ │
│ └─────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│ Workers: 3/4 active | CPU: 45% | Memory: 8.2 GB                         │
│ Estimated completion: 2 hours 15 minutes                                │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│ [a]dd items | [p]ause queue | [r]etry failed | [c]lear completed        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Staging Directory Monitor

```
> polymath watch ~/ingest_staging/

┌─────────────────────────────────────────────────────────────────────────┐
│ STAGING DIRECTORY MONITOR                                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ Watching: /home/user/work/polymax/ingest_staging/                       │
│ Mode: Auto-ingest on file drop                                          │
│                                                                         │
│ Waiting for files... (drop PDFs here to auto-ingest)                    │
│                                                                         │
│ Recent activity:                                                        │
│ ├── 14:32:15 ✓ Chen_2024.pdf → Ingested (doc_id: 550e8400...)           │
│ ├── 14:30:45 ✓ Wang_2024.pdf → Ingested (doc_id: 660f9500...)           │
│ └── 14:28:00 ✗ scan.pdf → Failed (OCR quality too low)                  │
│                                                                         │
│ [Press Ctrl+C to stop watching]                                         │
│                                                                         │
│ ─────────────────────────────────────────────────────────────────────── │
│ 14:35:22 📥 New file detected: NewPaper_2024.pdf                        │
│ 14:35:23 🔍 Validating...                                               │
│ 14:35:25 📄 Parsing PDF...                                              │
│ 14:35:45 🧠 Extracting concepts...                                      │
│ 14:36:12 💾 Storing...                                                  │
│ 14:36:15 ✓ Complete! (doc_id: 770a0600-e29b-41d4-a716-556677890000)     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```
