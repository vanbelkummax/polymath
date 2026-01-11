# Polymath Search Evaluation Report

**Timestamp**: 2026-01-11T00:37:02.121936
**Queries Evaluated**: 14

## Summary Metrics

| Metric | Value |
|--------|-------|
| Mean Recall@10 | 0.793 |
| Mean MRR | 0.708 |
| Failed Queries (recall<0.3) | 0 |

## Performance by Domain

| Domain | Recall | MRR | Count |
|--------|--------|-----|-------|
| cancer_biology | 1.000 | 1.000 | 1 |
| comp_pathology | 0.875 | 0.583 | 2 |
| graph_ml | 1.000 | 0.500 | 1 |
| information_theory | 0.500 | 1.000 | 2 |
| ml_foundations | 0.833 | 0.667 | 3 |
| single_cell | 0.550 | 0.375 | 2 |
| spatial_biology | 0.917 | 0.833 | 3 |

## Query Details

### ⚠️ information bottleneck deep learning

- **Domain**: information_theory
- **Recall**: 0.33
- **Terms Found**: information bottleneck
- **Terms Missing**: mutual information, compression
- **First Relevant Rank**: 1

### ⚠️ residual network deep learning

- **Domain**: ml_foundations
- **Recall**: 0.50
- **Terms Found**: resnet, residual
- **Terms Missing**: skip connection, he kaiming
- **First Relevant Rank**: 2

### ⚠️ single cell RNA sequencing clustering

- **Domain**: single_cell
- **Recall**: 0.50
- **Terms Found**: scrna-seq, clustering
- **Terms Missing**: single cell, scanpy
- **First Relevant Rank**: 4

### ✅ cell type annotation transfer learning

- **Domain**: single_cell
- **Recall**: 0.60
- **Terms Found**: cell type, annotation, transfer
- **Terms Missing**: scanvi, scarches
- **First Relevant Rank**: 2

### ✅ entropy regularization neural network

- **Domain**: information_theory
- **Recall**: 0.67
- **Terms Found**: entropy, regularization
- **Terms Missing**: neural network
- **First Relevant Rank**: 1

### ✅ visium spatial transcriptomics gene expression

- **Domain**: spatial_biology
- **Recall**: 0.75
- **Terms Found**: visium, spatial transcriptomics, spot
- **Terms Missing**: 10x genomics
- **First Relevant Rank**: 1

### ✅ pathology foundation model whole slide image

- **Domain**: comp_pathology
- **Recall**: 0.75
- **Terms Found**: foundation model, pathology, wsi
- **Terms Missing**: whole slide
- **First Relevant Rank**: 1

### ✅ attention is all you need transformer

- **Domain**: ml_foundations
- **Recall**: 1.00
- **Terms Found**: transformer, attention, vaswani
- **Terms Missing**: None
- **First Relevant Rank**: 1

### ✅ BERT pre-training language model

- **Domain**: ml_foundations
- **Recall**: 1.00
- **Terms Found**: bert, pre-training, language model, devlin
- **Terms Missing**: None
- **First Relevant Rank**: 2

### ✅ H&E histology to gene expression prediction

- **Domain**: spatial_biology
- **Recall**: 1.00
- **Terms Found**: h&e, histology, gene expression, prediction, spatial
- **Terms Missing**: None
- **First Relevant Rank**: 2

### ✅ spatial deconvolution cell type

- **Domain**: spatial_biology
- **Recall**: 1.00
- **Terms Found**: deconvolution, cell type, spatial, rctd, cell2location
- **Terms Missing**: None
- **First Relevant Rank**: 1

### ✅ multiple instance learning histopathology

- **Domain**: comp_pathology
- **Recall**: 1.00
- **Terms Found**: mil, multiple instance, pathology, attention
- **Terms Missing**: None
- **First Relevant Rank**: 6

### ✅ graph neural network molecular property

- **Domain**: graph_ml
- **Recall**: 1.00
- **Terms Found**: gnn, graph neural, molecular, mpnn, property prediction
- **Terms Missing**: None
- **First Relevant Rank**: 2

### ✅ colorectal cancer tumor microenvironment spatial

- **Domain**: cancer_biology
- **Recall**: 1.00
- **Terms Found**: colorectal, crc, tumor microenvironment, tme, spatial
- **Terms Missing**: None
- **First Relevant Rank**: 1
