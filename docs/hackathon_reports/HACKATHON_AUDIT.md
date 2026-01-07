# Hackathon Audit - Spatial Multimodal Coverage

Generated: 2026-01-07T04:52:43.568674Z
Postgres DSN: dbname=polymath user=polymath host=/var/run/postgresql
Concept source: neo4j

## Global Counts
- Documents: 29352
- Passages: 532259
- Code files: 64698
- Code chunks: 411729
- Concepts: 200
- Concept links: 0

## Group Coverage Summary
| Group | Title hits | Concept docs | Code files |
| --- | --- | --- | --- |
| core | 646 | 983 | 4843 |
| modalities | 6 | 5 | 30 |
| imaging | 162 | 154 | 0 |
| tasks | 438 | 726 | 161 |
| methods | 332 | 975 | 4880 |
| tooling | 9 | 0 | 937 |

## Keyword Coverage

### core
| Keyword | Title hits | Concept docs | Code files | Status |
| --- | --- | --- | --- | --- |
| spatial | 214 | 186 | 468 | ok |
| multimodal | 74 | 0 | 466 | ok |
| multi-omics | 16 | 0 | 0 | ok |
| multiomics | 1 | 0 | 0 | thin |
| multi modal | 0 | 0 | 0 | missing |
| multi-modal | 15 | 0 | 0 | ok |
| multiome | 1 | 0 | 1 | thin |
| integration | 91 | 0 | 3780 | ok |
| fusion | 271 | 803 | 141 | ok |

### modalities
| Keyword | Title hits | Concept docs | Code files | Status |
| --- | --- | --- | --- | --- |
| visium | 1 | 5 | 20 | ok |
| visium hd | 0 | 0 | 0 | missing |
| xenium | 3 | 0 | 3 | ok |
| merfish | 0 | 0 | 2 | ok |
| cosmx | 0 | 0 | 1 | thin |
| slide-seq | 1 | 0 | 0 | thin |
| slideseq | 0 | 0 | 0 | missing |
| seqfish | 0 | 0 | 2 | ok |
| stereo-seq | 1 | 0 | 0 | thin |
| stereoseq | 0 | 0 | 1 | thin |
| in situ sequencing | 0 | 0 | 0 | missing |
| smfish | 0 | 0 | 2 | ok |

### imaging
| Keyword | Title hits | Concept docs | Code files | Status |
| --- | --- | --- | --- | --- |
| h&e | 19 | 26 | 0 | ok |
| h and e | 6 | 0 | 0 | ok |
| histology | 27 | 35 | 0 | ok |
| pathology | 105 | 109 | 0 | ok |
| immunofluorescence | 6 | 0 | 0 | ok |
| ihc | 7 | 0 | 0 | ok |

### tasks
| Keyword | Title hits | Concept docs | Code files | Status |
| --- | --- | --- | --- | --- |
| alignment | 68 | 0 | 3 | ok |
| registration | 22 | 0 | 27 | ok |
| deconvolution | 33 | 20 | 0 | ok |
| segmentation | 197 | 644 | 81 | ok |
| imputation | 8 | 70 | 2 | ok |
| mapping | 83 | 0 | 48 | ok |
| label transfer | 0 | 0 | 0 | missing |
| cell type | 30 | 2 | 0 | ok |
| co-embedding | 0 | 0 | 0 | missing |
| cross-modal | 9 | 0 | 0 | ok |

### methods
| Keyword | Title hits | Concept docs | Code files | Status |
| --- | --- | --- | --- | --- |
| optimal transport | 3 | 0 | 0 | thin |
| graph matching | 0 | 0 | 0 | missing |
| contrastive | 28 | 70 | 12 | ok |
| transformer | 181 | 457 | 4779 | ok |
| graph neural | 35 | 0 | 0 | ok |
| gnn | 9 | 0 | 50 | ok |
| variational | 14 | 32 | 1 | ok |
| bayesian | 60 | 536 | 5 | ok |
| manifold | 9 | 0 | 35 | ok |

### tooling
| Keyword | Title hits | Concept docs | Code files | Status |
| --- | --- | --- | --- | --- |
| seurat | 3 | 0 | 64 | ok |
| scanpy | 0 | 0 | 310 | ok |
| squidpy | 1 | 0 | 115 | ok |
| cell2location | 0 | 0 | 57 | ok |
| tangram | 3 | 0 | 23 | ok |
| stalign | 0 | 0 | 0 | missing |
| spatialde | 1 | 0 | 0 | thin |
| bayesprism | 0 | 0 | 0 | missing |
| destvi | 0 | 0 | 3 | ok |
| totalvi | 0 | 0 | 3 | ok |
| scvi | 1 | 0 | 375 | ok |

## Missing Keywords (No Paper or Code Signal)
- bayesprism, co-embedding, graph matching, in situ sequencing, label transfer, multi modal, slideseq, stalign, visium hd

## Thin Keywords (Low Coverage)
- cosmx, multiome, multiomics, optimal transport, slide-seq, spatialde, stereo-seq, stereoseq

## Concept Summary (Patterns)

### Concepts matching: spatial
| Concept | Count |
| --- | --- |
| spatial_transcriptomics | 146 |
| spatial transcriptomics | 39 |
| spatial deconvolution | 5 |
| spatial_transcriptomic | 4 |

### Concepts matching: modal
- None

## Suggested Acquisition Actions
- Pull missing modalities and benchmarks as PDFs, then ingest them.
- Add missing code repos, then re-ingest your hackathon repo.
- Re-run this audit after each major ingestion batch.
