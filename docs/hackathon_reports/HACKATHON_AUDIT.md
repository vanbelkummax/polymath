# Hackathon Audit - Spatial Multimodal Coverage

Generated: 2026-01-07T06:17:06.202499Z
Postgres DSN: dbname=polymath user=polymath host=/var/run/postgresql
Concept source: neo4j

## Global Counts
- Documents: 29485
- Passages: 545258
- Code files: 65441
- Code chunks: 416397
- Concepts: 200
- Concept links: 0

## Group Coverage Summary
| Group | Title hits | Concept docs | Code files |
| --- | --- | --- | --- |
| core | 670 | 983 | 4900 |
| modalities | 13 | 5 | 174 |
| imaging | 164 | 154 | 0 |
| tasks | 449 | 726 | 184 |
| methods | 341 | 975 | 4880 |
| tooling | 14 | 0 | 1028 |

## Keyword Coverage

### core
| Keyword | Title hits | Concept docs | Code files | Status |
| --- | --- | --- | --- | --- |
| spatial | 230 | 186 | 518 | ok |
| multimodal | 76 | 0 | 466 | ok |
| multi-omics | 16 | 0 | 0 | ok |
| multiomics | 1 | 0 | 0 | thin |
| multi modal | 0 | 0 | 0 | missing |
| multi-modal | 15 | 0 | 0 | ok |
| multiome | 1 | 0 | 1 | thin |
| integration | 93 | 0 | 3781 | ok |
| fusion | 276 | 803 | 147 | ok |

### modalities
| Keyword | Title hits | Concept docs | Code files | Status |
| --- | --- | --- | --- | --- |
| visium | 4 | 5 | 20 | ok |
| visium hd | 2 | 0 | 0 | thin |
| xenium | 6 | 0 | 3 | ok |
| merfish | 0 | 0 | 5 | ok |
| cosmx | 1 | 0 | 113 | ok |
| slide-seq | 2 | 0 | 0 | thin |
| slideseq | 0 | 0 | 25 | ok |
| seqfish | 0 | 0 | 4 | ok |
| stereo-seq | 1 | 0 | 0 | thin |
| stereoseq | 0 | 0 | 1 | thin |
| in situ sequencing | 0 | 0 | 0 | missing |
| smfish | 0 | 0 | 4 | ok |

### imaging
| Keyword | Title hits | Concept docs | Code files | Status |
| --- | --- | --- | --- | --- |
| h&e | 19 | 26 | 0 | ok |
| h and e | 6 | 0 | 0 | ok |
| histology | 28 | 35 | 0 | ok |
| pathology | 106 | 109 | 0 | ok |
| immunofluorescence | 6 | 0 | 0 | ok |
| ihc | 7 | 0 | 0 | ok |

### tasks
| Keyword | Title hits | Concept docs | Code files | Status |
| --- | --- | --- | --- | --- |
| alignment | 73 | 0 | 9 | ok |
| registration | 22 | 0 | 35 | ok |
| deconvolution | 35 | 20 | 1 | ok |
| segmentation | 200 | 644 | 83 | ok |
| imputation | 8 | 70 | 2 | ok |
| mapping | 84 | 0 | 54 | ok |
| label transfer | 0 | 0 | 0 | missing |
| cell type | 32 | 2 | 0 | ok |
| co-embedding | 0 | 0 | 0 | missing |
| cross-modal | 9 | 0 | 0 | ok |

### methods
| Keyword | Title hits | Concept docs | Code files | Status |
| --- | --- | --- | --- | --- |
| optimal transport | 3 | 0 | 0 | thin |
| graph matching | 1 | 0 | 0 | thin |
| contrastive | 29 | 70 | 12 | ok |
| transformer | 185 | 457 | 4779 | ok |
| graph neural | 37 | 0 | 0 | ok |
| gnn | 9 | 0 | 50 | ok |
| variational | 14 | 32 | 1 | ok |
| bayesian | 62 | 536 | 5 | ok |
| manifold | 9 | 0 | 35 | ok |

### tooling
| Keyword | Title hits | Concept docs | Code files | Status |
| --- | --- | --- | --- | --- |
| seurat | 3 | 0 | 86 | ok |
| scanpy | 0 | 0 | 310 | ok |
| squidpy | 1 | 0 | 115 | ok |
| cell2location | 0 | 0 | 57 | ok |
| tangram | 4 | 0 | 23 | ok |
| stalign | 1 | 0 | 5 | ok |
| spatialde | 2 | 0 | 48 | ok |
| bayesprism | 1 | 0 | 15 | ok |
| destvi | 1 | 0 | 3 | ok |
| totalvi | 0 | 0 | 3 | ok |
| scvi | 1 | 0 | 377 | ok |

## Missing Keywords (No Paper or Code Signal)
- co-embedding, in situ sequencing, label transfer, multi modal

## Thin Keywords (Low Coverage)
- graph matching, multiome, multiomics, optimal transport, slide-seq, stereo-seq, stereoseq, visium hd

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
