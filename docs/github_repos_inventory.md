# GitHub Repos Collection - Full Inventory

**Location**: `/home/user/work/polymax/data/github_repos/`
**Total**: 65 repos, 11GB
**Purpose**: Curated code implementations across 5 strategic tiers

---

## Tier 1: Vanderbilt Ecosystem (Political + Practical)

### Ken-Lau-Lab
**Why**: Your rotation lab, political necessity + spatial biology expertise

| Repo | Purpose | Key Features |
|------|---------|--------------|
| `spatial_CRC_atlas` | Multimodal CRC atlas | Spatial transcriptomics + imaging |
| `dropkick` | scRNA-seq QC | Automated quality control |
| `pCreode` | Trajectory inference | Density-based clustering |
| `intestinal_organoids` | Organoid analysis | Single-cell + spatial methods |

### hrlblab (Huo Lab)
**Why**: Your current lab, medical image analysis focus

| Repo | Purpose | Key Features |
|------|---------|--------------|
| `Map3D` | 3D medical segmentation | Multi-scale attention |
| `PathSeg` | Pathology segmentation | Deep learning for H&E |
| `ASIGN` | Attention-based MIL | Instance-level learning |
| `CircleNet` | Nucleus detection | Circle representation |
| `CS-MIL` | Confidence-scored MIL | Uncertainty quantification |

### MASILab (Landman Lab)
**Why**: Neuroimaging expertise, image analysis pipelines

| Repo | Purpose | Key Features |
|------|---------|--------------|
| `3DUX-Net` | 3D U-Net variant | Cross-attention mechanisms |
| `SLANTbrainSeg` | Brain segmentation | Multi-atlas approach |
| `PreQual` | QC for neuroimaging | Automated quality checks |

---

## Tier 2: Virtual Sequencing Stack (H&E→ST Project)

### mahmoodlab
**Why**: SOTA pathology foundation models, your Img2ST baseline

| Repo | Purpose | Key Features |
|------|---------|--------------|
| `UNI` | Universal pathology encoder | 100M H&E images, ViT-L |
| `CONCH` | Vision-language foundation | Contrastive learning |
| `CLAM` | Attention-based MIL | Clustering + attention |
| `HIPT` | Hierarchical pretraining | Multi-scale ViT |

### theislab
**Why**: Single-cell + spatial transcriptomics methods

| Repo | Purpose | Key Features |
|------|---------|--------------|
| `squidpy` | Spatial analysis toolkit | Graph-based methods |
| `cell2location` | Spatial deconvolution | Probabilistic cell mapping |
| `scvi-tools` | Single-cell variational inference | VAE framework |
| `scanpy` | Single-cell analysis | Python standard |

### owkin
**Why**: Self-supervised pathology, scaling laws

| Repo | Purpose | Key Features |
|------|---------|--------------|
| `HistoSSLscaling` | SSL scaling for pathology | Performance vs dataset size |

---

## Tier 3: BioReasoner Stack (Agentic Reasoning)

### satijalab
**Why**: R spatial standard, widely used

| Repo | Purpose | Key Features |
|------|---------|--------------|
| `seurat` | R spatial/single-cell toolkit | Industry standard |
| `azimuth` | Reference-based annotation | Transfer learning |

### MSKCC-Computational-Pathology
**Why**: MIL baselines, cancer genomics

| Repo | Purpose | Key Features |
|------|---------|--------------|
| `CLAM` | MIL for pathology | Attention mechanisms |

---

## Tier 4: Polymath Stack (Cross-Domain Bridges)

### stanfordnlp
**Why**: Prompt optimization for reasoning agents

| Repo | Purpose | Key Features |
|------|---------|--------------|
| `dspy` | Declarative LM programming | Prompt optimization |

### pyg-team
**Why**: Graph neural networks for spatial data

| Repo | Purpose | Key Features |
|------|---------|--------------|
| `pytorch_geometric` | GNN library | Message passing framework |

### giotto-ai
**Why**: Topological data analysis

| Repo | Purpose | Key Features |
|------|---------|--------------|
| `giotto-tda` | TDA toolkit | Persistent homology |

### infer-actively
**Why**: Active inference, Free Energy Principle

| Repo | Purpose | Key Features |
|------|---------|--------------|
| `pymdp` | Active inference in Python | FEP implementation |

---

## Tier 5: Foundations (ML/AI/CS)

### AI/ML Cookbooks
- `openai/openai-cookbook` - OpenAI API examples
- `anthropics/claude-code` - Claude Code examples
- `anthropics/skills` - Claude skills library

### Core ML Libraries
- `huggingface/transformers` - Transformer models
- `scikit-learn/scikit-learn` - Classical ML
- `pytorch/pytorch` - Deep learning framework

### Knowledge Collections
- `papers-we-love/papers-we-love` - Classic CS papers
- `awesome-computer-vision` - CV resources
- `awesome-deep-learning` - DL resources
- `awesome-machine-learning` - General ML
- `awesome-python` - Python resources

---

## Growth Strategy

### Additions Since 2026-01-04
- **12 new repos** (53 → 65, +22% growth)
- Focus areas: Spatial transcriptomics, foundation models, active inference

### Future Targets
- More spatial biology tools (STUtility, Giotto)
- Causal inference libraries (DoWhy, CausalML)
- Bayesian deep learning (Pyro, NumPyro)

---

## Usage Patterns

### Ingestion
```bash
# Batch ingest all repos
python3 /home/user/work/polymax/scripts/ingest_github_batch.py

# Check progress
tail -f /home/user/work/polymax/logs/github_ingest_*.log
```

### Search Strategies
1. **Direct code search**: `mcp__polymath__semantic_search("attention MIL pathology")`
2. **Cross-domain bridges**: `polymath_cli.py crossref "topology" "gene_expression"`
3. **Implementation lookup**: Search for specific method implementations

---

**Last Updated**: 2026-01-05
**Total Size**: 11GB
**Ingestion Status**: Continuous (Literature Sentry auto-discovers new repos)
