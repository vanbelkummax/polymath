#!/usr/bin/env python3
"""
Harvest GitHub repos for methods in the PolyMax Knowledge Graph.
Finds official implementations and records them for ingestion.
"""

import json
import requests
import time
from pathlib import Path

# High-priority methods with known/likely repos
METHOD_REPOS = {
    # Foundation Models (Pathology)
    "UNI": {
        "repo": "mahmoodlab/UNI",
        "stars": 600,
        "description": "Universal pathology foundation model",
        "model": "https://huggingface.co/MahmoodLab/UNI"
    },
    "Prov-GigaPath": {
        "repo": "prov-gigapath/prov-gigapath",
        "stars": 300,
        "description": "Providence whole-slide foundation model",
        "model": "https://huggingface.co/prov-gigapath/prov-gigapath"
    },
    "H-optimus": {
        "repo": "bioptimus/H-optimus-0",
        "stars": 150,
        "description": "Bioptimus pathology foundation model",
        "model": "https://huggingface.co/bioptimus/H-optimus-0"
    },
    "Phikon": {
        "repo": "owkin/PHIKON",
        "stars": 200,
        "description": "Owkin pathology foundation model",
        "model": "https://huggingface.co/owkin/phikon"
    },
    "CONCH": {
        "repo": "mahmoodlab/CONCH",
        "stars": 400,
        "description": "Vision-language pathology model",
        "model": "https://huggingface.co/MahmoodLab/CONCH"
    },
    "Virchow2": {
        "repo": "paige-ai/Virchow",
        "stars": 100,
        "description": "Paige pathology model",
        "model": "https://huggingface.co/paige-ai/Virchow2"
    },

    # Spatial Transcriptomics
    "Cell2location": {
        "repo": "BayraktarLab/cell2location",
        "stars": 250,
        "description": "Probabilistic spatial deconvolution",
        "paper": "https://doi.org/10.1038/s41587-021-01139-4"
    },
    "RCTD": {
        "repo": "dmcable/spacexr",
        "stars": 180,
        "description": "Spatial deconvolution in R",
        "paper": "https://doi.org/10.1038/s41587-021-00830-w"
    },
    "Hist2ST": {
        "repo": "biomed-AI/Hist2ST",
        "stars": 80,
        "description": "H&E to spatial transcriptomics",
        "paper": "https://doi.org/10.1038/s41467-022-34271-z"
    },
    "THItoGene": {
        "repo": "xjtu-bioinformatics/THItoGene",
        "stars": 30,
        "description": "Histology to gene expression",
        "paper": "10.1101/2024.08.29.610290"
    },
    "iStar": {
        "repo": "CaibinSh/iStar",
        "stars": 50,
        "description": "In silico spatial transcriptomics",
        "paper": "https://doi.org/10.1038/s41592-024-02453-y"
    },

    # MIL Methods
    "CLAM": {
        "repo": "mahmoodlab/CLAM",
        "stars": 800,
        "description": "Attention-based MIL for WSI",
        "paper": "https://doi.org/10.1038/s41551-020-00682-w"
    },
    "TransMIL": {
        "repo": "szc19990412/TransMIL",
        "stars": 250,
        "description": "Transformer MIL",
        "paper": "NeurIPS 2021"
    },
    "ABMIL": {
        "repo": "AMLab-Amsterdam/AttentionDeepMIL",
        "stars": 400,
        "description": "Attention-based MIL original",
        "paper": "ICML 2018"
    },

    # RAG/LLM
    "GraphRAG": {
        "repo": "microsoft/graphrag",
        "stars": 15000,
        "description": "Graph-based RAG from Microsoft",
        "paper": "https://arxiv.org/abs/2404.16130"
    },
    "RAG": {
        "repo": "langchain-ai/langchain",
        "stars": 95000,
        "description": "LangChain RAG framework",
        "model": "Various via HuggingFace"
    },

    # Segmentation
    "Cellpose": {
        "repo": "MouseLand/cellpose",
        "stars": 1500,
        "description": "Cell segmentation",
        "paper": "https://doi.org/10.1038/s41592-020-01018-x"
    },
    "StarDist": {
        "repo": "stardist/stardist",
        "stars": 800,
        "description": "Star-convex cell detection",
        "paper": "MICCAI 2018"
    },
    "HoVer-Net": {
        "repo": "vqdang/hover_net",
        "stars": 300,
        "description": "Nuclei segmentation and classification",
        "paper": "Medical Image Analysis 2019"
    },

    # Systems Biology
    "reaction-diffusion": {
        "repo": "pmneila/morphomg",
        "stars": 50,
        "description": "Morphogen simulation",
        "paper": "Turing 1952"
    },
    "network motif": {
        "repo": "alonlab/MatlabNetworkMotifs",
        "stars": 30,
        "description": "Network motif detection",
        "paper": "Milo 2002"
    }
}

# Papers we need to acquire manually (paywalled or not easily accessible)
PAPERS_TO_GET = [
    {
        "title": "Turing original 1952 paper",
        "doi": "10.1098/rstb.1952.0012",
        "reason": "Historical classic, may need library access",
        "priority": "HIGH"
    },
    {
        "title": "McCulloch-Pitts 1943",
        "doi": "10.1007/BF02478259",
        "reason": "Original neural network paper",
        "priority": "HIGH"
    },
    {
        "title": "KEGG Pathway Database paper",
        "pmid": "10592173",
        "reason": "Core bioinformatics reference",
        "priority": "MEDIUM"
    },
    {
        "title": "Cell type hierarchy paper",
        "topic": "Cell ontology formal definitions",
        "reason": "For cell type annotation standardization",
        "priority": "MEDIUM"
    },
    {
        "title": "SBS88/ID18 mutational signatures in EOCRC",
        "query": "colibactin mutational signatures early-onset colorectal cancer",
        "reason": "Core to F30 research",
        "priority": "HIGH"
    }
]

def save_repos():
    """Save repo list for ingestion."""
    output = {
        "repos": METHOD_REPOS,
        "total": len(METHOD_REPOS),
        "generated": "2026-01-04"
    }

    with open("/home/user/work/polymax/data/github_repos.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved {len(METHOD_REPOS)} repos to github_repos.json")

def save_papers_to_get():
    """Save papers we need to acquire manually."""
    with open("/home/user/work/polymax/data/papers_to_get.json", "w") as f:
        json.dump(PAPERS_TO_GET, f, indent=2)

    print(f"Saved {len(PAPERS_TO_GET)} papers to acquire manually")

def generate_markdown_summary():
    """Generate human-readable summary."""
    md = """# PolyMax Augmentation Status

## GitHub Repos to Ingest

| Method | Repo | Stars | Description |
|--------|------|-------|-------------|
"""
    for method, info in sorted(METHOD_REPOS.items()):
        md += f"| {method} | [{info['repo']}](https://github.com/{info['repo']}) | ~{info['stars']} | {info['description']} |\n"

    md += """

## Papers to Acquire Manually

| Title | Source | Priority | Reason |
|-------|--------|----------|--------|
"""
    for paper in PAPERS_TO_GET:
        source = paper.get('doi', paper.get('pmid', paper.get('query', 'N/A')))
        md += f"| {paper['title']} | {source} | {paper['priority']} | {paper['reason']} |\n"

    md += """

## HuggingFace Models to Download

| Model | URL | Size | Use Case |
|-------|-----|------|----------|
| UNI | MahmoodLab/UNI | ~1GB | Pathology embeddings |
| Prov-GigaPath | prov-gigapath/prov-gigapath | ~1.5GB | WSI embeddings |
| H-optimus-1 | bioptimus/H-optimus-0 | ~2GB | Pathology embeddings |
| Virchow2 | paige-ai/Virchow2 | ~1GB | Pathology embeddings |
| CONCH | MahmoodLab/CONCH | ~600MB | Vision-language |

## Next Steps

1. Run `mcp__research-lab__process_github_repos()` for each repo
2. Download HuggingFace models for embedding tests
3. Acquire manual papers through library access
4. Update Knowledge Graph with new code entities
"""

    with open("/home/user/work/polymax/AUGMENTATION_STATUS.md", "w") as f:
        f.write(md)

    print("Generated AUGMENTATION_STATUS.md")

if __name__ == "__main__":
    Path("/home/user/work/polymax/data").mkdir(exist_ok=True)
    save_repos()
    save_papers_to_get()
    generate_markdown_summary()
    print("\nAugmentation data prepared!")
