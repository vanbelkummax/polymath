#!/usr/bin/env python3
"""
Selective Vanderbilt Lab Ingestion Plan

Strategy: Tiered priority system based on lab affiliation and research relevance
- Tier 1 (Ken Lau): ALL repos (direct collaborators, spatial CRC)
- Tier 2 (Huo/HRLB): Top 15 repos (current lab, comp path)
- Tier 3 (MASILab): Top 10 repos (neuroimaging, 180 total)
- Tier 4 (mahmoodlab): Fill gaps (comp path foundation models)
- Tier 5 (Sarkar/IGlab): ALL repos (small sets, spatial omics)
"""

import requests
import json
from typing import List, Dict
from dataclasses import dataclass
import time

GITHUB_API = "https://api.github.com"

@dataclass
class RepoMetadata:
    full_name: str
    stars: int
    description: str
    language: str
    updated_at: str
    size_kb: int
    relevance_score: float = 0.0
    tier: str = ""


def get_repo_metadata(org: str, repo_name: str) -> RepoMetadata:
    """Fetch repo metadata from GitHub API."""
    url = f"{GITHUB_API}/repos/{org}/{repo_name}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Warning: Failed to fetch {org}/{repo_name}")
        return None

    data = response.json()
    return RepoMetadata(
        full_name=data['full_name'],
        stars=data['stargazers_count'],
        description=data.get('description', ''),
        language=data.get('language', 'Unknown'),
        updated_at=data['updated_at'],
        size_kb=data['size']
    )


def score_relevance(repo: RepoMetadata, tier: str) -> float:
    """
    Score repo relevance to user's research:
    - Spatial transcriptomics, pathology, CRC, imaging
    """
    score = 0.0
    desc_lower = (repo.description or '').lower()

    # Keywords: spatial omics, pathology, imaging
    if any(kw in desc_lower for kw in ['spatial', 'transcriptomics', 'st-', 'visium']):
        score += 0.3
    if any(kw in desc_lower for kw in ['pathology', 'histology', 'h&e', 'wsi']):
        score += 0.3
    if any(kw in desc_lower for kw in ['segmentation', 'detection', 'image']):
        score += 0.2
    if any(kw in desc_lower for kw in ['crc', 'cancer', 'tumor', 'colon']):
        score += 0.2

    # Star-based boost (log scale to handle power-law)
    import math
    if repo.stars > 0:
        star_boost = min(0.3, math.log10(repo.stars + 1) / 10)
        score += star_boost

    # Language preference (Python > other)
    if repo.language == 'Python':
        score += 0.1

    # Tier-specific boost
    tier_boosts = {
        'Tier 1 (Ken Lau)': 0.5,  # Always ingest
        'Tier 2 (Huo/HRLB)': 0.3,
        'Tier 3 (MASILab)': 0.1,
        'Tier 4 (mahmoodlab)': 0.4,
        'Tier 5 (Sarkar/IGlab)': 0.4
    }
    score += tier_boosts.get(tier, 0.0)

    return score


def get_org_repos(org: str) -> List[Dict]:
    """Fetch all repos from an organization."""
    repos = []
    page = 1
    while True:
        url = f"{GITHUB_API}/orgs/{org}/repos?page={page}&per_page=100"
        response = requests.get(url)
        if response.status_code != 200:
            break
        data = response.json()
        if not data:
            break
        repos.extend(data)
        page += 1
        time.sleep(0.5)  # Rate limiting
    return repos


def main():
    print("=" * 80)
    print("SELECTIVE VANDERBILT LAB INGESTION PLAN")
    print("=" * 80)
    print()

    # Tier 1: Ken Lau Lab (ALL 17 repos - direct collaborators)
    print("Tier 1: Ken Lau Lab (spatial CRC, trajectory inference)")
    print("-" * 80)
    ken_lau_repos = [
        "spatial_CRC_atlas", "spatial_CRC_atlas_imaging",
        "dropkick", "pCreode", "CoSTA",
        "Ostomy_image_metadata_extraction", "IHC_analysis",
        "CRCDiet", "DADA2", "MicrobAT", "Microbiome_analysis",
        "Tbet_mouse_RNASeq", "Stomach_immune_scRNASeq",
        "scMappR", "scMappR_Data", "Organoid_scRNASeq", "RNA_edits"
    ]

    tier1_scored = []
    for repo_name in ken_lau_repos:
        meta = get_repo_metadata("Ken-Lau-Lab", repo_name)
        if meta:
            meta.tier = "Tier 1 (Ken Lau)"
            meta.relevance_score = score_relevance(meta, meta.tier)
            tier1_scored.append(meta)
            desc = (meta.description or "No description")[:50]
            print(f"  ✓ {repo_name:40s} [{meta.stars:4d}⭐] {desc}")

    print(f"\nTier 1 Total: {len(tier1_scored)} repos → INGEST ALL")
    print()

    # Tier 2: Huo Lab/HRLB (Top 15 of 34 - current lab, comp path)
    print("Tier 2: Huo Lab/HRLB (computational pathology, medical imaging)")
    print("-" * 80)
    hrlb_priority = [
        "CircleNet", "CS-MIL", "ASIGN", "Map3D",
        "COVID-19-Detection", "MaskMitochondria",
        "CS2-Net", "MpoxSLDNet", "MpoxVSM",
        "LambdaNet", "MediMeta", "CSA-Net",
        "WSI-BERT", "PathSeg", "COVID-Net"
    ]

    tier2_scored = []
    for repo_name in hrlb_priority:
        meta = get_repo_metadata("hrlblab", repo_name)
        if meta:
            meta.tier = "Tier 2 (Huo/HRLB)"
            meta.relevance_score = score_relevance(meta, meta.tier)
            tier2_scored.append(meta)

    tier2_scored.sort(key=lambda x: x.relevance_score, reverse=True)
    for repo in tier2_scored[:15]:
        print(f"  ✓ {repo.full_name:40s} [{repo.stars:4d}⭐] Score: {repo.relevance_score:.2f}")

    print(f"\nTier 2 Total: {len(tier2_scored[:15])} repos (top 15 by relevance)")
    print()

    # Tier 3: MASILab (Top 10 of 180 - neuroimaging, high-impact only)
    print("Tier 3: MASILab (neuroimaging, 180 total → select top 10)")
    print("-" * 80)
    masi_priority = [
        "SLANTbrainSeg", "3DUX-Net", "PreQual",
        "Synb0-DISCO", "TractoSCR", "MaCRUISE",
        "SLANT-etal", "slant_tha", "slant_hippo",
        "spline-ARCTIC"
    ]

    tier3_scored = []
    for repo_name in masi_priority:
        meta = get_repo_metadata("MASILab", repo_name)
        if meta:
            meta.tier = "Tier 3 (MASILab)"
            meta.relevance_score = score_relevance(meta, meta.tier)
            tier3_scored.append(meta)

    tier3_scored.sort(key=lambda x: x.stars, reverse=True)
    for repo in tier3_scored[:10]:
        print(f"  ✓ {repo.full_name:40s} [{repo.stars:4d}⭐] Score: {repo.relevance_score:.2f}")

    print(f"\nTier 3 Total: {len(tier3_scored[:10])} repos (top 10 by stars)")
    print()

    # Tier 4: mahmoodlab (Fill gaps - check what's missing)
    print("Tier 4: mahmoodlab (comp path foundation models)")
    print("-" * 80)
    mahmood_priority = [
        "UNI", "CONCH", "CLAM", "HIPT",
        "MMP", "PORPOISE", "PathomicFusion",
        "MCAT", "DSMIL", "TransMIL"
    ]

    tier4_scored = []
    for repo_name in mahmood_priority:
        meta = get_repo_metadata("mahmoodlab", repo_name)
        if meta:
            meta.tier = "Tier 4 (mahmoodlab)"
            meta.relevance_score = score_relevance(meta, meta.tier)
            tier4_scored.append(meta)

    tier4_scored.sort(key=lambda x: x.relevance_score, reverse=True)
    for repo in tier4_scored[:10]:
        print(f"  ✓ {repo.full_name:40s} [{repo.stars:4d}⭐] Score: {repo.relevance_score:.2f}")

    print(f"\nTier 4 Total: {len(tier4_scored[:10])} repos (top 10 by relevance)")
    print()

    # Tier 5: Sarkar/IGlab (ALL - small sets, spatial omics)
    print("Tier 5: Sarkar/IGlab (spatial omics, single-cell)")
    print("-" * 80)
    iglab_repos = [
        "MAGE_ab_generation", "MAGE_RNA_FISH",
        "codex_human_ovarian_cancer_10x_spatial",
        "SlideSeqV2_analysis_MERFISH_analysis",
        "PASTE", "SPIRAL"
    ]

    tier5_scored = []
    for repo_name in iglab_repos:
        meta = get_repo_metadata("IGlab-VUMC", repo_name)
        if meta:
            meta.tier = "Tier 5 (Sarkar/IGlab)"
            meta.relevance_score = score_relevance(meta, meta.tier)
            tier5_scored.append(meta)
            desc = (meta.description or "No description")[:50]
            print(f"  ✓ {repo_name:40s} [{meta.stars:4d}⭐] {desc}")

    print(f"\nTier 5 Total: {len(tier5_scored)} repos → INGEST ALL")
    print()

    # Summary
    print("=" * 80)
    print("INGESTION SUMMARY")
    print("=" * 80)
    total_repos = len(tier1_scored) + len(tier2_scored[:15]) + len(tier3_scored[:10]) + len(tier4_scored[:10]) + len(tier5_scored)
    print(f"Total repos to ingest: {total_repos}")
    print(f"  Tier 1 (Ken Lau):    {len(tier1_scored)} repos")
    print(f"  Tier 2 (Huo/HRLB):   {len(tier2_scored[:15])} repos")
    print(f"  Tier 3 (MASILab):    {len(tier3_scored[:10])} repos")
    print(f"  Tier 4 (mahmoodlab): {len(tier4_scored[:10])} repos")
    print(f"  Tier 5 (Sarkar/IGlab): {len(tier5_scored)} repos")
    print()

    # Generate ingestion script
    all_repos = (
        tier1_scored +
        tier2_scored[:15] +
        tier3_scored[:10] +
        tier4_scored[:10] +
        tier5_scored
    )

    with open("/home/user/work/polymax/reports/vanderbilt_ingestion_plan.sh", "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Selective Vanderbilt Lab Ingestion\n")
        f.write(f"# Total: {total_repos} repos\n")
        f.write("# Generated: 2026-01-05\n\n")
        f.write("set -e\n")
        f.write("cd /home/user/work/polymax/data/github_repos\n\n")

        for tier_name, repos in [
            ("Tier 1 (Ken Lau)", tier1_scored),
            ("Tier 2 (Huo/HRLB)", tier2_scored[:15]),
            ("Tier 3 (MASILab)", tier3_scored[:10]),
            ("Tier 4 (mahmoodlab)", tier4_scored[:10]),
            ("Tier 5 (Sarkar/IGlab)", tier5_scored)
        ]:
            f.write(f"\n# {tier_name}\n")
            for repo in repos:
                org, name = repo.full_name.split('/')
                f.write(f"echo 'Cloning {repo.full_name}...'\n")
                f.write(f"git clone https://github.com/{repo.full_name}.git || true\n")
                f.write(f"python3 /home/user/work/polymax/lib/unified_ingest.py {name} --type code\n")
                f.write(f"echo '  → {name} ingested'\n\n")

    print("Generated: /home/user/work/polymax/reports/vanderbilt_ingestion_plan.sh")
    print()

    # Save metadata JSON
    with open("/home/user/work/polymax/reports/vanderbilt_repos_metadata.json", "w") as f:
        json.dump([{
            'full_name': r.full_name,
            'stars': r.stars,
            'description': r.description,
            'language': r.language,
            'tier': r.tier,
            'relevance_score': r.relevance_score
        } for r in all_repos], f, indent=2)

    print("Saved metadata: /home/user/work/polymax/reports/vanderbilt_repos_metadata.json")
    print()


if __name__ == "__main__":
    main()
