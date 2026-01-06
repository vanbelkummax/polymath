#!/usr/bin/env python3
"""
Audit GitHub Repository Ingestion Status
Compares downloaded repos vs ingested repos in ChromaDB
"""

import os
import chromadb
from pathlib import Path
from collections import defaultdict

# Paths
GITHUB_REPOS_DIR = "/home/user/work/polymax/data/github_repos"
CHROMADB_PATH = "/home/user/work/polymax/chromadb/polymath_v2"

# Strategic tier lists
TIER_1_VANDERBILT = [
    "Ken-Lau-Lab/spatial_CRC_atlas",
    "Ken-Lau-Lab/dropkick",
    "Ken-Lau-Lab/pCreode",
    "hrlblab/Map3D",
    "hrlblab/PathSeg",
    "MASILab/SLANTbrainSeg",
    "MASILab/PreQual",
]

TIER_2_VIRTUAL_SEQUENCING = [
    "mahmoodlab/UNI",
    "mahmoodlab/CONCH",
    "mahmoodlab/CLAM",
    "mahmoodlab/HIPT",
    "theislab/squidpy",
    "theislab/cell2location",
    "theislab/scvi-tools",
    "owkin/HistoSSLscaling",
]

TIER_3_BIOREASONER = [
    "stanfordnlp/dspy",
    "infer-actively/pymdp",
    "pyg-team/pytorch_geometric",
    "giotto-ai/giotto-tda",
]

TIER_4_FOUNDATIONS = [
    "satijalab/seurat",
    "MSKCC-Computational-Pathology/",
    "openai/openai-cookbook",
    "anthropics/anthropic-sdk-python",
    "anthropics/claude-code",
    "anthropics/skills",
    "huggingface/transformers",
]

def get_downloaded_repos():
    """Get list of all downloaded repos."""
    repos = []
    if os.path.exists(GITHUB_REPOS_DIR):
        repos = [d for d in os.listdir(GITHUB_REPOS_DIR)
                if os.path.isdir(os.path.join(GITHUB_REPOS_DIR, d))]
    return sorted(repos)

def get_ingested_repos():
    """Get repos that have been ingested into ChromaDB."""
    client = chromadb.PersistentClient(CHROMADB_PATH)
    collection = client.get_collection('polymath_corpus')

    # Get all items
    result = collection.get(include=['metadatas'])
    metadatas = result['metadatas']

    # Extract repo info
    repo_chunks = defaultdict(int)
    repo_files = defaultdict(set)

    for meta in metadatas:
        if not meta:
            continue

        # Check for repo_name or source containing github
        repo_name = None

        if 'repo_name' in meta:
            repo_name = meta['repo_name']
        elif 'source' in meta and 'github_repos' in meta.get('source', ''):
            # Extract from paths like "github_repos/dspy/..."
            source = meta['source']
            if 'github_repos/' in source:
                parts = source.split('github_repos/')
                if len(parts) > 1:
                    repo_part = parts[1].split('/')[0]
                    repo_name = f"github_repos/{repo_part}"

        if repo_name:
            repo_chunks[repo_name] += 1
            if 'source' in meta:
                repo_files[repo_name].add(meta['source'])

    return dict(repo_chunks), {k: len(v) for k, v in repo_files.items()}

def map_repo_to_tier(repo_name, tier_lists):
    """Check which tier a repo belongs to."""
    # Normalize repo name
    normalized = repo_name.replace('github_repos/', '').lower()

    for tier_name, tier_list in tier_lists.items():
        for tier_repo in tier_list:
            # Match by repo name (case insensitive)
            tier_normalized = tier_repo.split('/')[-1].lower()
            if tier_normalized in normalized or normalized in tier_normalized:
                return tier_name, tier_repo

    return None, None

def main():
    print("=" * 80)
    print("POLYMATH GITHUB REPOSITORY INGESTION AUDIT")
    print("=" * 80)
    print()

    # Get data
    downloaded_repos = get_downloaded_repos()
    ingested_chunks, ingested_files = get_ingested_repos()

    print(f"Downloaded repos: {len(downloaded_repos)}")
    print(f"Ingested repos: {len(ingested_chunks)}")
    print()

    # Define tiers
    tiers = {
        "Tier 1: Vanderbilt Ecosystem (MUST HAVE)": TIER_1_VANDERBILT,
        "Tier 2: Virtual Sequencing (CRITICAL)": TIER_2_VIRTUAL_SEQUENCING,
        "Tier 3: BioReasoner (IMPORTANT)": TIER_3_BIOREASONER,
        "Tier 4: Foundations": TIER_4_FOUNDATIONS,
    }

    # Check each tier
    missing_critical = []

    for tier_name, tier_list in tiers.items():
        print(f"\n{tier_name}")
        print("-" * 80)

        for repo in tier_list:
            repo_short = repo.split('/')[-1]

            # Find matching downloaded repo
            matched_download = None
            for dr in downloaded_repos:
                if repo_short.lower() in dr.lower():
                    matched_download = dr
                    break

            # Find in ingested
            matched_ingested = None
            for ing_repo, chunks in ingested_chunks.items():
                if repo_short.lower() in ing_repo.lower():
                    matched_ingested = ing_repo
                    break

            # Status
            if matched_ingested:
                chunks = ingested_chunks[matched_ingested]
                files = ingested_files.get(matched_ingested, 0)
                status = f"✓ INGESTED ({chunks} chunks, {files} files)"
            elif matched_download:
                status = f"⚠ DOWNLOADED ONLY (not ingested)"
                if "MUST HAVE" in tier_name or "CRITICAL" in tier_name:
                    missing_critical.append((repo, matched_download))
            else:
                status = "✗ MISSING (not downloaded)"
                if "MUST HAVE" in tier_name or "CRITICAL" in tier_name:
                    missing_critical.append((repo, None))

            print(f"  {repo_short:30s} {status}")

    # Top ingested repos by chunks
    print("\n\nTOP 20 INGESTED REPOS BY CHUNK COUNT")
    print("-" * 80)
    sorted_repos = sorted(ingested_chunks.items(), key=lambda x: x[1], reverse=True)[:20]
    for repo, chunks in sorted_repos:
        files = ingested_files.get(repo, 0)
        repo_display = repo.replace('github_repos/', '')
        print(f"  {repo_display:40s} {chunks:6d} chunks  ({files} files)")

    # Summary
    print("\n\nSUMMARY")
    print("=" * 80)
    print(f"Total repos downloaded: {len(downloaded_repos)}")
    print(f"Total repos ingested: {len(ingested_chunks)}")
    print(f"Missing critical (Tier 1-2): {len(missing_critical)}")

    if missing_critical:
        print("\n\nRECOMMENDED INGESTION PRIORITY FOR EXAM PREP")
        print("-" * 80)
        for i, (repo, downloaded_name) in enumerate(missing_critical, 1):
            if downloaded_name:
                path = f"/home/user/work/polymax/data/github_repos/{downloaded_name}"
                print(f"{i}. {repo}")
                print(f"   Path: {path}")
                print(f"   Command: python3 /home/user/work/polymax/lib/unified_ingest.py {path} --type code")
            else:
                print(f"{i}. {repo} (NOT DOWNLOADED - need to clone first)")
            print()

    # Repos ingested but not in tier list
    print("\n\nOTHER INGESTED REPOS (not in strategic tiers)")
    print("-" * 80)
    tier_repos_normalized = set()
    for tier_list in tiers.values():
        for repo in tier_list:
            tier_repos_normalized.add(repo.split('/')[-1].lower())

    other_repos = []
    for repo in ingested_chunks.keys():
        repo_short = repo.replace('github_repos/', '').lower()
        if not any(tr in repo_short for tr in tier_repos_normalized):
            other_repos.append((repo, ingested_chunks[repo]))

    other_repos.sort(key=lambda x: x[1], reverse=True)
    for repo, chunks in other_repos[:20]:
        files = ingested_files.get(repo, 0)
        print(f"  {repo:40s} {chunks:6d} chunks  ({files} files)")

if __name__ == "__main__":
    main()
