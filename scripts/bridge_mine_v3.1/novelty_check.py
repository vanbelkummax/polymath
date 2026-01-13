#!/usr/bin/env python3
"""
Novelty Check for Transfer Candidates

Checks PubMed/Semantic Scholar for prior art combining method + spatial transcriptomics.
"""

import os
import sys
import json
import time
import requests
from typing import List, Dict, Tuple

# Top transfer candidates from data analysis
TRANSFER_CANDIDATES = [
    {
        "method": "latent_diffusion_models",
        "display": "Latent Diffusion Models (LDMs)",
        "spatial_pct": 5.4,
        "total_papers": 37,
        "transfer_hypothesis": "Use LDMs to generate high-resolution spatial gene expression from H&E images, similar to how Stable Diffusion generates images from text",
        "search_terms": ["latent diffusion spatial transcriptomics", "LDM gene expression", "stable diffusion spatial omics"]
    },
    {
        "method": "denoising_diffusion_probabilistic_models",
        "display": "Denoising Diffusion Probabilistic Models (DDPMs)",
        "spatial_pct": 7.7,
        "total_papers": 39,
        "transfer_hypothesis": "Apply DDPMs to impute missing gene expression values in sparse spatial transcriptomics data",
        "search_terms": ["DDPM spatial transcriptomics", "diffusion model gene expression imputation", "denoising diffusion spatial omics"]
    },
    {
        "method": "inpainting",
        "display": "Image Inpainting",
        "spatial_pct": 9.7,
        "total_papers": 31,
        "transfer_hypothesis": "Treat spatial gene expression as images and use inpainting techniques to fill missing spots/cells",
        "search_terms": ["inpainting spatial transcriptomics", "gene expression inpainting", "spatial omics inpainting"]
    },
    {
        "method": "missing_value_imputation",
        "display": "Missing Value Imputation (classical)",
        "spatial_pct": 9.7,
        "total_papers": 31,
        "transfer_hypothesis": "Apply matrix completion/imputation methods to sparse ST data considering spatial structure",
        "search_terms": ["imputation spatial transcriptomics", "matrix completion spatial omics", "missing value spatial gene expression"]
    },
    {
        "method": "score_matching",
        "display": "Score Matching / Score-Based Models",
        "spatial_pct": 0.0,
        "total_papers": 20,
        "transfer_hypothesis": "Use score-based generative models to learn spatial gene expression distributions and sample novel patterns",
        "search_terms": ["score matching spatial transcriptomics", "score-based model gene expression", "energy-based model spatial omics"]
    },
    {
        "method": "neural_radiance_field",
        "display": "Neural Radiance Fields (NeRF)",
        "spatial_pct": 0.0,
        "total_papers": 15,
        "transfer_hypothesis": "Use NeRF-style implicit representations to reconstruct 3D gene expression from serial sections",
        "search_terms": ["NeRF spatial transcriptomics", "neural radiance field tissue", "implicit neural representation spatial omics"]
    },
    {
        "method": "optical_flow",
        "display": "Optical Flow Estimation",
        "spatial_pct": 22.6,
        "total_papers": 31,
        "transfer_hypothesis": "Use optical flow to register serial tissue sections for 3D reconstruction",
        "search_terms": ["optical flow serial section", "optical flow tissue registration", "optical flow spatial transcriptomics"]
    },
    {
        "method": "compressed_sensing",
        "display": "Compressed Sensing",
        "spatial_pct": 27.7,
        "total_papers": 47,
        "transfer_hypothesis": "Apply compressed sensing to recover full gene expression from sub-sampled spatial measurements",
        "search_terms": ["compressed sensing spatial transcriptomics", "compressed sensing gene expression", "sparse recovery spatial omics"]
    },
    {
        "method": "deformable_registration",
        "display": "Deformable Registration",
        "spatial_pct": 21.1,
        "total_papers": 38,
        "transfer_hypothesis": "Use deformable registration to align spatial transcriptomics data across different tissue sections/patients",
        "search_terms": ["deformable registration spatial transcriptomics", "non-rigid registration spatial omics", "diffeomorphic registration gene expression"]
    },
    {
        "method": "momentum_contrast",
        "display": "Momentum Contrast (MoCo)",
        "spatial_pct": 0.0,
        "total_papers": 35,
        "transfer_hypothesis": "Apply MoCo contrastive learning to learn self-supervised representations of spatial gene expression patterns",
        "search_terms": ["MoCo spatial transcriptomics", "contrastive learning spatial omics", "momentum contrast gene expression"]
    }
]


def check_pubmed(query: str) -> Tuple[int, List[str]]:
    """Check PubMed for papers matching query."""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": 5,
        "retmode": "json"
    }

    try:
        resp = requests.get(base_url, params=params, timeout=10)
        data = resp.json()
        count = int(data.get("esearchresult", {}).get("count", 0))
        ids = data.get("esearchresult", {}).get("idlist", [])
        return count, ids
    except Exception as e:
        print(f"  PubMed error: {e}")
        return -1, []


def check_semantic_scholar(query: str) -> Tuple[int, List[Dict]]:
    """Check Semantic Scholar for papers matching query."""
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": 5,
        "fields": "title,year,citationCount"
    }

    try:
        resp = requests.get(base_url, params=params, timeout=10)
        if resp.status_code == 429:
            print("  S2 rate limited, waiting...")
            time.sleep(5)
            return -1, []
        data = resp.json()
        total = data.get("total", 0)
        papers = data.get("data", [])
        return total, papers
    except Exception as e:
        print(f"  S2 error: {e}")
        return -1, []


def run_novelty_check():
    """Run novelty check on all transfer candidates."""
    print("="*70)
    print("NOVELTY CHECK FOR TRANSFER CANDIDATES")
    print("="*70)

    results = []

    for i, candidate in enumerate(TRANSFER_CANDIDATES):
        print(f"\n{i+1}. {candidate['display']}")
        print(f"   Spatial penetration: {candidate['spatial_pct']}%")
        print(f"   Hypothesis: {candidate['transfer_hypothesis'][:80]}...")

        novelty_score = 100  # Start with max novelty
        prior_art = []

        for search_term in candidate['search_terms']:
            # Check PubMed
            pm_count, pm_ids = check_pubmed(search_term)
            if pm_count > 0:
                print(f"   PubMed '{search_term}': {pm_count} results")
                novelty_score -= min(30, pm_count * 5)
                prior_art.append(f"PubMed: {pm_count}")

            # Check Semantic Scholar
            s2_count, s2_papers = check_semantic_scholar(search_term)
            if s2_count > 0:
                print(f"   S2 '{search_term}': {s2_count} results")
                novelty_score -= min(30, s2_count * 2)
                if s2_papers:
                    for p in s2_papers[:2]:
                        prior_art.append(f"S2: {p.get('title', 'Unknown')[:50]}...")

            time.sleep(0.5)  # Rate limiting

        novelty_score = max(0, novelty_score)

        result = {
            **candidate,
            "novelty_score": novelty_score,
            "prior_art_found": len(prior_art) > 0,
            "prior_art": prior_art,
            "recommendation": "INVESTIGATE" if novelty_score >= 50 else "EXISTING" if novelty_score >= 20 else "WELL-STUDIED"
        }
        results.append(result)

        if novelty_score >= 70:
            print(f"   ✓ HIGH NOVELTY ({novelty_score}/100) - Potential discovery!")
        elif novelty_score >= 40:
            print(f"   ~ MEDIUM NOVELTY ({novelty_score}/100) - Some prior work exists")
        else:
            print(f"   ✗ LOW NOVELTY ({novelty_score}/100) - Well-studied area")

    # Sort by novelty score
    results.sort(key=lambda x: -x['novelty_score'])

    # Save results
    output_dir = '/home/user/work/polymax/reports/bridge_mine_v4_a6_bucket'
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, 'TOP10_GAPS_WITH_NOVELTY.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n\nSaved to {output_file}")

    # Generate summary
    print("\n" + "="*70)
    print("TOP 10 TRANSFER OPPORTUNITIES (by Novelty)")
    print("="*70)

    for i, r in enumerate(results):
        status = "🎯" if r['novelty_score'] >= 50 else "📚"
        print(f"\n{i+1}. {status} {r['display']}")
        print(f"   Novelty: {r['novelty_score']}/100 | Spatial: {r['spatial_pct']}%")
        print(f"   {r['transfer_hypothesis']}")
        print(f"   Recommendation: {r['recommendation']}")

    # Generate markdown report
    report_file = os.path.join(output_dir, 'TOP10_TRANSFER_OPPORTUNITIES.md')
    with open(report_file, 'w') as f:
        f.write("# Top 10 Transfer Opportunities for Spatial Transcriptomics\n\n")
        f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Executive Summary\n\n")
        f.write("These methods have high usage in the literature but low penetration into spatial transcriptomics, ")
        f.write("representing potential transfer opportunities.\n\n")

        f.write("| Rank | Method | Novelty | Spatial % | Recommendation |\n")
        f.write("|------|--------|---------|-----------|----------------|\n")
        for i, r in enumerate(results):
            f.write(f"| {i+1} | {r['display']} | {r['novelty_score']}/100 | {r['spatial_pct']}% | {r['recommendation']} |\n")

        f.write("\n## Detailed Analysis\n\n")
        for i, r in enumerate(results):
            f.write(f"### {i+1}. {r['display']}\n\n")
            f.write(f"- **Novelty Score**: {r['novelty_score']}/100\n")
            f.write(f"- **Current Spatial Penetration**: {r['spatial_pct']}%\n")
            f.write(f"- **Total Papers in Corpus**: {r['total_papers']}\n")
            f.write(f"- **Recommendation**: {r['recommendation']}\n\n")
            f.write(f"**Transfer Hypothesis**: {r['transfer_hypothesis']}\n\n")
            if r['prior_art']:
                f.write("**Prior Art Found**:\n")
                for pa in r['prior_art'][:3]:
                    f.write(f"- {pa}\n")
            f.write("\n---\n\n")

    print(f"\nSaved markdown report to {report_file}")

    return results


if __name__ == "__main__":
    run_novelty_check()
