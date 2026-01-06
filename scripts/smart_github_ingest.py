#!/usr/bin/env python3
"""
Smart GitHub Repository Batch Ingestion

Uses intelligent filtering to ingest repositories as flattened documents:
- README + structure first
- Key implementation files prioritized
- Token-budgeted to prevent dilution
- Density filtering to skip generated code

Each repo becomes a single, searchable artifact with cross-file context preserved.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import logging

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
from unified_ingest import UnifiedIngestor

GITHUB_REPOS_DIR = "/home/user/work/polymax/data/github_repos"

# Tiered approach: Different token budgets for different priorities
TIER_1_REPOS = {
    # Vanderbilt ecosystem (political + practical)
    "Map3D", "SLANTbrainSeg", "PreQual", "NSC-seq",
    # Ken Lau Lab
    "spatial_CRC_atlas", "dropkick", "pCreode",
}

TIER_2_REPOS = {
    # Virtual sequencing stack (H&E→ST)
    "UNI", "CONCH", "CLAM", "HIPT",  # Mahmood Lab
    "squidpy", "cell2location", "scvi-tools",  # Theis Lab
    "HistoSSLscaling",  # Owkin
}

TIER_3_REPOS = {
    # BioReasoner stack
    "seurat",  # Satija Lab
    "dspy",  # Stanford
    "pymdp",  # Active inference
    "pytorch_geometric",  # GNNs
    "giotto-tda",  # Topological data analysis
}

# Everything else gets standard ingestion
# Tier 4 = general repos (TheAlgorithms, awesome-*, etc.)


def get_repo_tier(repo_name: str) -> int:
    """Determine repo tier for prioritization."""
    if repo_name in TIER_1_REPOS:
        return 1
    elif repo_name in TIER_2_REPOS:
        return 2
    elif repo_name in TIER_3_REPOS:
        return 3
    else:
        return 4


def main():
    # Setup logging
    log_file = f"/home/user/work/polymax/logs/smart_github_ingest_{datetime.now().strftime('%Y%m%d_%H%M')}.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info("="*60)
    logger.info("SMART GITHUB REPOSITORY INGESTION")
    logger.info("="*60)

    ingestor = UnifiedIngestor()

    repos = sorted([d for d in Path(GITHUB_REPOS_DIR).iterdir() if d.is_dir()])
    logger.info(f"Found {len(repos)} repositories")

    # Group by tier
    tier_groups = {1: [], 2: [], 3: [], 4: []}
    for repo in repos:
        tier = get_repo_tier(repo.name)
        tier_groups[tier].append(repo)

    logger.info(f"Tier 1 (Vanderbilt): {len(tier_groups[1])} repos")
    logger.info(f"Tier 2 (Virtual Seq): {len(tier_groups[2])} repos")
    logger.info(f"Tier 3 (BioReasoner): {len(tier_groups[3])} repos")
    logger.info(f"Tier 4 (General): {len(tier_groups[4])} repos")

    total_chunks = 0
    total_concepts = set()
    results = []

    # Process all repos in tier order
    for tier in [1, 2, 3, 4]:
        tier_repos = tier_groups[tier]
        if not tier_repos:
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"PROCESSING TIER {tier} ({len(tier_repos)} repos)")
        logger.info(f"{'='*60}")

        for i, repo_dir in enumerate(tier_repos):
            repo_name = repo_dir.name

            logger.info(f"\n[{i+1}/{len(tier_repos)}] {repo_name}")

            try:
                result = ingestor.ingest_github_repo(
                    str(repo_dir),
                    repo_name=repo_name,
                    ingestion_method="smart_repo_ingest",
                )

                if result.chunks_added > 0:
                    total_chunks += result.chunks_added
                    total_concepts.update(result.concepts_linked)

                    logger.info(f"  ✓ Chunks: {result.chunks_added}")
                    logger.info(f"  ✓ Concepts: {', '.join(result.concepts_linked[:5])}")
                    if result.concepts_linked and len(result.concepts_linked) > 5:
                        logger.info(f"    ... and {len(result.concepts_linked) - 5} more")

                    results.append({
                        "tier": tier,
                        "repo": repo_name,
                        "chunks": result.chunks_added,
                        "concepts": list(result.concepts_linked),
                        "neo4j": result.neo4j_node_created,
                        "postgres": result.postgres_synced,
                    })
                else:
                    logger.warning(f"  ✗ No content ingested")
                    if result.errors:
                        for err in result.errors:
                            logger.warning(f"    {err}")

            except Exception as e:
                logger.error(f"  ✗ Error: {e}")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("INGESTION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total repositories: {len(repos)}")
    logger.info(f"Successfully ingested: {len(results)}")
    logger.info(f"Total chunks: {total_chunks}")
    logger.info(f"Unique concepts: {len(total_concepts)}")

    # Write detailed report
    report_path = f"/home/user/work/polymax/reports/smart_github_ingest_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, "w") as f:
        f.write("# Smart GitHub Repository Ingestion Report\n\n")
        f.write(f"**Date:** {datetime.now().isoformat()}\n\n")

        f.write("## Summary\n\n")
        f.write(f"- Repositories processed: {len(repos)}\n")
        f.write(f"- Successfully ingested: {len(results)}\n")
        f.write(f"- Total chunks: {total_chunks}\n")
        f.write(f"- Unique concepts: {len(total_concepts)}\n\n")

        f.write("## Ingestion Method\n\n")
        f.write("Each repository was ingested as a **single flattened document** using:\n\n")
        f.write("- **Smart Filtering**: README, structure, key implementation files\n")
        f.write("- **Density Filtering**: Skip generated code, migrations, test fixtures\n")
        f.write("- **Token Budgeting**: Max 50K tokens per repo\n")
        f.write("- **Concept Extraction**: Cross-domain linking via imports and docstrings\n\n")

        f.write("## By Tier\n\n")
        for tier in [1, 2, 3, 4]:
            tier_results = [r for r in results if r["tier"] == tier]
            if not tier_results:
                continue

            tier_names = {
                1: "Vanderbilt Ecosystem",
                2: "Virtual Sequencing Stack",
                3: "BioReasoner Stack",
                4: "General/Foundations"
            }

            f.write(f"### Tier {tier}: {tier_names[tier]}\n\n")
            f.write(f"Repositories: {len(tier_results)}\n\n")

            for r in tier_results:
                f.write(f"**{r['repo']}**\n")
                f.write(f"- Chunks: {r['chunks']}\n")
                if r['concepts']:
                    f.write(f"- Concepts: {', '.join(r['concepts'][:10])}\n")
                f.write("\n")

        f.write("## Concept Distribution\n\n")
        f.write("Top concepts across all repositories:\n\n")

        # Count concept frequency
        concept_counts = {}
        for r in results:
            for concept in r['concepts']:
                concept_counts[concept] = concept_counts.get(concept, 0) + 1

        top_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        for concept, count in top_concepts:
            f.write(f"- `{concept}`: {count} repos\n")

    logger.info(f"\nDetailed report: {report_path}")
    logger.info(f"Log file: {log_file}")


if __name__ == "__main__":
    main()
