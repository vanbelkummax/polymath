#!/usr/bin/env python3
"""
Test smart ingestion on a small subset of repos.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import logging

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
from unified_ingest import UnifiedIngestor

GITHUB_REPOS_DIR = "/home/user/work/polymax/data/github_repos"

# Test on 3 diverse repos
TEST_REPOS = [
    "UNI",           # Mahmood Lab foundation model (Tier 1, should have good content)
    "agents.md",     # Simple documentation repo (Tier 4, should be quick)
    "Map3D",         # Vanderbilt HRLBLAB repo (Tier 1, medical imaging)
]


def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)

    logger.info("="*60)
    logger.info("SMART INGESTION TEST (3 repos)")
    logger.info("="*60)

    ingestor = UnifiedIngestor()

    for repo_name in TEST_REPOS:
        repo_dir = Path(GITHUB_REPOS_DIR) / repo_name

        if not repo_dir.exists():
            logger.warning(f"✗ {repo_name} not found, skipping")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {repo_name}")
        logger.info(f"{'='*60}")

        try:
            result = ingestor.ingest_github_repo(str(repo_dir), repo_name=repo_name)

            logger.info(f"\nResult:")
            logger.info(f"  Artifact ID: {result.artifact_id}")
            logger.info(f"  Chunks added: {result.chunks_added}")
            logger.info(f"  Concepts: {len(result.concepts_linked)}")
            if result.concepts_linked:
                logger.info(f"    → {', '.join(result.concepts_linked[:5])}")
                if len(result.concepts_linked) > 5:
                    logger.info(f"    ... and {len(result.concepts_linked) - 5} more")
            logger.info(f"  Neo4j created: {result.neo4j_node_created}")
            logger.info(f"  Postgres synced: {result.postgres_synced}")

            if result.errors:
                logger.warning(f"  Errors:")
                for err in result.errors:
                    logger.warning(f"    - {err}")

            if result.chunks_added > 0:
                logger.info(f"\n✓ SUCCESS: {repo_name} ingested")
            else:
                logger.warning(f"\n✗ WARNING: {repo_name} produced no chunks")

        except Exception as e:
            logger.error(f"\n✗ ERROR: {repo_name} failed")
            logger.error(f"  {e}")
            import traceback
            logger.error(traceback.format_exc())

    logger.info(f"\n{'='*60}")
    logger.info("TEST COMPLETE")
    logger.info("="*60)
    logger.info("\nIf all 3 repos succeeded, you can run the full batch:")
    logger.info("  python3 /home/user/work/polymax/scripts/smart_github_ingest.py")


if __name__ == "__main__":
    main()
