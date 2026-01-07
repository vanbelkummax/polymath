#!/usr/bin/env python3
"""
Knowledge Base V2 Migration Orchestrator

Runs all migration steps in order:
1. Ensure derived tables
2. Backfill chunk concepts (LLM)
3. Rebuild ChromaDB (BGE-M3)
4. Rebuild Neo4j (concept graph v2)

Supports --only, --skip, --dry-run, --resume flags.
"""

import sys
import argparse
import logging
import subprocess
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).parent


def run_step(step_name: str, script_path: Path, extra_args: list, dry_run: bool):
    """Run a migration step."""
    logger.info(f"=" * 60)
    logger.info(f"STEP: {step_name}")
    logger.info(f"=" * 60)

    cmd = [sys.executable, str(script_path)] + extra_args

    if dry_run and "--dry-run" not in extra_args:
        cmd.append("--dry-run")

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        logger.info(f"✓ {step_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {step_name} failed with code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(description="KB V2 Migration Orchestrator")
    parser.add_argument("--only", help="Run only this step")
    parser.add_argument("--skip", action="append", default=[], help="Skip this step")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--pg-dsn", default="dbname=polymath user=polymath host=/var/run/postgresql")
    parser.add_argument("--limit", type=int, help="Limit items per step")

    args = parser.parse_args()

    # Build common args
    common_args = []
    if args.resume:
        common_args.append("--resume")
    else:
        common_args.append("--no-resume")

    if args.pg_dsn:
        common_args.extend(["--pg-dsn", args.pg_dsn])

    if args.limit:
        common_args.extend(["--limit", str(args.limit)])

    # Define steps
    steps = [
        ("backfill_concepts", SCRIPTS_DIR / "backfill_chunk_concepts_llm.py", common_args),
        ("rebuild_chroma", SCRIPTS_DIR / "rebuild_chroma_bge_m3.py", common_args + ["--target", "passages"]),
        ("rebuild_chroma_code", SCRIPTS_DIR / "rebuild_chroma_bge_m3.py", common_args + ["--target", "chunks"]),
        ("rebuild_neo4j", SCRIPTS_DIR / "rebuild_neo4j_concepts_v2.py", common_args),
    ]

    # Filter steps
    if args.only:
        steps = [(name, path, extra) for name, path, extra in steps if name == args.only]

    steps = [(name, path, extra) for name, path, extra in steps if name not in args.skip]

    logger.info(f"Migration plan: {len(steps)} steps")
    for name, _, _ in steps:
        logger.info(f"  - {name}")

    # Run steps
    results = {}
    for step_name, script_path, extra_args in steps:
        success = run_step(step_name, script_path, extra_args, args.dry_run)
        results[step_name] = success

        if not success:
            logger.error(f"Migration failed at step: {step_name}")
            break

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("MIGRATION SUMMARY")
    logger.info("=" * 60)

    for step_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{status}  {step_name}")

    all_success = all(results.values())

    if all_success:
        logger.info("")
        logger.info("✓ All steps completed successfully!")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Run validation: python scripts/validate_kb_v2.py")
        logger.info("  2. Test hybrid search with new collections")
        logger.info("  3. Test Neo4j bridge queries")
        return 0
    else:
        logger.error("")
        logger.error("✗ Migration incomplete. Fix errors and re-run with --resume")
        return 1


if __name__ == "__main__":
    sys.exit(main())
