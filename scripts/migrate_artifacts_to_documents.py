#!/usr/bin/env python3
"""
Migrate existing artifacts to document registry.

This script:
1. Reads all artifacts from artifacts table
2. Computes deterministic doc_id for each
3. Inserts into documents table
4. Updates artifacts.doc_id to link them

Version: HARDENING_2026-01-05
"""
import sys
sys.path.insert(0, '/home/user/work/polymax')

import psycopg2
from lib.doc_identity import compute_doc_id, upsert_document
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def migrate_artifacts_to_documents(dry_run: bool = False):
    """
    Migrate all artifacts to documents table.

    Args:
        dry_run: If True, don't actually write to DB, just print what would happen
    """
    # Connect to database
    conn = psycopg2.connect("dbname=polymath user=polymath host=/var/run/postgresql")
    cursor = conn.cursor()

    try:
        # Get all artifacts
        cursor.execute("""
            SELECT id, title, authors, year, artifact_type, file_path
            FROM artifacts
            WHERE title IS NOT NULL
            ORDER BY created_at
        """)
        artifacts = cursor.fetchall()

        logger.info(f"Found {len(artifacts)} artifacts to migrate")

        migrated = 0
        skipped = 0
        errors = 0

        for artifact_id, title, authors, year, artifact_type, file_path in artifacts:
            try:
                # Skip if no title
                if not title or title.strip() == "":
                    logger.warning(f"Skipping artifact {artifact_id}: no title")
                    skipped += 1
                    continue

                # Default year if missing
                if not year:
                    year = 2020  # Fallback year for old artifacts

                # Default authors if missing
                if not authors or len(authors) == 0:
                    authors = ["Unknown"]

                # Compute doc_id
                doc_id = compute_doc_id(
                    title=title,
                    authors=authors,
                    year=year,
                    doi=None,  # Will be backfilled in Phase 3
                    pmid=None,
                    arxiv_id=None
                )

                if dry_run:
                    logger.info(f"Would create doc {doc_id} for: {title[:50]}...")
                else:
                    # Upsert document
                    upsert_document(
                        doc_id=doc_id,
                        title=title,
                        authors=authors,
                        year=year,
                        publication_type='paper' if artifact_type == 'paper' else 'code',
                        parser_version='legacy',
                        db_conn=conn
                    )

                    # Update artifact to link to document
                    cursor.execute("""
                        UPDATE artifacts
                        SET doc_id = %s
                        WHERE id = %s
                    """, (str(doc_id), artifact_id))

                    conn.commit()

                migrated += 1

                if migrated % 100 == 0:
                    logger.info(f"Migrated {migrated}/{len(artifacts)} artifacts")

            except Exception as e:
                logger.error(f"Error migrating artifact {artifact_id}: {e}")
                errors += 1
                conn.rollback()

        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info(f"MIGRATION {'DRY RUN ' if dry_run else ''}COMPLETE")
        logger.info(f"Total artifacts: {len(artifacts)}")
        logger.info(f"Migrated: {migrated}")
        logger.info(f"Skipped: {skipped}")
        logger.info(f"Errors: {errors}")

        # Verify
        if not dry_run:
            cursor.execute("SELECT COUNT(*) FROM documents")
            doc_count = cursor.fetchone()[0]
            logger.info(f"\nDocuments table now has {doc_count} entries")

            cursor.execute("SELECT COUNT(*) FROM artifacts WHERE doc_id IS NOT NULL")
            linked_count = cursor.fetchone()[0]
            logger.info(f"Artifacts with doc_id: {linked_count}")

            cursor.execute("SELECT COUNT(DISTINCT doc_id) FROM artifacts WHERE doc_id IS NOT NULL")
            unique_docs = cursor.fetchone()[0]
            logger.info(f"Unique documents: {unique_docs}")

            # Check for duplicates (same doc_id from multiple artifacts)
            cursor.execute("""
                SELECT doc_id, COUNT(*) as count
                FROM artifacts
                WHERE doc_id IS NOT NULL
                GROUP BY doc_id
                HAVING COUNT(*) > 1
                ORDER BY count DESC
                LIMIT 10
            """)
            duplicates = cursor.fetchall()
            if duplicates:
                logger.warning(f"\nFound {len(duplicates)} doc_ids with multiple artifacts:")
                for doc_id, count in duplicates[:5]:
                    logger.warning(f"  {doc_id}: {count} artifacts")

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        conn.rollback()
        raise

    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Migrate artifacts to document registry")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually write, just print")
    parser.add_argument("--execute", action="store_true", help="Actually run the migration")

    args = parser.parse_args()

    if not args.execute and not args.dry_run:
        print("ERROR: Must specify either --dry-run or --execute")
        print("\nUsage:")
        print("  # Preview what will happen:")
        print("  python3 scripts/migrate_artifacts_to_documents.py --dry-run")
        print("\n  # Actually run migration:")
        print("  python3 scripts/migrate_artifacts_to_documents.py --execute")
        sys.exit(1)

    migrate_artifacts_to_documents(dry_run=args.dry_run)
