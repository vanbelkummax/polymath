#!/usr/bin/env python3
"""
Sync citation metadata from Postgres to Neo4j Paper nodes.
Run after fix_citations_postgres.sql

Usage:
  python3 scripts/sync_citations_neo4j.py              # Update existing only (safe)
  python3 scripts/sync_citations_neo4j.py --create-missing  # Also create missing nodes
"""
import psycopg2
from neo4j import GraphDatabase
import sys
import os

POSTGRES_DSN = "dbname=polymath user=polymath host=/var/run/postgresql"
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "polymathic2026")
NEO4J_AUTH = (NEO4J_USER, NEO4J_PASSWORD)

def main():
    # Parse args
    create_missing = "--create-missing" in sys.argv

    print("Connecting to databases...")
    pg = psycopg2.connect(POSTGRES_DSN)
    neo = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

    # Get citation-ready docs from Postgres
    cur = pg.cursor()
    cur.execute("""
        SELECT doc_id::text, title, authors, year, doi, venue, citation_ready
        FROM documents
        WHERE doc_id IN (SELECT DISTINCT doc_id FROM passages)
    """)

    docs = cur.fetchall()
    print(f"Found {len(docs)} documents to sync")
    print(f"Mode: {'update + create missing' if create_missing else 'update existing only (safe)'}")

    updated = 0
    created = 0
    skipped = 0
    errors = 0
    batch_size = 500

    with neo.session() as session:
        for i, (doc_id, title, authors, year, doi, venue, citation_ready) in enumerate(docs):
            try:
                # Try to update existing Paper node
                result = session.run("""
                    MATCH (p:Paper {id: $doc_id})
                    SET p.title = $title,
                        p.authors = $authors,
                        p.year = $year,
                        p.doi = $doi,
                        p.venue = $venue,
                        p.citation_ready = $citation_ready
                    RETURN p
                """, doc_id=doc_id, title=title, authors=authors,
                   year=year, doi=doi, venue=venue, citation_ready=citation_ready)

                if result.single():
                    updated += 1
                elif create_missing:
                    # Create new Paper node only if flag set
                    session.run("""
                        CREATE (p:Paper {
                            id: $doc_id,
                            title: $title,
                            authors: $authors,
                            year: $year,
                            doi: $doi,
                            venue: $venue,
                            citation_ready: $citation_ready
                        })
                    """, doc_id=doc_id, title=title, authors=authors,
                       year=year, doi=doi, venue=venue, citation_ready=citation_ready)
                    created += 1
                else:
                    skipped += 1

            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  Error on {doc_id}: {e}")

            if (i + 1) % batch_size == 0:
                print(f"  Progress: {i+1}/{len(docs)} ({updated} updated, {created} created, {skipped} skipped)")

    print(f"\n=== NEO4J SYNC COMPLETE ===")
    print(f"Updated: {updated}")
    print(f"Created: {created}")
    print(f"Skipped (no existing node): {skipped}")
    print(f"Errors: {errors}")

    # Verify
    with neo.session() as session:
        result = session.run("""
            MATCH (p:Paper)
            RETURN
                count(p) as total,
                count(p.year) as has_year,
                count(p.authors) as has_authors,
                count(p.citation_ready) as has_flag
        """)
        row = result.single()
        print(f"\nNeo4j Paper nodes now:")
        print(f"  Total: {row['total']:,}")
        print(f"  Has year: {row['has_year']:,}")
        print(f"  Has authors: {row['has_authors']:,}")
        print(f"  Has citation_ready: {row['has_flag']:,}")

    pg.close()
    neo.close()

if __name__ == "__main__":
    main()
