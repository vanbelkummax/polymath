#!/usr/bin/env python3
"""Validate KB V2 migration."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
import chromadb
from neo4j import GraphDatabase

from lib.config import CHROMADB_PATH, PAPERS_COLLECTION, CODE_COLLECTION, NEO4J_URI, NEO4J_PASSWORD


def main():
    print("=" * 60)
    print("KB V2 VALIDATION")
    print("=" * 60)

    errors = []

    # 1. Postgres
    try:
        conn = psycopg2.connect("dbname=polymath user=polymath host=/var/run/postgresql")
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM chunks")
        chunk_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM passages")
        passage_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM chunk_concepts")
        chunk_concept_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM passage_concepts")
        passage_concept_count = cursor.fetchone()[0]

        print(f"\n✓ Postgres:")
        print(f"  Chunks: {chunk_count:,}")
        print(f"  Passages: {passage_count:,}")
        print(f"  Chunk concepts: {chunk_concept_count:,}")
        print(f"  Passage concepts: {passage_concept_count:,}")

        if chunk_concept_count == 0 and passage_concept_count == 0:
            errors.append("No concepts extracted yet")

        cursor.close()
        conn.close()

    except Exception as e:
        errors.append(f"Postgres error: {e}")

    # 2. ChromaDB
    try:
        client = chromadb.PersistentClient(path=str(CHROMADB_PATH))

        papers_coll = client.get_collection(PAPERS_COLLECTION)
        papers_count = papers_coll.count()

        print(f"\n✓ ChromaDB:")
        print(f"  Papers collection ({PAPERS_COLLECTION}): {papers_count:,}")

        try:
            code_coll = client.get_collection(CODE_COLLECTION)
            code_count = code_coll.count()
            print(f"  Code collection ({CODE_COLLECTION}): {code_count:,}")
        except:
            print(f"  Code collection not yet created")

        if papers_count == 0:
            errors.append("Papers collection is empty")

    except Exception as e:
        errors.append(f"ChromaDB error: {e}")

    # 3. Neo4j
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=("neo4j", NEO4J_PASSWORD))

        with driver.session() as session:
            result = session.run("MATCH (c:Concept) RETURN count(c) as cnt")
            concept_count = result.single()["cnt"]

            result = session.run("MATCH ()-[m:MENTIONS]->() RETURN count(m) as cnt")
            mentions_count = result.single()["cnt"]

            result = session.run("MATCH (p:Passage) RETURN count(p) as cnt")
            passage_node_count = result.single()["cnt"]

            print(f"\n✓ Neo4j:")
            print(f"  Concepts: {concept_count:,}")
            print(f"  MENTIONS edges: {mentions_count:,}")
            print(f"  Passage nodes: {passage_node_count:,}")

            if concept_count == 0:
                errors.append("No concepts in Neo4j")

        driver.close()

    except Exception as e:
        errors.append(f"Neo4j error: {e}")

    # Summary
    print("\n" + "=" * 60)
    if errors:
        print("✗ VALIDATION FAILED")
        for err in errors:
            print(f"  - {err}")
        return 1
    else:
        print("✓ VALIDATION PASSED")
        print("=" * 60)
        return 0


if __name__ == "__main__":
    sys.exit(main())
