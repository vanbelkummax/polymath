#!/usr/bin/env python3
"""
Rebuild Neo4j concept graph with typed Chunk→Concept edges.

Creates versioned graph schema:
- Nodes: Artifact, Chunk, Passage, Concept
- Edges: HAS_CHUNK, HAS_PASSAGE, MENTIONS (with confidence/version)
"""

import sys
import os
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from neo4j import GraphDatabase
from tqdm import tqdm

from lib.kb_derived import ensure_derived_tables, update_migration_checkpoint, get_migration_checkpoint
from lib.config import NEO4J_URI, NEO4J_PASSWORD, NEO4J_USER

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Neo4jV2Rebuilder:
    """Rebuild Neo4j with concept graph v2 schema."""

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_constraints(self):
        """Create constraints and indexes."""
        with self.driver.session() as session:
            # Unique constraints
            constraints = [
                "CREATE CONSTRAINT artifact_id_unique IF NOT EXISTS FOR (a:Artifact) REQUIRE a.artifact_id IS UNIQUE",
                "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
                "CREATE CONSTRAINT passage_id_unique IF NOT EXISTS FOR (p:Passage) REQUIRE p.passage_id IS UNIQUE",
            ]

            # Indexes
            indexes = [
                "CREATE INDEX concept_name_idx IF NOT EXISTS FOR (c:Concept) ON (c.name)",
                "CREATE INDEX concept_type_idx IF NOT EXISTS FOR (c:Concept) ON (c.type)",
                "CREATE INDEX artifact_year_idx IF NOT EXISTS FOR (a:Artifact) ON (a.year)",
            ]

            for stmt in constraints + indexes:
                try:
                    session.run(stmt)
                except Exception as e:
                    logger.warning(f"Constraint/index creation warning: {e}")

        logger.info("Constraints and indexes created")

    def wipe_v2_graph(self):
        """Delete all v2 nodes (Artifact, Chunk, Passage, Concept)."""
        with self.driver.session() as session:
            session.run("MATCH (n:Artifact) DETACH DELETE n")
            session.run("MATCH (n:Chunk) DETACH DELETE n")
            session.run("MATCH (n:Passage) DETACH DELETE n")
            session.run("MATCH (n:Concept) DETACH DELETE n")

        logger.info("Wiped v2 graph nodes")

    def rebuild_from_postgres(
        self,
        pg_dsn: str,
        extractor_version: str,
        edge_version: str,
        limit: int = None,
        resume: bool = True,
        dry_run: bool = False,
        target: str = "both"
    ):
        """Rebuild graph from Postgres."""
        import psycopg2

        conn = psycopg2.connect(pg_dsn)
        ensure_derived_tables(conn)

        job_name = f"rebuild_neo4j_v2_{extractor_version}"

        # Get checkpoint
        start_after = None
        if resume:
            checkpoint = get_migration_checkpoint(conn, job_name)
            if checkpoint and checkpoint["status"] != "completed":
                start_after = checkpoint["cursor_position"]
                logger.info(f"Resuming from {start_after}")

        if target in ["passages", "both"]:
            self._rebuild_passages(conn, start_after, extractor_version, edge_version, limit, dry_run)

        if target in ["chunks", "both"]:
            self._rebuild_chunks(conn, start_after, extractor_version, edge_version, limit, dry_run)

        conn.close()

    def _rebuild_passages(self, conn, start_after, extractor_version, edge_version, limit, dry_run):
        """Rebuild passage-based nodes."""
        cursor = conn.cursor()

        # Query passages with their concepts
        if start_after:
            query = """
            SELECT p.passage_id::text, p.doc_id::text, d.title, d.year,
                   p.passage_text,
                   array_agg(pc.concept_name) as concepts,
                   array_agg(pc.concept_type) as types,
                   array_agg(pc.confidence) as confidences
            FROM passages p
            JOIN documents d ON p.doc_id = d.doc_id
            LEFT JOIN passage_concepts pc ON p.passage_id = pc.passage_id
                AND pc.extractor_version = %s
            WHERE p.passage_id::text > %s
            GROUP BY p.passage_id, p.doc_id, d.title, d.year, p.passage_text
            ORDER BY p.passage_id
            """
            params = (extractor_version, start_after)
        else:
            query = """
            SELECT p.passage_id::text, p.doc_id::text, d.title, d.year,
                   p.passage_text,
                   array_agg(pc.concept_name) as concepts,
                   array_agg(pc.concept_type) as types,
                   array_agg(pc.confidence) as confidences
            FROM passages p
            JOIN documents d ON p.doc_id = d.doc_id
            LEFT JOIN passage_concepts pc ON p.passage_id = pc.passage_id
                AND pc.extractor_version = %s
            GROUP BY p.passage_id, p.doc_id, d.title, d.year, p.passage_text
            ORDER BY p.passage_id
            """
            params = (extractor_version,)

        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query, params)

        processed = 0

        with self.driver.session() as session:
            for row in tqdm(cursor, desc="Rebuilding passages"):
                passage_id, doc_id, title, year, text, concepts, types, confidences = row

                if dry_run:
                    processed += 1
                    continue

                # MERGE Artifact (document)
                session.run("""
                MERGE (a:Artifact {artifact_id: $doc_id})
                ON CREATE SET a.title = $title, a.year = $year, a.source = 'paper'
                """, doc_id=doc_id, title=title or "Untitled", year=year or 0)

                # MERGE Passage
                session.run("""
                MERGE (p:Passage {passage_id: $passage_id})
                ON CREATE SET p.text_preview = substring($text, 0, 200)
                """, passage_id=passage_id, text=text or "")

                # MERGE relationship
                session.run("""
                MATCH (a:Artifact {artifact_id: $doc_id})
                MATCH (p:Passage {passage_id: $passage_id})
                MERGE (a)-[:HAS_PASSAGE]->(p)
                """, doc_id=doc_id, passage_id=passage_id)

                # MERGE concepts and relationships
                if concepts and concepts[0]:
                    for i, concept_name in enumerate(concepts):
                        if not concept_name:
                            continue

                        concept_type = types[i] if types and i < len(types) else "domain"
                        confidence = confidences[i] if confidences and i < len(confidences) else 0.5

                        session.run("""
                        MERGE (c:Concept {name: $name})
                        ON CREATE SET c.type = $type
                        """, name=concept_name, type=concept_type)

                        session.run("""
                        MATCH (p:Passage {passage_id: $passage_id})
                        MATCH (c:Concept {name: $name})
                        MERGE (p)-[m:MENTIONS]->(c)
                        SET m.confidence = $confidence,
                            m.extractor_version = $extractor_version,
                            m.edge_version = $edge_version,
                            m.weight = $confidence
                        """, passage_id=passage_id, name=concept_name,
                             confidence=float(confidence or 0.5),
                             extractor_version=extractor_version,
                             edge_version=edge_version)

                processed += 1

                if processed % 100 == 0:
                    update_migration_checkpoint(
                        conn, f"rebuild_neo4j_v2_{extractor_version}",
                        passage_id, "running",
                        items_processed=100,
                        cursor_type="passage_id"
                    )

        cursor.close()
        logger.info(f"Rebuilt {processed} passages")

    def _rebuild_chunks(self, conn, start_after, extractor_version, edge_version, limit, dry_run):
        """Rebuild chunk-based nodes (code)."""
        cursor = conn.cursor()

        if start_after:
            query = """
            SELECT c.id, c.artifact_id::text, c.content,
                   array_agg(cc.concept_name) as concepts,
                   array_agg(cc.concept_type) as types,
                   array_agg(cc.confidence) as confidences
            FROM chunks c
            LEFT JOIN chunk_concepts cc ON c.id = cc.chunk_id
                AND cc.extractor_version = %s
            WHERE c.id > %s
            GROUP BY c.id, c.artifact_id, c.content
            ORDER BY c.id
            """
            params = (extractor_version, start_after)
        else:
            query = """
            SELECT c.id, c.artifact_id::text, c.content,
                   array_agg(cc.concept_name) as concepts,
                   array_agg(cc.concept_type) as types,
                   array_agg(cc.confidence) as confidences
            FROM chunks c
            LEFT JOIN chunk_concepts cc ON c.id = cc.chunk_id
                AND cc.extractor_version = %s
            GROUP BY c.id, c.artifact_id, c.content
            ORDER BY c.id
            """
            params = (extractor_version,)

        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query, params)

        processed = 0

        with self.driver.session() as session:
            for row in tqdm(cursor, desc="Rebuilding chunks"):
                chunk_id, artifact_id, content, concepts, types, confidences = row

                if dry_run:
                    processed += 1
                    continue

                # MERGE Chunk
                session.run("""
                MERGE (c:Chunk {chunk_id: $chunk_id})
                ON CREATE SET c.text_preview = substring($content, 0, 200)
                """, chunk_id=chunk_id, content=content or "")

                # Link to artifact if exists
                if artifact_id:
                    session.run("""
                    MERGE (a:Artifact {artifact_id: $artifact_id})
                    """, artifact_id=artifact_id)

                    session.run("""
                    MATCH (a:Artifact {artifact_id: $artifact_id})
                    MATCH (c:Chunk {chunk_id: $chunk_id})
                    MERGE (a)-[:HAS_CHUNK]->(c)
                    """, artifact_id=artifact_id, chunk_id=chunk_id)

                # MERGE concepts
                if concepts and concepts[0]:
                    for i, concept_name in enumerate(concepts):
                        if not concept_name:
                            continue

                        concept_type = types[i] if types and i < len(types) else "domain"
                        confidence = confidences[i] if confidences and i < len(confidences) else 0.5

                        session.run("""
                        MERGE (c:Concept {name: $name})
                        ON CREATE SET c.type = $type
                        """, name=concept_name, type=concept_type)

                        session.run("""
                        MATCH (chunk:Chunk {chunk_id: $chunk_id})
                        MATCH (c:Concept {name: $name})
                        MERGE (chunk)-[m:MENTIONS]->(c)
                        SET m.confidence = $confidence,
                            m.extractor_version = $extractor_version,
                            m.edge_version = $edge_version,
                            m.weight = $confidence
                        """, chunk_id=chunk_id, name=concept_name,
                             confidence=float(confidence or 0.5),
                             extractor_version=extractor_version,
                             edge_version=edge_version)

                processed += 1

        cursor.close()
        logger.info(f"Rebuilt {processed} chunks")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pg-dsn", default="dbname=polymath user=polymath host=/var/run/postgresql")
    parser.add_argument("--neo4j-uri", default=NEO4J_URI)
    parser.add_argument("--neo4j-user", default=NEO4J_USER)
    parser.add_argument("--neo4j-pass", default=os.environ.get("NEO4J_PASSWORD", NEO4J_PASSWORD))
    parser.add_argument("--extractor-version", default="llm_v2")
    parser.add_argument("--edge-version", default="llm_v2")
    parser.add_argument("--wipe-neo4j", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--target", choices=["passages", "chunks", "both"], default="both")

    args = parser.parse_args()

    rebuilder = Neo4jV2Rebuilder(args.neo4j_uri, args.neo4j_user, args.neo4j_pass)

    try:
        rebuilder.create_constraints()

        if args.wipe_neo4j:
            logger.warning("Wiping Neo4j v2 graph...")
            rebuilder.wipe_v2_graph()

        rebuilder.rebuild_from_postgres(
            args.pg_dsn,
            args.extractor_version,
            args.edge_version,
            limit=args.limit,
            resume=args.resume,
            dry_run=args.dry_run,
            target=args.target
        )

        logger.info("Neo4j rebuild complete!")

    finally:
        rebuilder.close()


if __name__ == "__main__":
    main()
