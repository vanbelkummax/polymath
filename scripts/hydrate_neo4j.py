#!/usr/bin/env python3
"""
Neo4j Hydration Script - Sync Postgres → Neo4j

Addresses the sync gap: Neo4j has 1.5K papers vs Postgres 30K.

This script:
1. Creates Paper nodes for all artifacts
2. Creates Concept nodes from passages
3. Creates MENTIONS edges (Paper → Concept)
4. Creates CO_OCCURS edges (Concept ↔ Concept in same doc)

Run with:
    python3 scripts/hydrate_neo4j.py --dry-run    # Preview
    python3 scripts/hydrate_neo4j.py --apply      # Execute
    python3 scripts/hydrate_neo4j.py --batch 1000 # Custom batch size

Estimated time: ~30 minutes for full sync (30K papers)
"""

import argparse
import logging
import os
import re
import sys
from collections import defaultdict
from typing import Set, Dict, List, Tuple

import psycopg2
from psycopg2.extras import RealDictCursor
from neo4j import GraphDatabase

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)

# Config
POSTGRES_DSN = "dbname=polymath user=polymath host=/var/run/postgresql"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "polymathic2026")

# Cross-domain concepts to extract
CONCEPT_PATTERNS = {
    # Signal Processing
    "compressed_sensing": r"compress(?:ed|ive)?\s*sens(?:ing|ed)",
    "sparse_coding": r"sparse\s*cod(?:ing|e)",
    "wavelet": r"wavelet",
    "fourier": r"fourier",
    "denoising": r"denois(?:ing|ed|er)",

    # Physics
    "entropy": r"entropy",
    "free_energy": r"free\s*energy",
    "thermodynamics": r"thermodynamic",
    "diffusion": r"diffusion",

    # Causality
    "causal_inference": r"causal\s*inference",
    "counterfactual": r"counterfactual",

    # Systems
    "feedback": r"feedback\s*(?:loop|control|system)",
    "control_theory": r"control\s*theory",
    "emergence": r"emergenc(?:e|t)",
    "cybernetics": r"cybernetic",

    # Cognitive
    "predictive_coding": r"predictive\s*coding",
    "bayesian": r"bayesian",
    "active_inference": r"active\s*inference",

    # ML/AI
    "transformer": r"transformer",
    "attention": r"attention\s*mechanism",
    "contrastive_learning": r"contrastive\s*learn",
    "foundation_model": r"foundation\s*model",
    "graph_neural_network": r"graph\s*(?:neural\s*)?network|GNN",

    # Biology
    "spatial_transcriptomics": r"spatial\s*transcript",
    "single_cell": r"single[\s-]*cell",
    "gene_expression": r"gene\s*expression",
    "cell_type": r"cell\s*type",

    # Methods
    "segmentation": r"segmentation",
    "classification": r"classif(?:ication|ier)",
    "clustering": r"cluster(?:ing)?",
    "regression": r"regression",
    "imputation": r"imputat(?:ion|ed)",
}


def extract_concepts(text: str) -> Set[str]:
    """Extract concepts from text using pattern matching."""
    if not text:
        return set()

    text_lower = text.lower()
    found = set()

    for concept, pattern in CONCEPT_PATTERNS.items():
        if re.search(pattern, text_lower):
            found.add(concept)

    return found


def get_papers_to_sync(pg_conn, batch_size: int = 1000, offset: int = 0) -> List[Dict]:
    """Get papers from Postgres that need Neo4j sync."""
    with pg_conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT
                a.id::text as artifact_id,
                a.doc_id::text,
                a.title,
                a.year,
                a.authors,
                COALESCE(
                    (SELECT string_agg(p.passage_text, ' ')
                     FROM passages p
                     WHERE p.doc_id = a.doc_id
                     LIMIT 5),
                    ''
                ) as sample_text
            FROM artifacts a
            WHERE a.artifact_type = 'paper'
            ORDER BY a.id
            LIMIT %s OFFSET %s
        """, (batch_size, offset))
        return [dict(r) for r in cur.fetchall()]


def sync_papers_to_neo4j(
    neo4j_driver,
    papers: List[Dict],
    dry_run: bool = False
) -> Tuple[int, int, int]:
    """
    Sync papers to Neo4j.

    Returns: (papers_created, concepts_created, edges_created)
    """
    papers_created = 0
    concepts_created = 0
    edges_created = 0

    # Track all concepts for CO_OCCURS edges
    doc_concepts: Dict[str, Set[str]] = defaultdict(set)
    all_concepts: Set[str] = set()

    with neo4j_driver.session() as session:
        for paper in papers:
            # Extract concepts from title + sample text
            text = f"{paper['title']} {paper['sample_text']}"
            concepts = extract_concepts(text)

            if paper['doc_id']:
                doc_concepts[paper['doc_id']].update(concepts)
            all_concepts.update(concepts)

            if dry_run:
                if concepts:
                    logger.debug(f"Would create: {paper['title'][:50]}... with {len(concepts)} concepts")
                papers_created += 1
                continue

            # Create/merge Paper node
            session.run("""
                MERGE (p:Paper {artifact_id: $artifact_id})
                SET p.title = $title,
                    p.year = $year,
                    p.doc_id = $doc_id
            """, {
                "artifact_id": paper['artifact_id'],
                "title": paper['title'],
                "year": paper['year'],
                "doc_id": paper['doc_id']
            })
            papers_created += 1

            # Create concept nodes and MENTIONS edges
            for concept in concepts:
                session.run("""
                    MERGE (c:CONCEPT {name: $concept})
                    WITH c
                    MATCH (p:Paper {artifact_id: $artifact_id})
                    MERGE (p)-[:MENTIONS]->(c)
                """, {
                    "concept": concept,
                    "artifact_id": paper['artifact_id']
                })
                edges_created += 1

        # Create CO_OCCURS edges between concepts in same document
        if not dry_run:
            for doc_id, concepts in doc_concepts.items():
                concept_list = list(concepts)
                for i, c1 in enumerate(concept_list):
                    for c2 in concept_list[i+1:]:
                        session.run("""
                            MATCH (c1:CONCEPT {name: $c1})
                            MATCH (c2:CONCEPT {name: $c2})
                            MERGE (c1)-[:CO_OCCURS]-(c2)
                        """, {"c1": c1, "c2": c2})
                        edges_created += 1

        concepts_created = len(all_concepts)

    return papers_created, concepts_created, edges_created


def get_current_neo4j_stats(driver) -> Dict:
    """Get current Neo4j node/edge counts."""
    with driver.session() as session:
        papers = session.run("MATCH (p:Paper) RETURN count(p) as cnt").single()["cnt"]
        concepts = session.run("MATCH (c:CONCEPT) RETURN count(c) as cnt").single()["cnt"]
        mentions = session.run("MATCH ()-[r:MENTIONS]->() RETURN count(r) as cnt").single()["cnt"]
        cooccurs = session.run("MATCH ()-[r:CO_OCCURS]-() RETURN count(r) as cnt").single()["cnt"]

    return {
        "papers": papers,
        "concepts": concepts,
        "mentions": mentions,
        "cooccurs": cooccurs
    }


def main():
    parser = argparse.ArgumentParser(description="Hydrate Neo4j from Postgres")
    parser.add_argument('--dry-run', action='store_true', help="Preview without changes")
    parser.add_argument('--apply', action='store_true', help="Apply changes")
    parser.add_argument('--batch', type=int, default=500, help="Batch size (default: 500)")
    parser.add_argument('--limit', type=int, default=None, help="Max papers to process")
    args = parser.parse_args()

    if not (args.dry_run or args.apply):
        parser.print_help()
        return

    # Connect
    pg_conn = psycopg2.connect(POSTGRES_DSN)
    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    # Show current stats
    stats = get_current_neo4j_stats(neo4j_driver)
    logger.info(f"Current Neo4j: {stats['papers']} papers, {stats['concepts']} concepts, {stats['mentions']} MENTIONS, {stats['cooccurs']} CO_OCCURS")

    # Count total papers
    with pg_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM artifacts WHERE artifact_type = 'paper'")
        total_papers = cur.fetchone()[0]

    limit = args.limit or total_papers
    logger.info(f"Processing {limit} of {total_papers} papers (batch={args.batch})")

    # Process in batches
    total_created = 0
    total_concepts = 0
    total_edges = 0
    offset = 0

    while offset < limit:
        batch_size = min(args.batch, limit - offset)
        papers = get_papers_to_sync(pg_conn, batch_size, offset)

        if not papers:
            break

        created, concepts, edges = sync_papers_to_neo4j(
            neo4j_driver, papers, dry_run=args.dry_run
        )

        total_created += created
        total_concepts += concepts
        total_edges += edges
        offset += len(papers)

        logger.info(f"Progress: {offset}/{limit} papers ({100*offset/limit:.1f}%)")

    # Final stats
    if args.apply:
        final_stats = get_current_neo4j_stats(neo4j_driver)
        logger.info(f"\nFinal Neo4j: {final_stats['papers']} papers, {final_stats['concepts']} concepts")
        logger.info(f"Added: {final_stats['papers'] - stats['papers']} papers, {final_stats['mentions'] - stats['mentions']} MENTIONS")
    else:
        logger.info(f"\nDry run complete. Would process {total_created} papers, {total_concepts} unique concepts, ~{total_edges} edges")

    pg_conn.close()
    neo4j_driver.close()


if __name__ == "__main__":
    main()
