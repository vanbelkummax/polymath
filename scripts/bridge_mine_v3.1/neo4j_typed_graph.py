#!/usr/bin/env python3
"""
Neo4j Typed Graph Builder for Bridge Mine v4

Builds a typed knowledge graph with:
1. METHOD, PROBLEM, DATASET, EVAL nodes (from passage_concepts)
2. :SOLVES edges (METHOD→PROBLEM based on paper co-occurrence)
3. :SIMILAR_TO edges (PROBLEM→PROBLEM based on embeddings)

This enables gap detection queries like:
    MATCH (m:METHOD)-[:SOLVES]->(p1:PROBLEM)-[:SIMILAR_TO]->(p2:PROBLEM)
    WHERE NOT (m)-[:SOLVES]->(p2)
    RETURN m, p2
"""

import os
import sys
import json
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass

sys.path.insert(0, '/home/user/polymath-repo')

import psycopg2
from neo4j import GraphDatabase
import numpy as np

# Database connections
PG_CONN = "dbname=polymath user=polymath"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "polymathic2026")

# Type mappings from passage_concepts to v3.1 ConceptType
TYPE_MAPPINGS = {
    # METHOD types
    'method': 'METHOD',
    'technique': 'METHOD',
    'algorithm': 'METHOD',
    'model': 'METHOD',
    'architecture': 'METHOD',

    # PROBLEM types (objectives, challenges)
    'objective': 'PROBLEM',

    # DATASET types
    'dataset': 'DATASET',

    # EVAL types (metrics)
    'metric': 'EVAL',

    # These need more careful handling
    'domain': 'DOMAIN',  # Could be PROBLEM or context
    'field': 'DOMAIN',
    'math_object': 'THEORY',
    'prior': 'THEORY',
}

# Filter out generic/noise concepts
NOISE_CONCEPTS = {
    'algorithm', 'algorithms', 'method', 'methods', 'technique', 'techniques',
    'model', 'models', 'approach', 'approaches', 'framework', 'frameworks',
    'system', 'systems', 'analysis', 'data', 'results', 'study', 'research',
    'paper', 'work', 'performance', 'accuracy', 'learning', 'training',
    'testing', 'validation', 'evaluation', 'experiment', 'experiments',
    'dataset', 'datasets', 'metric', 'metrics', 'benchmark', 'benchmarks',
}

# Minimum mentions for a concept to be included
MIN_MENTIONS = 5


@dataclass
class TypedConcept:
    """A concept with its v3.1 type."""
    name: str
    concept_type: str
    mention_count: int
    doc_ids: Set[str]


def get_postgres_conn():
    """Get Postgres connection."""
    return psycopg2.connect(PG_CONN)


def get_neo4j_driver():
    """Get Neo4j driver."""
    return GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)


def extract_typed_concepts(pg_conn) -> Dict[str, TypedConcept]:
    """
    Extract typed concepts from passage_concepts table.

    Returns dict of concept_name -> TypedConcept
    """
    print("Extracting typed concepts from Postgres...")

    cursor = pg_conn.cursor()

    # Get concepts with their types and doc associations
    cursor.execute("""
        SELECT
            pc.concept_name,
            pc.concept_type,
            COUNT(*) as mentions,
            ARRAY_AGG(DISTINCT p.doc_id::text) as doc_ids
        FROM passage_concepts pc
        JOIN passages p ON pc.passage_id = p.passage_id
        WHERE pc.concept_type IN ('method', 'technique', 'algorithm', 'model',
                                   'architecture', 'objective', 'dataset', 'metric')
        GROUP BY pc.concept_name, pc.concept_type
        HAVING COUNT(*) >= %s
        ORDER BY mentions DESC
    """, (MIN_MENTIONS,))

    concepts = {}
    for row in cursor:
        name, pg_type, mentions, doc_ids = row

        # Skip noise
        if name.lower() in NOISE_CONCEPTS:
            continue

        # Map to v3.1 type
        v31_type = TYPE_MAPPINGS.get(pg_type, 'UNKNOWN')

        if name not in concepts:
            concepts[name] = TypedConcept(
                name=name,
                concept_type=v31_type,
                mention_count=mentions,
                doc_ids=set(doc_ids) if doc_ids else set()
            )
        else:
            # Merge if same concept appears with different types
            existing = concepts[name]
            existing.mention_count += mentions
            existing.doc_ids.update(doc_ids if doc_ids else [])
            # Keep the more specific type
            if v31_type in ('METHOD', 'PROBLEM') and existing.concept_type in ('UNKNOWN', 'DOMAIN'):
                existing.concept_type = v31_type

    cursor.close()

    print(f"  Extracted {len(concepts)} typed concepts")

    # Count by type
    type_counts = defaultdict(int)
    for c in concepts.values():
        type_counts[c.concept_type] += 1
    for t, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {t}: {cnt}")

    return concepts


def find_solves_edges(concepts: Dict[str, TypedConcept]) -> List[Tuple[str, str, int]]:
    """
    Find METHOD→PROBLEM edges based on paper co-occurrence.

    A METHOD "solves" a PROBLEM if they appear in the same paper frequently.
    Returns list of (method_name, problem_name, co_occurrence_count)
    """
    print("\nFinding :SOLVES edges (METHOD→PROBLEM co-occurrence)...")

    methods = {n: c for n, c in concepts.items() if c.concept_type == 'METHOD'}
    problems = {n: c for n, c in concepts.items() if c.concept_type == 'PROBLEM'}

    print(f"  {len(methods)} methods, {len(problems)} problems")

    edges = []

    for method_name, method in methods.items():
        for problem_name, problem in problems.items():
            # Count shared documents
            shared_docs = method.doc_ids & problem.doc_ids
            if len(shared_docs) >= 2:  # Require at least 2 shared docs
                edges.append((method_name, problem_name, len(shared_docs)))

    # Sort by co-occurrence count
    edges.sort(key=lambda x: -x[2])

    print(f"  Found {len(edges)} :SOLVES edges")
    if edges:
        print(f"  Top 5:")
        for m, p, cnt in edges[:5]:
            print(f"    {m} → {p} ({cnt} papers)")

    return edges


def find_similar_problems(concepts: Dict[str, TypedConcept], pg_conn) -> List[Tuple[str, str, float]]:
    """
    Find PROBLEM→PROBLEM similarity edges using embeddings.

    Returns list of (problem1, problem2, similarity_score)
    """
    print("\nFinding :SIMILAR_TO edges (PROBLEM→PROBLEM by embedding)...")

    problems = {n: c for n, c in concepts.items() if c.concept_type == 'PROBLEM'}

    if len(problems) < 2:
        print("  Not enough problems for similarity")
        return []

    # Get embeddings from ChromaDB for these concepts
    try:
        import chromadb
        from FlagEmbedding import BGEM3FlagModel

        # Load BGE-M3 model for 1024-dim embeddings (matching ChromaDB collection)
        print("  Loading BGE-M3 model...")
        model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

        client = chromadb.PersistentClient(path="/home/user/polymath-repo/chromadb")
        collection = client.get_collection("polymath_bge_m3")

        # Embed problem concepts using BGE-M3 (1024-dim)
        problem_embeddings = {}
        problem_list = list(problems.keys())

        # Batch embed all problems for efficiency
        print(f"  Embedding {min(len(problem_list), 200)} problem concepts...")
        batch_problems = problem_list[:200]  # Limit for speed
        embeddings_result = model.encode(batch_problems, batch_size=32)
        problem_vecs = embeddings_result['dense_vecs']  # Shape: (n, 1024)

        for i, problem_name in enumerate(batch_problems):
            vec = problem_vecs[i]
            assert len(vec) == 1024, f"Expected 1024-dim, got {len(vec)}"
            problem_embeddings[problem_name] = np.array(vec)

        print(f"  Got {len(problem_embeddings)} problem embeddings (1024-dim BGE-M3)")

        # Compute pairwise similarities
        edges = []
        problem_names = list(problem_embeddings.keys())

        for i, p1 in enumerate(problem_names):
            for p2 in problem_names[i+1:]:
                # Cosine similarity
                e1, e2 = problem_embeddings[p1], problem_embeddings[p2]
                sim = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))

                if sim > 0.7:  # Threshold for similarity
                    edges.append((p1, p2, float(sim)))

        edges.sort(key=lambda x: -x[2])
        print(f"  Found {len(edges)} :SIMILAR_TO edges (threshold > 0.7)")
        if edges:
            print(f"  Top 5:")
            for p1, p2, sim in edges[:5]:
                print(f"    {p1} ~ {p2} ({sim:.3f})")

        return edges

    except Exception as e:
        print(f"  Error getting embeddings: {e}")
        return []


def build_typed_graph(concepts: Dict[str, TypedConcept],
                      solves_edges: List[Tuple[str, str, int]],
                      similar_edges: List[Tuple[str, str, float]],
                      driver):
    """
    Build the typed graph in Neo4j.
    """
    print("\nBuilding typed graph in Neo4j...")

    with driver.session() as session:
        # Create constraints for unique concept names per type
        print("  Creating constraints...")
        for concept_type in ['METHOD', 'PROBLEM', 'DATASET', 'EVAL']:
            try:
                session.run(f"""
                    CREATE CONSTRAINT IF NOT EXISTS
                    FOR (n:{concept_type}) REQUIRE n.name IS UNIQUE
                """)
            except Exception as e:
                print(f"    Constraint for {concept_type}: {e}")

        # Create typed nodes
        print("  Creating typed nodes...")
        batch_size = 500
        concepts_list = list(concepts.values())

        for i in range(0, len(concepts_list), batch_size):
            batch = concepts_list[i:i+batch_size]

            # Group by type
            by_type = defaultdict(list)
            for c in batch:
                by_type[c.concept_type].append({
                    'name': c.name,
                    'mention_count': c.mention_count,
                    'doc_count': len(c.doc_ids)
                })

            for concept_type, nodes in by_type.items():
                if concept_type in ('METHOD', 'PROBLEM', 'DATASET', 'EVAL'):
                    session.run(f"""
                        UNWIND $nodes AS node
                        MERGE (c:{concept_type} {{name: node.name}})
                        SET c.mention_count = node.mention_count,
                            c.doc_count = node.doc_count,
                            c.updated_at = datetime()
                    """, nodes=nodes)

            if (i + batch_size) % 2000 == 0:
                print(f"    Processed {i + batch_size}/{len(concepts_list)} concepts")

        print(f"    Created {len(concepts_list)} typed nodes")

        # Create :SOLVES edges
        print("  Creating :SOLVES edges...")
        for i in range(0, len(solves_edges), batch_size):
            batch = [{'method': m, 'problem': p, 'count': c} for m, p, c in solves_edges[i:i+batch_size]]
            session.run("""
                UNWIND $edges AS edge
                MATCH (m:METHOD {name: edge.method})
                MATCH (p:PROBLEM {name: edge.problem})
                MERGE (m)-[r:SOLVES]->(p)
                SET r.paper_count = edge.count,
                    r.updated_at = datetime()
            """, edges=batch)

        print(f"    Created {len(solves_edges)} :SOLVES edges")

        # Create :SIMILAR_TO edges
        print("  Creating :SIMILAR_TO edges...")
        for i in range(0, len(similar_edges), batch_size):
            batch = [{'p1': p1, 'p2': p2, 'sim': s} for p1, p2, s in similar_edges[i:i+batch_size]]
            session.run("""
                UNWIND $edges AS edge
                MATCH (p1:PROBLEM {name: edge.p1})
                MATCH (p2:PROBLEM {name: edge.p2})
                MERGE (p1)-[r:SIMILAR_TO]->(p2)
                SET r.similarity = edge.sim,
                    r.updated_at = datetime()
            """, edges=batch)

        print(f"    Created {len(similar_edges)} :SIMILAR_TO edges")

        # Report final stats
        result = session.run("""
            MATCH (n)
            WHERE n:METHOD OR n:PROBLEM OR n:DATASET OR n:EVAL
            RETURN labels(n)[0] as label, count(*) as cnt
            ORDER BY cnt DESC
        """)
        print("\n  Final node counts:")
        for r in result:
            print(f"    {r['label']}: {r['cnt']}")

        result = session.run("""
            MATCH ()-[r]->()
            WHERE type(r) IN ['SOLVES', 'SIMILAR_TO']
            RETURN type(r) as rel, count(*) as cnt
        """)
        print("\n  Final edge counts:")
        for r in result:
            print(f"    {r['rel']}: {r['cnt']}")


def run_gap_detection_query(driver) -> List[dict]:
    """
    Run the v4 gap detection query:
    Find methods that solve similar problems but haven't been applied to target problem.
    """
    print("\n" + "="*60)
    print("GAP DETECTION QUERY")
    print("="*60)

    with driver.session() as session:
        result = session.run("""
            MATCH (m:METHOD)-[:SOLVES]->(p1:PROBLEM)-[:SIMILAR_TO]->(p2:PROBLEM)
            WHERE NOT (m)-[:SOLVES]->(p2)
            WITH m, p1, p2,
                 m.mention_count as method_mentions,
                 p1.mention_count as source_mentions,
                 p2.mention_count as target_mentions
            RETURN m.name as method,
                   p1.name as source_problem,
                   p2.name as target_problem,
                   method_mentions,
                   source_mentions,
                   target_mentions
            ORDER BY method_mentions DESC, target_mentions DESC
            LIMIT 20
        """)

        gaps = []
        for r in result:
            gap = dict(r)
            gaps.append(gap)
            print(f"\n  {gap['method']}")
            print(f"    SOLVES: {gap['source_problem']} ({gap['source_mentions']} mentions)")
            print(f"    GAP →: {gap['target_problem']} ({gap['target_mentions']} mentions)")

        if not gaps:
            print("  No gaps found (need more :SOLVES and :SIMILAR_TO edges)")

        return gaps


def main():
    """Build typed graph and run gap detection."""
    print("="*60)
    print("NEO4J TYPED GRAPH BUILDER FOR BRIDGE MINE v4")
    print("="*60)

    # Connect to databases
    pg_conn = get_postgres_conn()
    driver = get_neo4j_driver()

    try:
        # Step 1: Extract typed concepts from Postgres
        concepts = extract_typed_concepts(pg_conn)

        # Step 2: Find :SOLVES edges
        solves_edges = find_solves_edges(concepts)

        # Step 3: Find :SIMILAR_TO edges
        similar_edges = find_similar_problems(concepts, pg_conn)

        # Step 4: Build the graph
        build_typed_graph(concepts, solves_edges, similar_edges, driver)

        # Step 5: Run gap detection
        gaps = run_gap_detection_query(driver)

        # Save gaps for v3.1 validation
        output_file = '/home/user/work/polymax/reports/bridge_mine_v3.1_latest/GAP_CANDIDATES.json'
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(gaps, f, indent=2)
        print(f"\nSaved {len(gaps)} gap candidates to {output_file}")

    finally:
        pg_conn.close()
        driver.close()


if __name__ == "__main__":
    main()
