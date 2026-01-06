#!/usr/bin/env python3
"""
Graph-Based Bridge Search (Neo4j Multi-Hop)

Finds domain bridges via symbolic path traversal in the knowledge graph.

Core Insight: If Paper A (Math) → Paper B (General) ← Paper C (Bio),
then B is the bridge, even if A and C use totally different vocabularies.
"""

import os
from neo4j import GraphDatabase
from typing import List, Dict, Tuple


class GraphBridgeSearcher:
    """
    Neo4j-based domain bridge discovery via multi-hop traversal.

    Strategy:
    1. Find concepts/papers related to concept_a (Math domain)
    2. Find concepts/papers related to concept_b (Bio domain)
    3. Find paths connecting them (1-3 hops)
    4. Rank by path strength and semantic centrality
    """

    def __init__(self):
        self._driver = None

    def _get_driver(self):
        if self._driver is None:
            uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
            password = os.environ.get("NEO4J_PASSWORD", "polymathic2026")
            self._driver = GraphDatabase.driver(uri, auth=("neo4j", password))
        return self._driver

    def find_bridge_paths(
        self,
        concept_a: str,
        concept_b: str,
        max_hops: int = 3,
        n_results: int = 10
    ) -> List[Dict]:
        """
        Find shortest paths connecting two concepts in the knowledge graph.

        Args:
            concept_a: First concept (e.g., "optimal_transport")
            concept_b: Second concept (e.g., "cell_differentiation")
            max_hops: Maximum path length (default: 3)
            n_results: Number of bridge nodes to return

        Returns:
            List of bridge papers/concepts with path information
        """

        driver = self._get_driver()

        # Normalize concept names (replace spaces with underscores)
        concept_a_norm = concept_a.lower().replace(" ", "_")
        concept_b_norm = concept_b.lower().replace(" ", "_")

        bridges = []

        with driver.session() as session:
            # Strategy 1: Find concepts that bridge both domains
            query_concept_bridge = """
            MATCH path = (ca:CONCEPT)-[*1..3]-(bridge:CONCEPT)-[*1..3]-(cb:CONCEPT)
            WHERE toLower(ca.name) CONTAINS $concept_a
              AND toLower(cb.name) CONTAINS $concept_b
              AND ca <> cb
            WITH bridge, path, length(path) as path_length
            ORDER BY path_length ASC
            LIMIT $limit
            RETURN DISTINCT bridge.name as bridge_name,
                   path_length,
                   'concept' as bridge_type
            """

            try:
                result = session.run(
                    query_concept_bridge,
                    concept_a=concept_a_norm,
                    concept_b=concept_b_norm,
                    limit=n_results
                )

                for record in result:
                    bridges.append({
                        'bridge_name': record['bridge_name'],
                        'path_length': record['path_length'],
                        'bridge_type': record['bridge_type'],
                        'score': 1.0 / record['path_length']  # Shorter = better
                    })
            except Exception as e:
                print(f"Warning: Concept bridge query failed: {e}")

            # Strategy 2: Find papers that mention both concept neighborhoods
            query_paper_bridge = """
            MATCH (ca:CONCEPT)-[:MENTIONS]-(paper_a:Paper)-[*0..2]-(bridge:Paper)-[*0..2]-(paper_b:Paper)-[:MENTIONS]-(cb:CONCEPT)
            WHERE toLower(ca.name) CONTAINS $concept_a
              AND toLower(cb.name) CONTAINS $concept_b
              AND ca <> cb
              AND bridge <> paper_a
              AND bridge <> paper_b
            WITH bridge, COUNT(*) as connection_strength
            ORDER BY connection_strength DESC
            LIMIT $limit
            RETURN bridge.title as bridge_name,
                   connection_strength,
                   'paper' as bridge_type,
                   bridge.id as bridge_id
            """

            try:
                result = session.run(
                    query_paper_bridge,
                    concept_a=concept_a_norm,
                    concept_b=concept_b_norm,
                    limit=n_results
                )

                for record in result:
                    bridges.append({
                        'bridge_name': record['bridge_name'],
                        'connection_strength': record['connection_strength'],
                        'bridge_type': record['bridge_type'],
                        'bridge_id': record.get('bridge_id', 'unknown'),
                        'score': float(record['connection_strength']) / 10.0  # Normalize
                    })
            except Exception as e:
                print(f"Warning: Paper bridge query failed: {e}")

            # Strategy 3: Find shared ancestor concepts (review papers, foundational concepts)
            query_ancestor_bridge = """
            MATCH (ca:CONCEPT)<-[:MENTIONS]-(paper_a:Paper)-[:CITES|USES*1..2]->(ancestor:Paper)<-[:CITES|USES*1..2]-(paper_b:Paper)-[:MENTIONS]->(cb:CONCEPT)
            WHERE toLower(ca.name) CONTAINS $concept_a
              AND toLower(cb.name) CONTAINS $concept_b
              AND ca <> cb
            WITH ancestor, COUNT(DISTINCT paper_a) + COUNT(DISTINCT paper_b) as centrality
            ORDER BY centrality DESC
            LIMIT $limit
            RETURN ancestor.title as bridge_name,
                   centrality,
                   'ancestor_paper' as bridge_type,
                   ancestor.id as bridge_id
            """

            try:
                result = session.run(
                    query_ancestor_bridge,
                    concept_a=concept_a_norm,
                    concept_b=concept_b_norm,
                    limit=n_results
                )

                for record in result:
                    bridges.append({
                        'bridge_name': record['bridge_name'],
                        'centrality': record['centrality'],
                        'bridge_type': record['bridge_type'],
                        'bridge_id': record.get('bridge_id', 'unknown'),
                        'score': float(record['centrality']) / 20.0  # Normalize
                    })
            except Exception as e:
                print(f"Warning: Ancestor bridge query failed: {e}")

        # Deduplicate and sort by score
        seen = set()
        unique_bridges = []
        for b in bridges:
            key = b.get('bridge_id', b['bridge_name'])
            if key not in seen:
                seen.add(key)
                unique_bridges.append(b)

        unique_bridges.sort(key=lambda x: x['score'], reverse=True)

        return unique_bridges[:n_results]

    def expand_concept_neighborhood(
        self,
        concept: str,
        n_related: int = 10
    ) -> List[str]:
        """
        Find concepts related to the given concept (for query expansion).

        Args:
            concept: Concept name
            n_related: Number of related concepts to return

        Returns:
            List of related concept names
        """

        driver = self._get_driver()
        concept_norm = concept.lower().replace(" ", "_")

        related = []

        with driver.session() as session:
            query = """
            MATCH (c:CONCEPT)-[r:RELATES_TO|CO_OCCURS]-(related:CONCEPT)
            WHERE toLower(c.name) CONTAINS $concept
            WITH related, COUNT(r) as strength
            ORDER BY strength DESC
            LIMIT $limit
            RETURN related.name as concept_name, strength
            """

            try:
                result = session.run(query, concept=concept_norm, limit=n_related)
                for record in result:
                    related.append(record['concept_name'])
            except Exception as e:
                print(f"Warning: Neighborhood expansion failed: {e}")

        return related


# CLI for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Graph Bridge Search")
    parser.add_argument("concept_a", help="First concept")
    parser.add_argument("concept_b", help="Second concept")
    parser.add_argument("-n", "--num", type=int, default=10, help="Number of results")
    parser.add_argument("--max-hops", type=int, default=3, help="Maximum path length")

    args = parser.parse_args()

    searcher = GraphBridgeSearcher()

    print(f"\n=== Graph Bridge Search: '{args.concept_a}' ↔ '{args.concept_b}' ===")
    print(f"Max hops: {args.max_hops}\n")

    bridges = searcher.find_bridge_paths(
        args.concept_a,
        args.concept_b,
        max_hops=args.max_hops,
        n_results=args.num
    )

    if not bridges:
        print("No bridge paths found.")
    else:
        print(f"Found {len(bridges)} bridge nodes:\n")
        for i, b in enumerate(bridges, 1):
            print(f"{i}. [{b['bridge_type']:15s}] Score: {b['score']:.3f}")
            print(f"   {b['bridge_name']}")
            if 'path_length' in b:
                print(f"   Path length: {b['path_length']}")
            if 'connection_strength' in b:
                print(f"   Connections: {b['connection_strength']}")
            if 'centrality' in b:
                print(f"   Centrality: {b['centrality']}")
            print()
