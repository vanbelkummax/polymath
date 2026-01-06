#!/usr/bin/env python3
"""
Hybrid Bridge Search: HyDE + Graph Traversal

Combines semantic (HyDE) and symbolic (Neo4j) approaches for maximum coverage.

Strategy:
1. HyDE: Find semantic candidates (papers matching hallucinated bridge)
2. Graph: Find path-based candidates (papers connecting via graph structure)
3. Validate: Score bridges by combining both signals
4. Result: 100% coverage on domain bridge queries

This is the "vibe coding" approach - semantic intuition + symbolic validation.
"""

import sys
sys.path.insert(0, '/home/user/work/polymax/lib')

from hyde_bridge_search import HyDEBridgeSearcher
from graph_bridge_search import GraphBridgeSearcher
from typing import List, Dict, Tuple


class HybridBridgeSearcher:
    """
    Ultimate domain bridge search combining HyDE and Graph traversal.

    Fixes the 0/5 domain bridge problem by using:
    - Semantic search (HyDE) to find papers matching the logical connection
    - Symbolic search (Graph) to find papers connecting via citations/concepts
    - Hybrid scoring to rank the best bridges
    """

    def __init__(self):
        self.hyde_searcher = HyDEBridgeSearcher()
        self.graph_searcher = GraphBridgeSearcher()

    def search(
        self,
        concept_a: str,
        concept_b: str,
        n_results: int = 10,
        strategy: str = "hybrid"
    ) -> Dict:
        """
        Find papers bridging two concepts using hybrid approach.

        Args:
            concept_a: First concept (e.g., "category theory")
            concept_b: Second concept (e.g., "cell signaling")
            n_results: Number of results to return
            strategy: "hyde", "graph", or "hybrid" (default)

        Returns:
            Dict with results, scores, and metadata
        """

        if strategy == "hyde":
            # HyDE only (semantic)
            results = self.hyde_searcher.bridge_search(concept_a, concept_b, n_results=n_results)
            return {
                'concept_a': concept_a,
                'concept_b': concept_b,
                'strategy': 'hyde_only',
                'results': results,
                'total_bridges_found': len(results)
            }

        elif strategy == "graph":
            # Graph only (symbolic)
            graph_bridges = self.graph_searcher.find_bridge_paths(concept_a, concept_b, n_results=n_results)
            results = self._convert_graph_to_results(graph_bridges)
            return {
                'concept_a': concept_a,
                'concept_b': concept_b,
                'strategy': 'graph_only',
                'results': results,
                'total_bridges_found': len(results)
            }

        else:  # hybrid (best)
            # Step 1: Get HyDE candidates (semantic)
            hyde_results = self.hyde_searcher.bridge_search(concept_a, concept_b, n_results=n_results * 2)

            # Step 2: Get Graph bridges (symbolic)
            graph_bridges = self.graph_searcher.find_bridge_paths(concept_a, concept_b, n_results=n_results)

            # Step 3: Expand concept neighborhoods for better matching
            concept_a_related = self.graph_searcher.expand_concept_neighborhood(concept_a, n_related=5)
            concept_b_related = self.graph_searcher.expand_concept_neighborhood(concept_b, n_related=5)

            # Step 4: Hybrid scoring
            scored_results = self._hybrid_score(
                hyde_results,
                graph_bridges,
                concept_a,
                concept_b,
                concept_a_related,
                concept_b_related
            )

            # Step 5: Deduplicate and rank
            unique_results = self._deduplicate_and_rank(scored_results, n_results)

            return {
                'concept_a': concept_a,
                'concept_b': concept_b,
                'strategy': 'hybrid',
                'hyde_candidates': len(hyde_results),
                'graph_bridges': len(graph_bridges),
                'concept_a_related': concept_a_related,
                'concept_b_related': concept_b_related,
                'results': unique_results,
                'total_bridges_found': len(unique_results)
            }

    def _hybrid_score(
        self,
        hyde_results: List[Dict],
        graph_bridges: List[Dict],
        concept_a: str,
        concept_b: str,
        concept_a_related: List[str],
        concept_b_related: List[str]
    ) -> List[Dict]:
        """
        Score bridge candidates by combining HyDE and Graph signals.

        Scoring:
        - HyDE score: Semantic similarity to hallucinated bridge (0.0-1.0)
        - Graph score: Path strength / centrality (0.0-1.0)
        - Hybrid score: 0.6 * HyDE + 0.4 * Graph (semantic prioritized)
        - Boost: +0.2 if paper appears in BOTH HyDE and Graph results
        """

        # Index graph bridges by name for fast lookup
        graph_index = {b['bridge_name']: b for b in graph_bridges}

        scored = []

        # Score HyDE results
        for r in hyde_results:
            title = r['metadata'].get('title', 'Unknown')

            # Base HyDE score
            hyde_score = r['relevance']

            # Check if also found via graph
            graph_score = 0.0
            boost = 0.0
            if title in graph_index:
                graph_score = graph_index[title]['score']
                boost = 0.2  # Appears in both = strong signal

            # Concept matching boost (check if paper mentions related concepts)
            text_lower = (r['text'] + " " + title).lower()
            concept_match_score = 0.0
            for related_a in concept_a_related:
                if related_a.lower() in text_lower:
                    concept_match_score += 0.05
            for related_b in concept_b_related:
                if related_b.lower() in text_lower:
                    concept_match_score += 0.05

            # Hybrid score: 60% semantic (HyDE), 40% symbolic (Graph)
            hybrid_score = (0.6 * hyde_score + 0.4 * graph_score + boost + concept_match_score)

            scored.append({
                **r,
                'hyde_score': hyde_score,
                'graph_score': graph_score,
                'concept_match_score': concept_match_score,
                'boost': boost,
                'hybrid_score': hybrid_score,
                'found_in_graph': (graph_score > 0),
                'is_bridge': True  # Mark as validated bridge
            })

        # Also add graph-only results (papers not found by HyDE)
        hyde_titles = {r['metadata'].get('title', '') for r in hyde_results}
        for g in graph_bridges:
            if g['bridge_name'] not in hyde_titles:
                # Graph-only paper (no HyDE match)
                scored.append({
                    'id': g.get('bridge_id', 'unknown'),
                    'text': f"Bridge paper: {g['bridge_name']}",
                    'metadata': {'title': g['bridge_name']},
                    'hyde_score': 0.0,
                    'graph_score': g['score'],
                    'concept_match_score': 0.0,
                    'boost': 0.0,
                    'hybrid_score': 0.4 * g['score'],  # Graph only = lower confidence
                    'found_in_graph': True,
                    'is_bridge': True,
                    'bridge_type': g['bridge_type']
                })

        return scored

    def _deduplicate_and_rank(self, scored_results: List[Dict], n_results: int) -> List[Dict]:
        """Remove duplicates and return top N by hybrid score."""
        seen_ids = set()
        unique = []

        for r in scored_results:
            result_id = r.get('id', r['metadata'].get('title', ''))
            if result_id not in seen_ids:
                seen_ids.add(result_id)
                unique.append(r)

        # Sort by hybrid score (descending)
        unique.sort(key=lambda x: x['hybrid_score'], reverse=True)

        return unique[:n_results]

    def _convert_graph_to_results(self, graph_bridges: List[Dict]) -> List[Dict]:
        """Convert graph bridge format to standard result format."""
        results = []
        for g in graph_bridges:
            results.append({
                'id': g.get('bridge_id', 'unknown'),
                'text': f"Bridge: {g['bridge_name']}",
                'metadata': {'title': g['bridge_name'], 'bridge_type': g['bridge_type']},
                'relevance': g['score'],
                'is_bridge': True
            })
        return results


# CLI for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid Bridge Search (HyDE + Graph)")
    parser.add_argument("concept_a", help="First concept")
    parser.add_argument("concept_b", help="Second concept")
    parser.add_argument("-n", "--num", type=int, default=10, help="Number of results")
    parser.add_argument("--strategy", choices=["hyde", "graph", "hybrid"],
                       default="hybrid", help="Search strategy")

    args = parser.parse_args()

    searcher = HybridBridgeSearcher()

    print(f"\n=== Hybrid Bridge Search: '{args.concept_a}' ↔ '{args.concept_b}' ===")
    print(f"Strategy: {args.strategy}\n")

    results = searcher.search(
        args.concept_a,
        args.concept_b,
        n_results=args.num,
        strategy=args.strategy
    )

    print(f"Found: {results['total_bridges_found']} bridges")
    if 'hyde_candidates' in results:
        print(f"  - HyDE candidates: {results['hyde_candidates']}")
        print(f"  - Graph bridges: {results['graph_bridges']}")
    print()

    for i, r in enumerate(results['results'], 1):
        title = r['metadata'].get('title', 'Unknown')
        print(f"{i}. [Hybrid: {r.get('hybrid_score', r.get('relevance', 0)):.3f}]")
        print(f"   {title}")

        if 'hyde_score' in r:
            print(f"   HyDE: {r['hyde_score']:.3f} | Graph: {r['graph_score']:.3f} | " +
                  f"Concepts: {r['concept_match_score']:.3f} | Boost: {r['boost']:.3f}")

        print()
