#!/usr/bin/env python3
"""
Enhanced Polymathic Search

Optimized search strategies for cross-domain polymathic queries.
Combines multiple retrieval methods for maximum recall across disciplines.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib.hybrid_search_v2 import HybridSearcherV2
from lib.rosetta_query_expander import expand_query
from typing import List, Dict, Any

# Create global searcher instance
_searcher = None

def _get_searcher():
    global _searcher
    if _searcher is None:
        _searcher = HybridSearcherV2()
    return _searcher


def polymathic_search(
    query: str,
    n_results: int = 20,
    use_rosetta: bool = True,
    multi_hop: bool = True,
    boost_cross_domain: bool = True
) -> Dict[str, Any]:
    """
    Enhanced search optimized for polymathic cross-domain queries.

    Strategy:
    1. Rosetta Stone expansion for vocabulary bridging
    2. Hybrid search (semantic + lexical)
    3. Multi-hop concept exploration (find related concepts)
    4. Cross-domain boosting (prioritize bridges)

    Args:
        query: Search query (can be domain-specific or cross-domain)
        n_results: Number of results to return
        use_rosetta: Use Rosetta Stone query expansion
        multi_hop: Explore related concepts for broader coverage
        boost_cross_domain: Boost results that bridge multiple domains

    Returns:
        Enhanced search results with cross-domain relevance scoring
    """

    # Step 1: Expand query if cross-domain
    expanded_query = query
    expansion_info = {"expanded": False, "terms": []}

    if use_rosetta:
        try:
            expanded_query = expand_query(query)
            if expanded_query != query:
                expansion_info["expanded"] = True
                expansion_info["terms"] = expanded_query.split(" OR ")
        except Exception as e:
            print(f"Warning: Rosetta expansion failed: {e}", file=sys.stderr)

    # Step 2: Execute hybrid search
    searcher = _get_searcher()
    results_obj = searcher.hybrid_search(expanded_query, n=n_results * 2, rerank=False)

    # Convert SearchResult objects to dicts for compatibility
    results = []
    for r in results_obj:
        results.append({
            'id': r.id,
            'text': r.content,
            'metadata': r.metadata,
            'relevance': r.score
        })

    # Step 3: Multi-hop concept exploration (if enabled)
    concept_results = []
    if multi_hop:
        # Extract concepts from initial results
        concepts = set()
        for r in results[:10]:  # Top 10 results
            metadata = r.get('metadata', {})
            if 'concepts' in metadata:
                concepts_raw = metadata.get('concepts', [])
                if isinstance(concepts_raw, str):
                    concepts.update([c.strip() for c in concepts_raw.split(',') if c.strip()])
                else:
                    concepts.update(concepts_raw)

        # Search for each concept to find related content
        for concept in list(concepts)[:5]:  # Limit to top 5 concepts
            try:
                concept_search_obj = searcher.hybrid_search(concept, n=5, rerank=False)
                # Convert SearchResult objects to dicts
                for r in concept_search_obj[:3]:
                    concept_results.append({
                        'id': r.id,
                        'text': r.content,
                        'metadata': r.metadata,
                        'relevance': r.score
                    })
            except:
                pass

    # Step 4: Deduplicate and boost cross-domain results
    seen_ids = set()
    final_results = []

    for result in results + concept_results:
        result_id = result.get('id', result.get('text', '')[:100])
        if result_id in seen_ids:
            continue
        seen_ids.add(result_id)

        # Boost cross-domain content
        if boost_cross_domain:
            metadata = result.get('metadata', {})
            result_type = metadata.get('type', '')

            # Cross-domain indicators
            is_bridge = False
            if 'concepts' in metadata:
                concepts_raw = metadata.get('concepts', [])
                if isinstance(concepts_raw, str):
                    concepts = [c.strip() for c in concepts_raw.split(',') if c.strip()]
                else:
                    concepts = concepts_raw
                # Check if concepts span multiple domains
                domains = set()
                for c in concepts:
                    if any(x in c.lower() for x in ['bio', 'genom', 'cell', 'protein']):
                        domains.add('biology')
                    if any(x in c.lower() for x in ['math', 'topology', 'algebra', 'graph']):
                        domains.add('mathematics')
                    if any(x in c.lower() for x in ['physics', 'thermo', 'entropy', 'energy']):
                        domains.add('physics')
                    if any(x in c.lower() for x in ['learning', 'neural', 'algorithm', 'model']):
                        domains.add('cs')

                is_bridge = len(domains) >= 2

            # Boost bridge content by increasing relevance score
            if is_bridge and 'relevance' in result:
                result['relevance'] *= 1.2
                result['is_bridge'] = True

        final_results.append(result)

    # Sort by relevance and limit
    final_results.sort(key=lambda x: x.get('relevance', 0), reverse=True)
    final_results = final_results[:n_results]

    # Add search metadata
    return {
        "query": query,
        "expanded_query": expanded_query if expansion_info["expanded"] else None,
        "expansion_info": expansion_info,
        "results": final_results,
        "total_found": len(final_results),
        "strategy": {
            "rosetta": use_rosetta,
            "multi_hop": multi_hop,
            "cross_domain_boost": boost_cross_domain
        }
    }


def oblique_search(
    domain_query: str,
    target_domain: str = None,
    n_results: int = 20
) -> Dict[str, Any]:
    """
    Oblique retrieval: Search for abstract patterns across domains.

    Example:
        domain_query = "predict gene expression from histology images"
        target_domain = "signal processing"

        Returns: Signal processing papers on "reconstruct signal from sparse samples"

    Args:
        domain_query: Domain-specific problem description
        target_domain: Target domain to search (optional, auto-detect if None)
        n_results: Number of results

    Returns:
        Search results from analogous problems in other domains
    """

    # Import abstract_problem from polymath_mcp
    from polymath_mcp import abstract_problem

    # Step 1: Abstract the problem
    abstraction = abstract_problem(domain_query)

    # Step 2: Search using abstract terms
    results = []
    for abstract_query in abstraction.get('suggested_queries', [])[:3]:
        search_results = polymathic_search(
            abstract_query,
            n_results=n_results // 3,
            use_rosetta=True,
            multi_hop=True,
            boost_cross_domain=True
        )
        results.extend(search_results['results'])

    # Deduplicate and sort
    seen = set()
    unique_results = []
    for r in results:
        r_id = r.get('id', '')
        if r_id not in seen:
            seen.add(r_id)
            unique_results.append(r)

    unique_results.sort(key=lambda x: x.get('relevance', 0), reverse=True)

    return {
        "original_query": domain_query,
        "abstraction": abstraction,
        "results": unique_results[:n_results],
        "total_found": len(unique_results)
    }


def domain_bridge_search(
    concept1: str,
    concept2: str,
    n_results: int = 20,
    use_hybrid: bool = True
) -> Dict[str, Any]:
    """
    Find papers/code that bridge two concepts from different domains.

    **UPGRADED**: Now uses Hybrid Bridge Search (HyDE + Graph Traversal)
    - Fixes the 0/5 exam failure by finding connections in LOGIC, not VOCABULARY
    - HyDE: Semantic search via hallucinated bridge documents
    - Graph: Symbolic search via Neo4j path traversal
    - Hybrid: Combines both for maximum coverage

    Example:
        concept1 = "optimal transport"
        concept2 = "Waddington landscape"

        OLD (keyword): Looks for both terms in same chunk → 0 results
        NEW (hybrid): Hallucinates "OT formalizes Waddington via manifold flow" →
                      Finds papers on "manifold learning + single-cell" bridging both

    Args:
        concept1: First concept (any domain)
        concept2: Second concept (any domain)
        n_results: Number of results
        use_hybrid: Use HyDE+Graph (True) or fallback to keyword (False)

    Returns:
        Papers/code that connect both concepts
    """

    if use_hybrid:
        # NEW: HyDE + Graph Traversal (fixes 0/5 bridge score)
        try:
            from hybrid_bridge_search import HybridBridgeSearcher
            searcher = HybridBridgeSearcher()
            result = searcher.search(concept1, concept2, n_results=n_results, strategy="hybrid")
            return result
        except Exception as e:
            print(f"Warning: Hybrid bridge search failed ({e}), falling back to keyword search", file=sys.stderr)
            # Fall through to old method

    # FALLBACK: Old keyword-based method (kept for compatibility)
    combined_query = f"{concept1} {concept2}"
    combined_results = polymathic_search(
        combined_query,
        n_results=n_results,
        use_rosetta=True,
        boost_cross_domain=True
    )

    # Filter for results mentioning both concepts
    bridge_results = []
    for r in combined_results['results']:
        text = r.get('text', '').lower()
        metadata_str = str(r.get('metadata', {})).lower()
        combined_text = text + metadata_str

        # Check if both concepts appear (fuzzy match)
        c1_match = any(word in combined_text for word in concept1.lower().split())
        c2_match = any(word in combined_text for word in concept2.lower().split())

        if c1_match and c2_match:
            r['is_bridge'] = True
            r['bridge_concepts'] = [concept1, concept2]
            bridge_results.append(r)

    return {
        "concept1": concept1,
        "concept2": concept2,
        "query": combined_query,
        "strategy": "keyword_fallback",
        "results": bridge_results,
        "total_bridges_found": len(bridge_results)
    }


# CLI for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Polymathic search tool")
    parser.add_argument("query", help="Search query")
    parser.add_argument("-n", "--num", type=int, default=10, help="Number of results")
    parser.add_argument("--mode", choices=["polymathic", "oblique", "bridge"], default="polymathic")
    parser.add_argument("--concept2", help="Second concept (for bridge mode)")

    args = parser.parse_args()

    if args.mode == "polymathic":
        results = polymathic_search(args.query, n_results=args.num)

        print(f"\n=== Polymathic Search: '{args.query}' ===")
        if results.get('expanded_query'):
            print(f"Expanded: {results['expanded_query'][:200]}...")
        print(f"Found: {results['total_found']} results\n")

        for i, r in enumerate(results['results'][:args.num], 1):
            relevance = r.get('relevance', 0)
            is_bridge = r.get('is_bridge', False)
            bridge_marker = "🌉 " if is_bridge else ""

            print(f"{i}. {bridge_marker}[{relevance:.3f}] {r.get('text', '')[:150]}...")
            print()

    elif args.mode == "oblique":
        results = oblique_search(args.query, n_results=args.num)

        print(f"\n=== Oblique Search: '{args.query}' ===")
        print(f"Abstraction: {results['abstraction'].get('abstract_terms', [])}")
        print(f"Found: {results['total_found']} results\n")

        for i, r in enumerate(results['results'][:args.num], 1):
            print(f"{i}. [{r.get('relevance', 0):.3f}] {r.get('text', '')[:150]}...")
            print()

    elif args.mode == "bridge":
        if not args.concept2:
            print("Error: --concept2 required for bridge mode")
            sys.exit(1)

        results = domain_bridge_search(args.query, args.concept2, n_results=args.num)

        print(f"\n=== Bridge Search: '{args.query}' ↔ '{args.concept2}' ===")
        print(f"Found: {results['total_bridges_found']} bridges\n")

        for i, r in enumerate(results['results'][:args.num], 1):
            print(f"{i}. [{r.get('relevance', 0):.3f}] {r.get('text', '')[:150]}...")
            print()
