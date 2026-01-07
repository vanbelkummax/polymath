#!/usr/bin/env python3
"""
Polymath CLI - Command-line interface for the Polymath research system
Designed for use with Codex, Claude Code, or any CLI-based AI assistant.

Usage:
    python3 polymath_cli.py search "spatial deconvolution"
    python3 polymath_cli.py graph "MATCH (p:Paper) RETURN p.title LIMIT 5"
    python3 polymath_cli.py papers --title "pathology" --limit 10
    python3 polymath_cli.py web "Visium HD 2025 benchmarks"
    python3 polymath_cli.py crossref "UNI" "Cell2location"
    python3 polymath_cli.py hypothesis
    python3 polymath_cli.py stats
"""

import argparse
import json
import sys
import os
from pathlib import Path

# Add MCP directory to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "mcp"))
sys.path.insert(0, str(ROOT))  # Add repo root for lib imports

# Import monitoring decorator
try:
    from monitoring.metrics_collector import track_query
    MONITORING_ENABLED = True
except ImportError:
    # Monitoring not available, create no-op decorator
    def track_query(operation):
        def decorator(func):
            return func
        return decorator
    MONITORING_ENABLED = False

# Import Rosetta Stone
try:
    sys.path.insert(0, str(ROOT / "lib"))
    from rosetta_query_expander import expand_query_with_llm
    ROSETTA_AVAILABLE = True
except Exception:
    ROSETTA_AVAILABLE = False
    expand_query_with_llm = None

def main():
    parser = argparse.ArgumentParser(
        description='Polymath CLI - Unified research interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s search "transformer spatial transcriptomics"
  %(prog)s graph "MATCH (c:CONCEPT) RETURN c.name LIMIT 20"
  %(prog)s papers --title "foundation model" --year 2024
  %(prog)s web "Cell2location benchmark 2025"
  %(prog)s crossref "deep learning" "deconvolution"
  %(prog)s hypothesis --domain spatial_biology
  %(prog)s stats
  %(prog)s add-paper --title "My Paper" --concepts "deep_learning,segmentation"
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Search command (semantic search on ChromaDB)
    search_parser = subparsers.add_parser('search', help='Semantic search on 77K paper chunks')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('-n', '--num', type=int, default=5, help='Number of results')
    search_parser.add_argument('--expand', action='store_true', help='Auto-expand query with cross-domain synonyms (Rosetta Stone)')

    # Graph command (Neo4j Cypher query)
    graph_parser = subparsers.add_parser('graph', help='Query Neo4j knowledge graph')
    graph_parser.add_argument('cypher', help='Cypher query')

    # Papers command (SQLite metadata search)
    papers_parser = subparsers.add_parser('papers', help='Search paper metadata')
    papers_parser.add_argument('--title', help='Title contains')
    papers_parser.add_argument('--pmid', help='PubMed ID')
    papers_parser.add_argument('--year', type=int, help='Publication year')
    papers_parser.add_argument('--limit', type=int, default=10, help='Max results')

    # Web command (Brave search)
    web_parser = subparsers.add_parser('web', help='Search the web via Brave')
    web_parser.add_argument('query', help='Search query')
    web_parser.add_argument('-n', '--num', type=int, default=5, help='Number of results')

    # Crossref command (multi-source cross-reference)
    crossref_parser = subparsers.add_parser('crossref', help='Cross-reference two concepts')
    crossref_parser.add_argument('concept1', help='First concept')
    crossref_parser.add_argument('concept2', help='Second concept')

    # Hypothesis command
    hyp_parser = subparsers.add_parser('hypothesis', help='Generate research hypotheses')
    hyp_parser.add_argument('--domain', default='spatial_biology', help='Research domain')

    # Stats command
    subparsers.add_parser('stats', help='Show system statistics')

    # Abstract command (oblique retrieval)
    abstract_parser = subparsers.add_parser('abstract', help='Abstract a problem for oblique retrieval')
    abstract_parser.add_argument('problem', help='Domain-specific problem description')

    # Add paper command
    add_parser = subparsers.add_parser('add-paper', help='Add a paper to the knowledge graph')
    add_parser.add_argument('--title', required=True, help='Paper title')
    add_parser.add_argument('--pmid', help='PubMed ID')
    add_parser.add_argument('--abstract', help='Paper abstract')
    add_parser.add_argument('--concepts', help='Comma-separated concepts')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Import polymath functions (lazy load for speed)
    from polymath_mcp import (
        semantic_search, query_graph, get_paper_metadata,
        search_web, cross_reference, generate_hypothesis,
        get_graph_stats, add_paper_to_graph, abstract_problem
    )

    # Execute command
    if args.command == 'search':
        # Rosetta Stone Expansion
        original_query = args.query
        expanded_terms = []
        search_query = original_query
        if args.expand:
            if ROSETTA_AVAILABLE and expand_query_with_llm:
                print(f"Expanding query: '{original_query}'...")
                expanded_terms = expand_query_with_llm(original_query)
                if expanded_terms:
                    search_query = f"{original_query} " + " ".join(expanded_terms)
                    print(f"Expanded terms: {', '.join(expanded_terms)}\n")
                else:
                    print("No expansion terms found; using original query.\n")
            else:
                print("Warning: Rosetta Stone not available, using original query.\n")

        # Wrap with monitoring
        if MONITORING_ENABLED:
            @track_query('semantic_search')
            def monitored_search(query, num):
                return semantic_search(query, num)
            result = monitored_search(search_query, args.num)
        else:
            result = semantic_search(search_query, args.num)

        print(f"\n=== Semantic Search: '{original_query}' ===\n")
        for r in result.get('results', []):
            print(f"[{r['relevance']:.3f}] {r['excerpt'][:200]}...")
            print()

    elif args.command == 'graph':
        # Wrap with monitoring
        if MONITORING_ENABLED:
            @track_query('graph_query')
            def monitored_graph(cypher):
                return query_graph(cypher)
            result = monitored_graph(args.cypher)
        else:
            result = query_graph(args.cypher)

        print(f"\n=== Graph Query ===\n")
        if result.get('success'):
            for record in result.get('records', []):
                print(json.dumps(record, indent=2, default=str))
        else:
            print(f"Error: {result.get('error')}")

    elif args.command == 'papers':
        result = get_paper_metadata(
            pmid=args.pmid,
            title_contains=args.title,
            year=args.year,
            limit=args.limit
        )
        print(f"\n=== Paper Search ===\n")
        for p in result.get('papers', []):
            print(f"[{p['year']}] {p['title']}")
            print(f"    PMID: {p['pmid']} | DOI: {p['doi']}")
            if p.get('abstract'):
                print(f"    Abstract: {p['abstract'][:150]}...")
            print()

    elif args.command == 'web':
        result = search_web(args.query, args.num)
        print(f"\n=== Web Search: '{args.query}' ===\n")
        for w in result.get('results', []):
            print(f"• {w['title']}")
            print(f"  {w['url']}")
            print(f"  {w['description'][:100]}...")
            print()

    elif args.command == 'crossref':
        # Wrap with monitoring
        if MONITORING_ENABLED:
            @track_query('cross_reference')
            def monitored_crossref(concept1, concept2):
                return cross_reference(concept1, concept2)
            result = monitored_crossref(args.concept1, args.concept2)
        else:
            result = cross_reference(args.concept1, args.concept2)

        print(f"\n=== Cross-Reference: {args.concept1} ↔ {args.concept2} ===\n")
        print(f"Graph connections: {len(result.get('graph_connections', []))}")
        print(f"Semantic matches: {len(result.get('semantic_matches', []))}")
        if result.get('web_search'):
            print(f"Web results: {len(result.get('web_search', []))}")
        print()
        for match in result.get('semantic_matches', [])[:3]:
            print(f"  • {match['excerpt'][:150]}...")

    elif args.command == 'hypothesis':
        result = generate_hypothesis(args.domain)
        print(f"\n=== Hypotheses for {args.domain} ===\n")
        for h in result.get('hypotheses', []):
            print(f"• {h['concept1']} ↔ {h['concept2']}")
            print(f"  {h['hypothesis']}")
            print(f"  Evidence: {'Yes' if h['has_semantic_evidence'] else 'No'}")
            print()

    elif args.command == 'stats':
        result = get_graph_stats()
        print("\n=== Polymath System Stats ===\n")
        print("Node Types:")
        for node in result.get('node_types', []):
            print(f"  {node['type']}: {node['count']}")
        print("\nRelationship Types:")
        for rel in result.get('relationship_types', []):
            print(f"  {rel['type']}: {rel['count']}")

        # Also show ChromaDB stats
        import chromadb
        from lib.config import (
            CHROMADB_PATH, PAPERS_COLLECTION, CODE_COLLECTION,
            PAPERS_COLLECTION_LEGACY, CODE_COLLECTION_LEGACY
        )
        from pathlib import Path

        legacy_path = Path("/home/user/work/polymax/chromadb")
        client = chromadb.PersistentClient(path=str(CHROMADB_PATH))
        collections = client.list_collections()
        collection_names = [c.name for c in collections]

        if not collection_names and legacy_path.exists() and CHROMADB_PATH != legacy_path:
            client = chromadb.PersistentClient(path=str(legacy_path))
            collections = client.list_collections()
            collection_names = [c.name for c in collections]

        if collection_names:
            papers_name = PAPERS_COLLECTION if PAPERS_COLLECTION in collection_names else PAPERS_COLLECTION_LEGACY
            code_name = CODE_COLLECTION if CODE_COLLECTION in collection_names else CODE_COLLECTION_LEGACY
            if papers_name in collection_names:
                papers = client.get_collection(papers_name)
                print(f"\nChromaDB Papers: {papers.count()}")
            if code_name in collection_names:
                code = client.get_collection(code_name)
                print(f"ChromaDB Code: {code.count()}")
        else:
            print("\nChromaDB: no collections found")

    elif args.command == 'abstract':
        result = abstract_problem(args.problem)
        print(f"\n=== Abstract Problem ===\n")
        print(f"Input: {args.problem}\n")
        print("Abstract terms:")
        for term in result.get('abstract_terms', []):
            print(f"  • {term}")
        print("\nSuggested search queries:")
        for q in result.get('suggested_queries', []):
            print(f"  • {q}")
        print("\nNext step: Run these searches with 'polymath_cli.py search \"<query>\"'")

    elif args.command == 'add-paper':
        concepts = args.concepts.split(',') if args.concepts else []
        result = add_paper_to_graph(
            title=args.title,
            pmid=args.pmid,
            abstract=args.abstract,
            concepts=concepts
        )
        print(f"\n=== Add Paper ===\n")
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
