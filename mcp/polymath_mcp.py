#!/usr/bin/env python3
"""
Polymath Unified MCP Server
Bridges Neo4j (ontology), ChromaDB (semantics), papers.db (metadata), and Brave Search

Tools:
- query_graph: Cypher queries on Neo4j knowledge graph
- semantic_search: Vector search on 77K chunks
- get_paper_metadata: Retrieve paper details from papers.db
- search_web: Brave Search for external knowledge
- cross_reference: Find papers bridging two concepts
- add_to_graph: Enrich the knowledge graph
- generate_hypothesis: Polymathic hypothesis generation
"""

import json
import sys
import os
import sqlite3
import requests
from collections import OrderedDict
from typing import Optional, List, Dict, Any

# Configuration
CHROMADB_PATH = "/home/user/work/polymax/chromadb/polymath_v2"
PAPERS_DB = "/home/user/mcp_servers/polymax-synthesizer/papers.db"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "polymathic2026"
BRAVE_API_KEY = "BSAVT4LaGM0xylZUm1sQ5PzgvlqdJ5A"
ABSTRACTION_PATTERNS_PATH = "/home/user/work/polymax/data/rosetta/abstraction_patterns.json"
_ABSTRACTION_PATTERNS = None

# Lazy imports for performance
_neo4j_driver = None
_chroma_client = None
_chroma_collection = None
_embedding_model = None
_embedding_cache = OrderedDict()
_embedding_cache_size = int(os.environ.get("EMBED_CACHE_SIZE", "256"))


def get_neo4j_driver():
    """Lazy load Neo4j driver."""
    global _neo4j_driver
    if _neo4j_driver is None:
        from neo4j import GraphDatabase
        _neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return _neo4j_driver


def get_chroma_collection():
    """Lazy load ChromaDB collection and embedding model."""
    global _chroma_client, _chroma_collection, _embedding_model
    if _chroma_collection is None:
        import chromadb
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer('all-mpnet-base-v2')
        _chroma_client = chromadb.PersistentClient(path=CHROMADB_PATH)
        _chroma_collection = _chroma_client.get_collection("polymath_corpus")
    return _chroma_collection, _embedding_model


def _encode_query(query: str):
    if _embedding_cache_size <= 0:
        return _embedding_model.encode([query]).tolist()

    cached = _embedding_cache.get(query)
    if cached is not None:
        _embedding_cache.move_to_end(query)
        return cached

    embedding = _embedding_model.encode([query]).tolist()
    _embedding_cache[query] = embedding
    if len(_embedding_cache) > _embedding_cache_size:
        _embedding_cache.popitem(last=False)
    return embedding


def prewarm():
    """Pre-load all resources for fast first-query response."""
    import sys
    print("Pre-warming resources...", file=sys.stderr)

    # Load embedding model and ChromaDB
    print("  Loading embedding model...", file=sys.stderr)
    get_chroma_collection()

    # Warm Neo4j connection
    print("  Connecting to Neo4j...", file=sys.stderr)
    get_neo4j_driver()

    print("  Ready.", file=sys.stderr)


def _load_abstraction_patterns() -> Dict[str, Any]:
    global _ABSTRACTION_PATTERNS
    if _ABSTRACTION_PATTERNS is not None:
        return _ABSTRACTION_PATTERNS

    if not os.path.exists(ABSTRACTION_PATTERNS_PATH):
        _ABSTRACTION_PATTERNS = {}
        return _ABSTRACTION_PATTERNS

    try:
        with open(ABSTRACTION_PATTERNS_PATH, "r", encoding="utf-8") as f:
            _ABSTRACTION_PATTERNS = json.load(f)
    except Exception:
        _ABSTRACTION_PATTERNS = {}

    return _ABSTRACTION_PATTERNS


# ============================================================================
# TOOL 1: Query Knowledge Graph (Neo4j)
# ============================================================================
def query_graph(cypher: str, params: Optional[Dict] = None) -> Dict:
    """
    Execute a Cypher query on the Neo4j knowledge graph.

    Args:
        cypher: Cypher query string
        params: Optional query parameters

    Returns:
        Query results as list of records

    Example queries:
        - Find methods: MATCH (m:METHOD) RETURN m.name LIMIT 10
        - Find paper concepts: MATCH (p:Paper)-[:MENTIONS]->(c:CONCEPT) RETURN p.title, c.name
        - Find connections: MATCH path = (a)-[*1..3]-(b) WHERE a.name = 'UNI' RETURN path LIMIT 5
    """
    try:
        driver = get_neo4j_driver()
        with driver.session() as session:
            result = session.run(cypher, params or {})
            records = []
            for record in result:
                # Convert to dict
                rec_dict = {}
                for key in record.keys():
                    val = record[key]
                    # Handle Node objects
                    if hasattr(val, 'labels'):
                        rec_dict[key] = {
                            'labels': list(val.labels),
                            'properties': dict(val)
                        }
                    # Handle Relationship objects
                    elif hasattr(val, 'type'):
                        rec_dict[key] = {
                            'type': val.type,
                            'properties': dict(val)
                        }
                    else:
                        rec_dict[key] = val
                records.append(rec_dict)
            return {"success": True, "records": records, "count": len(records)}
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_graph_stats() -> Dict:
    """Get statistics about the knowledge graph."""
    stats = query_graph("""
        MATCH (n)
        WITH labels(n)[0] as type, count(*) as count
        RETURN type, count ORDER BY count DESC
    """)

    rels = query_graph("""
        MATCH ()-[r]->()
        WITH type(r) as type, count(*) as count
        RETURN type, count ORDER BY count DESC
    """)

    return {
        "node_types": stats.get("records", []),
        "relationship_types": rels.get("records", [])
    }


# ============================================================================
# TOOL 2: Semantic Search (ChromaDB)
# ============================================================================
def semantic_search(query: str, n_results: int = 10) -> Dict:
    """
    Search the 77K paper chunks by semantic meaning.

    Args:
        query: Natural language query
        n_results: Number of results (default 10)

    Returns:
        Matching documents sorted by relevance
    """
    try:
        collection, _ = get_chroma_collection()
        query_embedding = _encode_query(query)

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=["documents", "distances"]
        )

        formatted = []
        for i, (doc, dist) in enumerate(zip(results["documents"][0], results["distances"][0])):
            formatted.append({
                "rank": i + 1,
                "distance": round(dist, 4),
                "relevance": round(1 - dist, 4),  # Convert distance to similarity
                "excerpt": doc[:500] + "..." if len(doc) > 500 else doc
            })

        return {
            "query": query,
            "results": formatted,
            "total": len(formatted)
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# TOOL 3: Get Paper Metadata (SQLite)
# ============================================================================
def get_paper_metadata(
    pmid: Optional[str] = None,
    title_contains: Optional[str] = None,
    year: Optional[int] = None,
    limit: int = 10
) -> Dict:
    """
    Retrieve paper metadata from the papers database.

    Args:
        pmid: Specific PubMed ID
        title_contains: Search by title substring
        year: Filter by publication year
        limit: Max results

    Returns:
        Paper metadata including title, abstract, authors, journal
    """
    try:
        conn = sqlite3.connect(PAPERS_DB)
        cursor = conn.cursor()

        query = "SELECT pmid, title, abstract, authors, journal, year, citations, doi FROM papers WHERE 1=1"
        params = []

        if pmid:
            query += " AND pmid = ?"
            params.append(pmid)
        if title_contains:
            query += " AND title LIKE ?"
            params.append(f"%{title_contains}%")
        if year:
            query += " AND year = ?"
            params.append(year)

        query += f" LIMIT {limit}"

        cursor.execute(query, params)

        papers = []
        for row in cursor.fetchall():
            papers.append({
                "pmid": row[0],
                "title": row[1],
                "abstract": row[2][:300] + "..." if row[2] and len(row[2]) > 300 else row[2],
                "authors": row[3],
                "journal": row[4],
                "year": row[5],
                "citations": row[6],
                "doi": row[7]
            })

        conn.close()
        return {"papers": papers, "count": len(papers)}
    except Exception as e:
        return {"error": str(e)}


def list_all_papers(limit: int = 100) -> Dict:
    """List all papers in the database."""
    try:
        conn = sqlite3.connect(PAPERS_DB)
        cursor = conn.cursor()
        cursor.execute("SELECT pmid, title, year FROM papers ORDER BY year DESC LIMIT ?", (limit,))
        papers = [{"pmid": r[0], "title": r[1], "year": r[2]} for r in cursor.fetchall()]
        conn.close()
        return {"papers": papers, "total": len(papers)}
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# TOOL 4: Web Search (Brave)
# ============================================================================
def search_web(query: str, count: int = 5) -> Dict:
    """
    Search the web using Brave Search API.

    Args:
        query: Search query
        count: Number of results (max 20)

    Returns:
        Web search results with titles, URLs, descriptions
    """
    try:
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": BRAVE_API_KEY
        }

        response = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": min(count, 20)},
            headers=headers,
            timeout=10
        )

        if response.status_code != 200:
            return {"error": f"Brave API error: {response.status_code}"}

        data = response.json()
        results = []

        for item in data.get("web", {}).get("results", []):
            results.append({
                "title": item.get("title"),
                "url": item.get("url"),
                "description": item.get("description", "")[:200]
            })

        return {"query": query, "results": results, "count": len(results)}
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# TOOL 5: Cross-Reference (Multi-source)
# ============================================================================
def cross_reference(concept1: str, concept2: str) -> Dict:
    """
    Find connections between two concepts using all data sources.

    Args:
        concept1: First concept
        concept2: Second concept

    Returns:
        Graph connections, semantic matches, and web results
    """
    results = {}

    # Check graph
    graph_result = query_graph(f"""
        MATCH path = (a)-[*1..3]-(b)
        WHERE toLower(a.name) CONTAINS toLower($c1)
          AND toLower(b.name) CONTAINS toLower($c2)
        RETURN path LIMIT 5
    """, {"c1": concept1, "c2": concept2})
    results["graph_connections"] = graph_result.get("records", [])

    # Semantic search for both
    semantic = semantic_search(f"{concept1} AND {concept2}", 5)
    results["semantic_matches"] = semantic.get("results", [])

    # Check papers database
    papers1 = get_paper_metadata(title_contains=concept1, limit=5)
    papers2 = get_paper_metadata(title_contains=concept2, limit=5)
    results["related_papers"] = {
        concept1: papers1.get("papers", []),
        concept2: papers2.get("papers", [])
    }

    # Web search if needed
    if len(results["graph_connections"]) == 0 and len(results["semantic_matches"]) < 3:
        web = search_web(f"{concept1} {concept2} connection relationship", 3)
        results["web_search"] = web.get("results", [])

    return results


# ============================================================================
# TOOL 6: Add to Graph (Enrichment)
# ============================================================================
def add_paper_to_graph(
    title: str,
    pmid: Optional[str] = None,
    abstract: Optional[str] = None,
    concepts: Optional[List[str]] = None
) -> Dict:
    """
    Add a paper and its concepts to the knowledge graph.

    Args:
        title: Paper title
        pmid: PubMed ID
        abstract: Paper abstract
        concepts: List of concept names to link

    Returns:
        Created node information
    """
    try:
        driver = get_neo4j_driver()
        with driver.session() as session:
            # Create paper node
            result = session.run("""
                MERGE (p:Paper {title: $title})
                SET p.pmid = $pmid, p.abstract = $abstract
                RETURN p
            """, {"title": title, "pmid": pmid, "abstract": abstract})

            paper_created = result.single() is not None

            # Link concepts
            concept_links = []
            for concept in (concepts or []):
                session.run("""
                    MATCH (p:Paper {title: $title})
                    MERGE (c:CONCEPT {name: $concept})
                    MERGE (p)-[:MENTIONS]->(c)
                """, {"title": title, "concept": concept})
                concept_links.append(concept)

            return {
                "success": True,
                "paper_created": paper_created,
                "concepts_linked": concept_links
            }
    except Exception as e:
        return {"success": False, "error": str(e)}


def add_relationship(node1_name: str, node2_name: str, relationship: str) -> Dict:
    """Add a relationship between two existing nodes."""
    try:
        driver = get_neo4j_driver()
        with driver.session() as session:
            result = session.run(f"""
                MATCH (a), (b)
                WHERE a.name = $n1 OR a.title = $n1
                  AND (b.name = $n2 OR b.title = $n2)
                MERGE (a)-[r:{relationship}]->(b)
                RETURN a, r, b
            """, {"n1": node1_name, "n2": node2_name})
            return {"success": True, "created": result.single() is not None}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================================
# TOOL 7: Generate Hypothesis
# ============================================================================
def abstract_problem(problem: str) -> Dict:
    """
    Abstract a domain-specific problem into a general systems problem.
    This enables "oblique retrieval" - finding cross-domain solutions.

    Args:
        problem: Domain-specific problem description

    Returns:
        Abstracted problem and suggested search queries
    """
    pattern_data = _load_abstraction_patterns()
    patterns = pattern_data.get("patterns") if isinstance(pattern_data, dict) else {}
    query_templates = pattern_data.get("query_templates") if isinstance(pattern_data, dict) else []
    max_terms = pattern_data.get("max_terms", 6) if isinstance(pattern_data, dict) else 6

    if not patterns:
        patterns = {
            "predict": ["infer hidden state from partial observation"],
            "expression": ["high dimensional signal"],
            "image": ["low dimensional observation"],
            "cell type": ["latent categorical variable"],
            "spatial": ["structured geometric constraint"],
            "deconvolution": ["unmixing superimposed signals"],
            "segmentation": ["partition continuous space"],
            "classification": ["map observation to category"],
            "gene": ["sparse binary program"],
            "protein": ["functional molecule concentration"],
        }

    problem_lower = problem.lower()
    abstract_terms = []
    for term, abstractions in patterns.items():
        if term.lower() in problem_lower:
            if isinstance(abstractions, list):
                abstract_terms.extend(abstractions)
            else:
                abstract_terms.append(str(abstractions))

    seen = set()
    unique_terms = []
    for term in abstract_terms:
        if term not in seen:
            seen.add(term)
            unique_terms.append(term)

    abstract_terms = unique_terms[:max_terms] if unique_terms else []

    if not query_templates:
        query_templates = [
            "{a} sparse",
            "{a} information theory",
            "compressed sensing {b}",
        ]

    a_term = abstract_terms[0] if abstract_terms else "inverse problem"
    b_term = abstract_terms[1] if len(abstract_terms) > 1 else a_term

    suggested_queries = []
    for template in query_templates:
        try:
            suggested = template.format(a=a_term, b=b_term)
        except Exception:
            continue
        if suggested and suggested not in suggested_queries:
            suggested_queries.append(suggested)

    return {
        "original": problem,
        "abstract_terms": abstract_terms,
        "suggested_queries": suggested_queries,
        "instruction": "Search for these abstract terms to find cross-domain solutions"
    }


def generate_hypothesis(domain: str = "spatial_biology") -> Dict:
    """
    Generate research hypotheses by finding unexplored concept pairs.
    Now uses cross-domain concepts for true polymathic discovery.

    Args:
        domain: Focus domain (spatial_biology, machine_learning, systems_biology)

    Returns:
        Ranked hypotheses with supporting evidence
    """
    # Find cross-domain concepts not connected to biology papers
    gaps = query_graph("""
        MATCH (a:CONCEPT)
        WHERE a.domain = 'cross_domain'
        MATCH (b:CONCEPT)
        WHERE NOT (a)-[]-(b) AND a <> b
        AND (b.domain IS NULL OR b.domain <> 'cross_domain')
        WITH a, b, rand() as r
        ORDER BY r
        LIMIT 10
        RETURN a.name as concept1, a.description as desc1, b.name as concept2
    """)

    hypotheses = []
    for gap in gaps.get("records", []):
        c1 = gap.get("concept1")
        desc1 = gap.get("desc1", "")
        c2 = gap.get("concept2")
        if not c1 or not c2:
            continue

        # Check if there's semantic evidence
        semantic = semantic_search(f"{c1} {c2}", 3)
        has_evidence = len(semantic.get("results", [])) > 0

        hypotheses.append({
            "concept1": c1,
            "concept2": c2,
            "cross_domain_concept": c1,
            "cross_domain_description": desc1,
            "biology_concept": c2,
            "hypothesis": f"Apply {c1} ({desc1[:50]}...) to {c2}",
            "has_semantic_evidence": has_evidence,
            "evidence_preview": semantic.get("results", [])[0]["excerpt"][:200] if has_evidence else None
        })

    return {
        "domain": domain,
        "hypotheses": hypotheses[:5],
        "method": "unexplored_pairs"
    }


# ============================================================================
# TOOL 8: Unified Ingestion
# ============================================================================
def ingest_paper(path: str) -> Dict:
    """
    Ingest a single PDF into all Polymath stores (ChromaDB + Neo4j + Postgres).

    Args:
        path: Path to PDF file

    Returns:
        Ingestion result with chunks added and concepts linked
    """
    try:
        # Import unified ingestor
        import sys
        sys.path.insert(0, "/home/user/work/polymax/lib")
        from unified_ingest import UnifiedIngestor

        ingestor = UnifiedIngestor()
        result = ingestor.ingest_pdf(path)

        return {
            "success": result.chunks_added > 0,
            "artifact_id": result.artifact_id,
            "title": result.title,
            "chunks_added": result.chunks_added,
            "concepts_linked": result.concepts_linked,
            "neo4j_synced": result.neo4j_node_created,
            "postgres_synced": result.postgres_synced,
            "errors": result.errors
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def ingest_batch(directory: str, move_after: bool = False) -> Dict:
    """
    Ingest all PDFs from a directory into all Polymath stores.

    Args:
        directory: Path to directory with PDFs
        move_after: Move processed files to 'processed/' subfolder

    Returns:
        Batch result with success/failure counts
    """
    try:
        import sys
        sys.path.insert(0, "/home/user/work/polymax/lib")
        from unified_ingest import UnifiedIngestor

        ingestor = UnifiedIngestor()
        result = ingestor.ingest_directory(directory, pattern="*.pdf", move_after=move_after)

        return {
            "success": True,
            "total_files": result.total_files,
            "successful": result.successful,
            "failed": result.failed,
            "chunks_added": result.chunks_added,
            "errors": result.errors
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def ingest_code_file(path: str, repo_name: str = None) -> Dict:
    """
    Ingest a code file into Polymath stores.

    Args:
        path: Path to code file
        repo_name: Optional repository name for context

    Returns:
        Ingestion result
    """
    try:
        import sys
        sys.path.insert(0, "/home/user/work/polymax/lib")
        from unified_ingest import UnifiedIngestor

        ingestor = UnifiedIngestor()
        result = ingestor.ingest_code(path, repo_name)

        return {
            "success": result.chunks_added > 0,
            "artifact_id": result.artifact_id,
            "title": result.title,
            "chunks_added": result.chunks_added,
            "concepts_linked": result.concepts_linked,
            "neo4j_synced": result.neo4j_node_created,
            "errors": result.errors
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_system_stats() -> Dict:
    """
    Get comprehensive Polymath system statistics.

    Returns:
        Stats from all stores (ChromaDB, Neo4j, Postgres)
    """
    stats = {}

    # ChromaDB stats
    try:
        coll, _ = get_chroma_collection()
        stats["chromadb"] = {
            "total_chunks": coll.count(),
            "collection": "polymath_corpus"
        }
    except Exception as e:
        stats["chromadb"] = {"error": str(e)}

    # Neo4j stats
    try:
        graph_stats = get_graph_stats()
        stats["neo4j"] = graph_stats
    except Exception as e:
        stats["neo4j"] = {"error": str(e)}

    # Postgres stats (if available)
    try:
        import psycopg2
        conn = psycopg2.connect(os.environ.get("POSTGRES_URL", "postgresql://localhost:5432/polymath"))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM artifacts")
        artifact_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM chunks")
        chunk_count = cursor.fetchone()[0]
        stats["postgres"] = {
            "artifacts": artifact_count,
            "chunks": chunk_count
        }
        conn.close()
    except Exception:
        stats["postgres"] = {"status": "not_configured"}

    return stats


# ============================================================================
# MCP Protocol Handler
# ============================================================================
def handle_request(request: dict) -> dict:
    """Handle MCP tool requests."""
    tool = request.get("tool")
    params = request.get("params", {})

    tools = {
        "query_graph": lambda p: query_graph(p.get("cypher", ""), p.get("params")),
        "get_graph_stats": lambda p: get_graph_stats(),
        "semantic_search": lambda p: semantic_search(p.get("query", ""), p.get("n_results", 10)),
        "get_paper_metadata": lambda p: get_paper_metadata(
            pmid=p.get("pmid"),
            title_contains=p.get("title_contains"),
            year=p.get("year"),
            limit=p.get("limit", 10)
        ),
        "list_papers": lambda p: list_all_papers(p.get("limit", 100)),
        "search_web": lambda p: search_web(p.get("query", ""), p.get("count", 5)),
        "cross_reference": lambda p: cross_reference(p.get("concept1", ""), p.get("concept2", "")),
        "add_paper": lambda p: add_paper_to_graph(
            p.get("title", ""),
            p.get("pmid"),
            p.get("abstract"),
            p.get("concepts", [])
        ),
        "add_relationship": lambda p: add_relationship(
            p.get("node1", ""),
            p.get("node2", ""),
            p.get("relationship", "RELATES_TO")
        ),
        "generate_hypothesis": lambda p: generate_hypothesis(p.get("domain", "spatial_biology")),
        "abstract_problem": lambda p: abstract_problem(p.get("problem", "")),
        "ingest_paper": lambda p: ingest_paper(p.get("path", "")),
        "ingest_batch": lambda p: ingest_batch(p.get("directory", ""), p.get("move", False)),
        "ingest_code": lambda p: ingest_code_file(p.get("path", ""), p.get("repo_name")),
        "system_stats": lambda p: get_system_stats(),
    }

    if tool in tools:
        return tools[tool](params)
    else:
        return {"error": f"Unknown tool: {tool}. Available: {list(tools.keys())}"}


# ============================================================================
# Test Mode
# ============================================================================
def run_tests():
    """Run comprehensive tests."""
    print("=" * 60)
    print("POLYMATH MCP SERVER - SYSTEM TEST")
    print("=" * 60)

    # Test 1: Graph Stats
    print("\n1. KNOWLEDGE GRAPH STATS:")
    stats = get_graph_stats()
    for node_type in stats.get("node_types", []):
        print(f"   {node_type['type']}: {node_type['count']} nodes")

    # Test 2: Semantic Search
    print("\n2. SEMANTIC SEARCH (spatial deconvolution):")
    results = semantic_search("spatial deconvolution cell type", 3)
    for r in results.get("results", []):
        print(f"   [{r['relevance']:.3f}] {r['excerpt'][:80]}...")

    # Test 3: Paper Metadata
    print("\n3. PAPER METADATA (pathology):")
    papers = get_paper_metadata(title_contains="pathology", limit=3)
    for p in papers.get("papers", []):
        print(f"   [{p['year']}] {p['title'][:60]}...")

    # Test 4: Web Search
    print("\n4. BRAVE WEB SEARCH (Visium HD 2025):")
    web = search_web("Visium HD spatial transcriptomics 2025", 3)
    for w in web.get("results", []):
        print(f"   {w['title'][:50]}...")
        print(f"      {w['url']}")

    # Test 5: Cross-reference
    print("\n5. CROSS-REFERENCE (UNI + deconvolution):")
    xref = cross_reference("UNI", "deconvolution")
    print(f"   Graph connections: {len(xref.get('graph_connections', []))}")
    print(f"   Semantic matches: {len(xref.get('semantic_matches', []))}")

    # Test 6: Hypothesis Generation
    print("\n6. HYPOTHESIS GENERATION:")
    hyp = generate_hypothesis()
    for h in hyp.get("hypotheses", [])[:3]:
        print(f"   • {h['concept1']} ↔ {h['concept2']}")
        print(f"     Evidence: {'Yes' if h['has_semantic_evidence'] else 'No'}")

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_tests()
    elif len(sys.argv) > 1 and sys.argv[1] == "prewarm":
        # Pre-warm only mode
        prewarm()
        print("Pre-warm complete. Exiting.", file=sys.stderr)
    else:
        # MCP mode - read from stdin
        # Pre-warm resources for fast first query
        prewarm()
        print("Polymath MCP Server ready. Reading from stdin...", file=sys.stderr)
        for line in sys.stdin:
            try:
                request = json.loads(line)
                response = handle_request(request)
                print(json.dumps(response))
                sys.stdout.flush()
            except json.JSONDecodeError as e:
                print(json.dumps({"error": f"Invalid JSON: {e}"}))
                sys.stdout.flush()
