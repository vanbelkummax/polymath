#!/usr/bin/env python3
"""
ChromaDB RAG MCP Server for PolyMax
Provides semantic search over the paper corpus

Tools:
- semantic_search: Search papers by meaning
- get_paper_chunks: Get all chunks for a paper
- find_similar_papers: Find papers similar to a query
"""

import json
import sys
import os
import chromadb
from sentence_transformers import SentenceTransformer

# Configuration - fresh ChromaDB on Linux ext4 (fast!)
CHROMADB_BASE = "/home/user/work/polymax/chromadb"
COLLECTION_PATHS = [
    (f"{CHROMADB_BASE}/polymath_v2", "polymath_corpus"),  # Migrated fresh collection
]

# Initialize embedding model
print("Loading embedding model...", file=sys.stderr)
model = SentenceTransformer('all-mpnet-base-v2')

def semantic_search(query: str, n_results: int = 10, collection_name: str = None) -> dict:
    """
    Search the paper corpus by semantic meaning.

    Args:
        query: Natural language query
        n_results: Number of results to return (default: 10)
        collection_name: Optional specific collection to search

    Returns:
        Matching documents with metadata
    """
    # Embed query
    query_embedding = model.encode([query]).tolist()

    all_results = []

    for db_path, coll_name in COLLECTION_PATHS:
        if collection_name and coll_name != collection_name:
            continue

        if not os.path.exists(db_path):
            continue

        try:
            client = chromadb.PersistentClient(path=db_path)
            collection = client.get_collection(coll_name)

            results = collection.query(
                query_embeddings=query_embedding,
                n_results=min(n_results, 5),  # Limit per collection
                include=["documents", "metadatas", "distances"]
            )

            for i, doc in enumerate(results["documents"][0]):
                all_results.append({
                    "collection": coll_name,
                    "document": doc[:500] + "..." if len(doc) > 500 else doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0
                })
        except Exception as e:
            print(f"Error searching {coll_name}: {e}", file=sys.stderr)
            continue

    # Sort by distance (lower = more similar)
    all_results.sort(key=lambda x: x.get("distance", 999))

    return {
        "query": query,
        "results": all_results[:n_results],
        "total_found": len(all_results)
    }

def get_corpus_stats() -> dict:
    """
    Get statistics about the RAG corpus.

    Returns:
        Collection sizes and total documents
    """
    stats = {}
    total = 0

    for db_path, coll_name in COLLECTION_PATHS:
        if not os.path.exists(db_path):
            continue

        try:
            client = chromadb.PersistentClient(path=db_path)
            collection = client.get_collection(coll_name)
            count = collection.count()
            stats[coll_name] = count
            total += count
        except Exception as e:
            stats[coll_name] = f"Error: {e}"

    return {"collections": stats, "total_chunks": total}

def find_similar_to_text(text: str, n_results: int = 5) -> dict:
    """
    Find papers similar to a given text snippet.

    Args:
        text: Text to find similar content for
        n_results: Number of results

    Returns:
        Similar documents
    """
    return semantic_search(text, n_results)

def get_papers_about(topic: str) -> dict:
    """
    Get papers specifically about a topic.

    Args:
        topic: Topic to search for

    Returns:
        Papers mentioning this topic with relevant excerpts
    """
    results = semantic_search(topic, n_results=15)

    # Group by paper
    papers = {}
    for r in results["results"]:
        paper = r.get("metadata", {}).get("paper", "Unknown")
        if paper not in papers:
            papers[paper] = {
                "excerpts": [],
                "collection": r["collection"]
            }
        papers[paper]["excerpts"].append(r["document"][:200])

    return {
        "topic": topic,
        "papers": papers,
        "paper_count": len(papers)
    }

def cross_reference(concept1: str, concept2: str) -> dict:
    """
    Find papers that discuss both concepts.

    Args:
        concept1: First concept
        concept2: Second concept

    Returns:
        Papers mentioning both concepts
    """
    # Search for combined query
    combined_query = f"{concept1} and {concept2}"
    results = semantic_search(combined_query, n_results=10)

    # Filter for papers mentioning both
    filtered = []
    for r in results["results"]:
        doc_lower = r["document"].lower()
        if concept1.lower() in doc_lower and concept2.lower() in doc_lower:
            filtered.append(r)

    return {
        "concept1": concept1,
        "concept2": concept2,
        "cross_references": filtered,
        "count": len(filtered)
    }


# MCP Protocol Handler
def handle_request(request: dict) -> dict:
    """Handle MCP tool requests"""
    tool = request.get("tool")
    params = request.get("params", {})

    if tool == "semantic_search":
        return semantic_search(
            params.get("query", ""),
            params.get("n_results", 10),
            params.get("collection", None)
        )
    elif tool == "get_corpus_stats":
        return get_corpus_stats()
    elif tool == "find_similar":
        return find_similar_to_text(params.get("text", ""), params.get("n_results", 5))
    elif tool == "get_papers_about":
        return get_papers_about(params.get("topic", ""))
    elif tool == "cross_reference":
        return cross_reference(params.get("concept1", ""), params.get("concept2", ""))
    else:
        return {"error": f"Unknown tool: {tool}"}


if __name__ == "__main__":
    # Test mode
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("Testing RAG MCP Server...")
        print("\n1. Corpus Stats:")
        print(json.dumps(get_corpus_stats(), indent=2))

        print("\n2. Semantic Search (reaction-diffusion pattern formation):")
        results = semantic_search("reaction-diffusion pattern formation", 3)
        print(json.dumps(results, indent=2))

        print("\n3. Cross-reference (morphogenesis + neural network):")
        results = cross_reference("morphogenesis", "neural network")
        print(json.dumps(results, indent=2))
    else:
        # MCP mode - read from stdin
        for line in sys.stdin:
            request = json.loads(line)
            response = handle_request(request)
            print(json.dumps(response))
            sys.stdout.flush()
