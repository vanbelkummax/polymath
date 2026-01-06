#!/usr/bin/env python3
"""
Export Polymath knowledge base for local AI training.

Outputs:
1. JSONL format for LLM fine-tuning
2. Concept embeddings for RAG
3. Graph structure for GNN training
"""

import json
import os
from pathlib import Path
from typing import List, Dict
from datetime import datetime

CHROMADB_PATH = "/home/user/work/polymax/chromadb/polymath_v2"
NEO4J_URI = "bolt://localhost:7687"
OUTPUT_DIR = "/home/user/work/polymax/data/training_export"


def export_chunks_jsonl(output_path: str, limit: int = None):
    """Export all chunks as JSONL for LLM training."""
    import chromadb

    client = chromadb.PersistentClient(path=CHROMADB_PATH)
    coll = client.get_collection("polymath_corpus")

    # Get all chunks
    total = coll.count()
    batch_size = 1000

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        offset = 0
        while offset < total and (limit is None or offset < limit):
            results = coll.get(
                limit=batch_size,
                offset=offset,
                include=['documents', 'metadatas']
            )

            for doc, meta in zip(results['documents'], results['metadatas']):
                record = {
                    "text": doc,
                    "title": meta.get('title', ''),
                    "source": meta.get('source', ''),
                    "type": meta.get('type', 'paper'),
                    "concepts": meta.get('concepts', '').split(',') if meta.get('concepts') else [],
                }
                f.write(json.dumps(record) + '\n')

            offset += batch_size
            print(f"Exported {min(offset, total)}/{total} chunks")

    print(f"Saved to {output_path}")
    return total


def export_concept_graph(output_path: str):
    """Export concept relationships for GNN training."""
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(NEO4J_URI, auth=("neo4j", "polymathic2026"))

    with driver.session() as session:
        # Export nodes
        nodes = []
        result = session.run("""
            MATCH (n)
            WHERE n:CONCEPT OR n:Paper OR n:Code OR n:METHOD
            RETURN labels(n)[0] as type,
                   coalesce(n.name, n.title) as name,
                   n.file_hash as id
        """)
        for record in result:
            nodes.append({
                "type": record["type"],
                "name": record["name"],
                "id": record["id"] or record["name"]
            })

        # Export edges
        edges = []
        result = session.run("""
            MATCH (a)-[r]->(b)
            RETURN coalesce(a.name, a.title, a.file_hash) as source,
                   type(r) as rel_type,
                   coalesce(b.name, b.title, b.file_hash) as target
        """)
        for record in result:
            edges.append({
                "source": record["source"],
                "type": record["rel_type"],
                "target": record["target"]
            })

    graph = {"nodes": nodes, "edges": edges}

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(graph, f, indent=2)

    print(f"Exported {len(nodes)} nodes, {len(edges)} edges to {output_path}")
    return graph


def export_qa_pairs(output_path: str):
    """Generate Q&A pairs for instruction tuning from concepts."""
    from neo4j import GraphDatabase
    import chromadb
    from sentence_transformers import SentenceTransformer

    driver = GraphDatabase.driver(NEO4J_URI, auth=("neo4j", "polymathic2026"))
    client = chromadb.PersistentClient(path=CHROMADB_PATH)
    coll = client.get_collection("polymath_corpus")
    embedder = SentenceTransformer('all-mpnet-base-v2')

    qa_pairs = []

    with driver.session() as session:
        # Get concepts with papers
        result = session.run("""
            MATCH (c:CONCEPT)<-[:MENTIONS]-(p:Paper)
            WITH c.name as concept, collect(p.title)[0..5] as papers
            WHERE size(papers) > 0
            RETURN concept, papers
            LIMIT 100
        """)

        for record in result:
            concept = record["concept"]
            papers = record["papers"]

            # Search for relevant content (use proper embeddings)
            query_embedding = embedder.encode([concept]).tolist()
            search_results = coll.query(
                query_embeddings=query_embedding,
                n_results=3,
                include=['documents']
            )

            if search_results['documents'][0]:
                context = " ".join(search_results['documents'][0][:2])

                qa_pairs.append({
                    "instruction": f"What is {concept.replace('_', ' ')} and how is it used in research?",
                    "context": context[:1000],
                    "response": f"{concept.replace('_', ' ').title()} is discussed in papers like {', '.join(papers[:3])}. Based on the literature: {context[:500]}...",
                    "concept": concept
                })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for qa in qa_pairs:
            f.write(json.dumps(qa) + '\n')

    print(f"Generated {len(qa_pairs)} Q&A pairs to {output_path}")
    return qa_pairs


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Export Polymath for AI training")
    parser.add_argument("--chunks", action="store_true", help="Export chunks JSONL")
    parser.add_argument("--graph", action="store_true", help="Export concept graph")
    parser.add_argument("--qa", action="store_true", help="Generate Q&A pairs")
    parser.add_argument("--all", action="store_true", help="Export everything")
    parser.add_argument("--limit", type=int, help="Limit chunks export")

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d")

    if args.all or args.chunks:
        export_chunks_jsonl(f"{OUTPUT_DIR}/chunks_{timestamp}.jsonl", args.limit)

    if args.all or args.graph:
        export_concept_graph(f"{OUTPUT_DIR}/graph_{timestamp}.json")

    if args.all or args.qa:
        export_qa_pairs(f"{OUTPUT_DIR}/qa_pairs_{timestamp}.jsonl")

    if not any([args.chunks, args.graph, args.qa, args.all]):
        print("Usage: python export_for_training.py --all")
        print("       python export_for_training.py --chunks --limit 10000")


if __name__ == "__main__":
    main()
