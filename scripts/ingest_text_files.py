#!/usr/bin/env python3
"""
Ingest plain text files (OCR output) into Polymath.
"""

import os
import sys
import hashlib
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

CHROMA_PATH = "/home/user/work/polymax/chromadb/polymath_v2"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_PASSWORD = "polymathic2026"

# Cross-domain concepts
CROSS_DOMAIN_CONCEPTS = {
    "compressed_sensing", "sparse_coding", "information_bottleneck", "wavelet",
    "entropy", "free_energy", "maximum_entropy", "thermodynamics", "diffusion",
    "causal_inference", "counterfactual", "bayesian",
    "feedback", "control_theory", "homeostasis", "autopoiesis", "emergence",
    "self_organization", "cybernetics", "pattern", "morphogenesis",
    "predictive_coding", "bayesian_brain", "active_inference",
    "neural_network", "deep_learning", "transformer", "attention",
    "quantum", "turing", "computation", "automata", "cellular_automata",
    "gradient_descent", "optimization", "variational",
    "bifurcation", "chaos", "attractor", "dynamical_system", "nonlinear",
    "regression", "classification", "cross_validation", "bootstrap",
}


def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> list:
    """Split text into overlapping chunks."""
    import re
    text = re.sub(r'\s+', ' ', text).strip()

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            for sep in ['. ', '.\n', '? ', '! ']:
                last_sep = text.rfind(sep, max(end - 200, start), end)
                if last_sep > start:
                    end = last_sep + 1
                    break

        chunk = text[start:end].strip()
        if len(chunk) > 50:
            chunks.append(chunk)
        start = end - overlap

    return chunks


def extract_concepts(text: str) -> list:
    """Find cross-domain concepts in text."""
    text_lower = text.lower()
    found = []
    for concept in CROSS_DOMAIN_CONCEPTS:
        if concept.replace('_', ' ') in text_lower or concept in text_lower:
            found.append(concept)
    return list(set(found))


def ingest_text_file(filepath: str, title: str = None):
    """Ingest a single text file."""
    import chromadb
    from sentence_transformers import SentenceTransformer
    from neo4j import GraphDatabase

    filepath = Path(filepath)
    if not title:
        title = filepath.stem.replace('_', ' ')

    print(f"Ingesting: {title}")

    # Read content
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    file_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

    # Chunk
    chunks = chunk_text(content)
    print(f"  Chunks: {len(chunks)}")

    # Extract concepts
    concepts = extract_concepts(content)
    print(f"  Concepts: {concepts}")

    # Embed and add to ChromaDB
    embedder = SentenceTransformer('all-mpnet-base-v2')
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection("polymath_corpus")

    chunk_ids = [f"textbook_{file_hash}_{i}" for i in range(len(chunks))]
    embeddings = embedder.encode(chunks).tolist()
    metadatas = [
        {
            "title": title,
            "source": str(filepath),
            "chunk_index": i,
            "file_hash": file_hash,
            "concepts": ",".join(concepts),
            "type": "textbook",
            "ingested_at": datetime.now().isoformat(),
        }
        for i in range(len(chunks))
    ]

    collection.add(ids=chunk_ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
    print(f"  Added to ChromaDB")

    # Add to Neo4j
    driver = GraphDatabase.driver(NEO4J_URI, auth=("neo4j", NEO4J_PASSWORD))
    with driver.session() as session:
        session.run("""
            MERGE (p:Paper {file_hash: $hash})
            SET p.title = $title, p.type = 'textbook', p.chunks = $chunks, p.ingested_at = datetime()
        """, hash=file_hash, title=title, chunks=len(chunks))

        for concept in concepts:
            session.run("""
                MATCH (p:Paper {file_hash: $hash})
                MERGE (c:CONCEPT {name: $concept})
                MERGE (p)-[:MENTIONS]->(c)
            """, hash=file_hash, concept=concept)

    driver.close()
    print(f"  Added to Neo4j with {len(concepts)} concept links")

    return len(chunks), concepts


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Text file or directory")
    args = parser.parse_args()

    path = Path(args.path)
    total_chunks = 0
    all_concepts = set()

    if path.is_file():
        chunks, concepts = ingest_text_file(path)
        total_chunks += chunks
        all_concepts.update(concepts)
    elif path.is_dir():
        for txt in path.glob("*.txt"):
            chunks, concepts = ingest_text_file(txt)
            total_chunks += chunks
            all_concepts.update(concepts)
        for md in path.glob("*.md"):
            chunks, concepts = ingest_text_file(md)
            total_chunks += chunks
            all_concepts.update(concepts)

    print(f"\n{'='*50}")
    print(f"Total chunks: {total_chunks}")
    print(f"Unique concepts: {len(all_concepts)}")


if __name__ == "__main__":
    main()
