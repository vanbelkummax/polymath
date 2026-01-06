#!/usr/bin/env python3
"""
Ingest OCR Results - Load extracted text into ChromaDB.
Run this AFTER ocr_to_files.py completes.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime

# No GPU needed
os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

import chromadb
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase

OCR_DIR = Path("/home/user/work/polymax/ocr_extracted")
CHROMADB_PATH = "/home/user/work/polymax/chromadb/polymath_v2"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "polymathic2026"

CONCEPTS = {
    "deep learning": "deep_learning", "neural network": "neural_network",
    "machine learning": "machine_learning", "information theory": "information_theory",
    "entropy": "entropy", "compressed sensing": "compressed_sensing",
    "thermodynamics": "thermodynamics", "cybernetics": "cybernetics",
    "complexity": "complexity", "evolution": "evolution",
    "causal": "causal_inference", "bayesian": "bayesian_inference",
    "quantum": "quantum", "nonlinear": "nonlinear_dynamics",
    "self-organization": "self_organization", "emergence": "emergence",
}


def extract_title(text: str, filename: str) -> str:
    for line in text.split('\n')[:30]:
        line = line.strip().strip('#').strip()
        if 15 < len(line) < 250 and not line.startswith(('http', '!', '|')):
            return line
    return filename.replace('_', ' ').replace('-', ' ')


def chunk_text(text: str, size: int = 1500, overlap: int = 200, max_chunks: int = 50) -> list:
    chunks = []
    for i in range(0, len(text), size - overlap):
        chunk = text[i:i+size]
        if len(chunk) > 100:
            chunks.append(chunk)
        if len(chunks) >= max_chunks:
            break
    return chunks


def extract_concepts(text: str) -> list:
    text_lower = text.lower()[:10000]
    return list(set(c for k, c in CONCEPTS.items() if k in text_lower))


def main():
    print("=" * 60)
    print("INGEST OCR RESULTS")
    print("=" * 60)

    txt_files = list(OCR_DIR.glob("*.txt"))
    print(f"Found {len(txt_files)} OCR result files\n")

    if not txt_files:
        print("No files to ingest. Run ocr_to_files.py first.")
        return

    # Connect
    print("Loading embedding model...")
    embed_model = SentenceTransformer('all-mpnet-base-v2')

    print("Connecting to databases...")
    client = chromadb.PersistentClient(path=CHROMADB_PATH)
    coll = client.get_or_create_collection("polymath_corpus")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    print(f"ChromaDB: {coll.count()} existing chunks\n")

    total_chunks = 0

    for i, txt_file in enumerate(txt_files, 1):
        text = txt_file.read_text()
        meta_file = OCR_DIR / f"{txt_file.stem}.json"

        if meta_file.exists():
            meta = json.loads(meta_file.read_text())
            source = meta.get("source", txt_file.name)
        else:
            source = txt_file.stem + ".pdf"

        title = extract_title(text, txt_file.stem)
        chunks = chunk_text(text)
        concepts = extract_concepts(text)

        print(f"[{i}/{len(txt_files)}] {source[:40]}...")
        print(f"    Title: {title[:50]}...")
        print(f"    Chunks: {len(chunks)}, Concepts: {len(concepts)}")

        # Add to ChromaDB
        file_hash = hashlib.md5(source.encode()).hexdigest()[:10]
        ids = [f"{file_hash}_{j}" for j in range(len(chunks))]

        embeddings = embed_model.encode(chunks).tolist()
        coll.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=[{"source": source, "title": title[:200], "method": "ocr"} for _ in chunks]
        )

        # Add to Neo4j
        with driver.session() as session:
            session.run("MERGE (p:Paper {title: $t}) SET p.source='ocr_ingest'", {"t": title[:200]})
            for c in concepts[:10]:
                session.run("""
                    MATCH (p:Paper {title: $t})
                    MERGE (c:CONCEPT {name: $c})
                    MERGE (p)-[:MENTIONS]->(c)
                """, {"t": title[:200], "c": c})

        total_chunks += len(chunks)

    driver.close()

    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    print(f"Files processed: {len(txt_files)}")
    print(f"Total chunks added: {total_chunks}")
    print(f"ChromaDB total: {coll.count()}")


if __name__ == "__main__":
    main()
