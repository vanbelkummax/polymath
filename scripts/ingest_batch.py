#!/usr/bin/env python3
"""
Fast batch PDF ingestion for Polymath system
Uses FAST ext4 path: /home/user/work/polymax/chromadb/polymath_v2
"""

import os
import sys
import json
import hashlib
import signal
import gc
from datetime import datetime
from pathlib import Path

# FAST PATHS ONLY
CHROMADB_PATH = "/home/user/work/polymax/chromadb/polymath_v2"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "polymathic2026"
PROGRESS_FILE = "/home/user/work/polymax/ingest_progress.json"

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("timeout")

def extract_pdf_text(pdf_path, max_pages=50, timeout_sec=30):
    """Extract text with timeout."""
    import fitz
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_sec)
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for i in range(min(len(doc), max_pages)):
            text += doc[i].get_text()
            if len(text) > 100000:
                break
        doc.close()
        signal.alarm(0)
        return text
    except:
        signal.alarm(0)
        return None

def chunk_text(text, size=1500, overlap=200, max_chunks=20):
    """Chunk with overlap."""
    chunks = []
    for i in range(0, len(text), size - overlap):
        chunk = text[i:i+size]
        if len(chunk) > 100:
            chunks.append(chunk)
        if len(chunks) >= max_chunks:
            break
    return chunks

def extract_title(text, filename):
    """Get title from first lines or filename."""
    for line in text.split('\n')[:10]:
        line = line.strip()
        if 20 < len(line) < 200 and not line.startswith('http'):
            return line
    return Path(filename).stem.replace('_', ' ')

# Concept keywords for Neo4j
CONCEPTS = {
    "deep learning": "deep_learning", "neural network": "neural_network",
    "machine learning": "machine_learning", "transformer": "transformer",
    "spatial transcriptomics": "spatial_transcriptomics", "single-cell": "single_cell",
    "pathology": "pathology", "cancer": "cancer", "tumor": "tumor",
    "information theory": "information_theory", "entropy": "entropy",
    "compressed sensing": "compressed_sensing", "game theory": "game_theory",
    "evolution": "evolution", "natural selection": "natural_selection",
    "cybernetics": "cybernetics", "complexity": "complexity",
    "thermodynamics": "thermodynamics", "causal": "causal_inference",
    "segregation": "schelling_model", "red queen": "red_queen",
    "co-evolution": "coevolution", "feedback": "feedback_control",
    "self-organization": "self_organization", "emergence": "emergence",
    "bayesian": "bayesian_inference", "markov": "markov_chain",
}

def extract_concepts(text):
    """Extract matching concepts."""
    text_lower = text.lower()
    return list(set(c for k, c in CONCEPTS.items() if k in text_lower))

def main(pdf_dir):
    if pdf_dir.startswith('/mnt/'):
        print("ERROR: Use Linux path, not Windows!")
        sys.exit(1)

    import chromadb
    from sentence_transformers import SentenceTransformer
    from neo4j import GraphDatabase

    # Load progress
    processed = set()
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            processed = set(json.load(f).get('processed', []))

    # Get PDFs
    pdfs = list(Path(pdf_dir).glob("*.pdf"))
    remaining = [p for p in pdfs if p.name not in processed]
    print(f"[{datetime.now():%H:%M}] {len(remaining)}/{len(pdfs)} PDFs to process")

    # Init
    print("Loading model...")
    model = SentenceTransformer('all-mpnet-base-v2')
    client = chromadb.PersistentClient(path=CHROMADB_PATH)
    coll = client.get_or_create_collection("polymath_corpus", metadata={"hnsw:space": "cosine"})
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    print(f"ChromaDB: {coll.count()} chunks")

    added_chunks = 0
    added_papers = 0

    for i, pdf in enumerate(remaining, 1):
        print(f"[{i}/{len(remaining)}] {pdf.name[:50]}...", end=" ", flush=True)

        text = extract_pdf_text(str(pdf))
        if not text:
            print("SKIP (no text)")
            processed.add(pdf.name)
            continue

        title = extract_title(text, pdf.name)
        chunks = chunk_text(text)
        concepts = extract_concepts(text)
        file_hash = hashlib.md5(pdf.name.encode()).hexdigest()[:10]

        # Add to ChromaDB
        ids = [f"{file_hash}_{j}" for j in range(len(chunks))]
        try:
            existing = coll.get(ids=ids[:1])
            if not existing['ids']:
                embeddings = model.encode(chunks).tolist()
                coll.add(
                    ids=ids,
                    documents=chunks,
                    embeddings=embeddings,
                    metadatas=[{"source": pdf.name, "title": title[:200]} for _ in chunks]
                )
                added_chunks += len(chunks)
        except Exception as e:
            print(f"ChromaDB error: {e}")

        # Add to Neo4j
        try:
            with driver.session() as session:
                session.run("MERGE (p:Paper {title: $t}) SET p.source='ingest'", {"t": title[:200]})
                for c in concepts:
                    session.run("""
                        MATCH (p:Paper {title: $t})
                        MERGE (c:CONCEPT {name: $c})
                        MERGE (p)-[:MENTIONS]->(c)
                    """, {"t": title[:200], "c": c})
            added_papers += 1
        except Exception as e:
            print(f"Neo4j error: {e}")

        processed.add(pdf.name)
        print(f"OK ({len(chunks)} chunks, {len(concepts)} concepts)")

        # Save progress every 10
        if i % 10 == 0:
            with open(PROGRESS_FILE, 'w') as f:
                json.dump({"processed": list(processed)}, f)
            gc.collect()

    driver.close()

    # Final save
    with open(PROGRESS_FILE, 'w') as f:
        json.dump({"processed": list(processed)}, f)

    print(f"\n{'='*50}")
    print(f"DONE: {added_papers} papers, {added_chunks} chunks added")
    print(f"ChromaDB total: {coll.count()}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 ingest_batch.py /path/to/pdfs/")
        sys.exit(1)
    main(sys.argv[1])
