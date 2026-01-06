#!/usr/bin/env python3
"""
Sequential OCR ingestion - one PDF at a time (safe, won't crash)
Uses Marker API with GPU
"""

import os
import sys
import json
import hashlib
import gc
import torch
from datetime import datetime
from pathlib import Path

CHROMADB_PATH = "/home/user/work/polymax/chromadb/polymath_v2"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "polymathic2026"
PROGRESS_FILE = "/home/user/work/polymax/ingest_ocr_seq_progress.json"


def extract_with_fitz(pdf_path):
    """Fast text extraction for text-based PDFs."""
    import fitz
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text if len(text.strip()) > 500 else None
    except:
        return None


def extract_with_marker(pdf_path, models, converter):
    """OCR with Marker (GPU)."""
    from marker.output import text_from_rendered
    rendered = converter(str(pdf_path))
    text, _, _ = text_from_rendered(rendered)
    return text


def chunk_text(text, size=1500, overlap=200, max_chunks=50):
    chunks = []
    for i in range(0, len(text), size - overlap):
        chunk = text[i:i+size]
        if len(chunk) > 100:
            chunks.append(chunk)
        if len(chunks) >= max_chunks:
            break
    return chunks


def extract_title(text, filename):
    for line in text.split('\n')[:30]:
        line = line.strip().strip('#').strip()
        if 15 < len(line) < 250 and not line.startswith(('http', '!', '|', '-', '*')):
            return line
    return Path(filename).stem.replace('_', ' ')


CONCEPTS = {
    "deep learning": "deep_learning", "neural network": "neural_network",
    "machine learning": "machine_learning", "information theory": "information_theory",
    "entropy": "entropy", "compressed sensing": "compressed_sensing",
    "game theory": "game_theory", "evolution": "evolution",
    "natural selection": "natural_selection", "cybernetics": "cybernetics",
    "complexity": "complexity", "thermodynamics": "thermodynamics",
    "causal": "causal_inference", "self-organization": "self_organization",
    "emergence": "emergence", "bayesian": "bayesian_inference",
    "quantum": "quantum", "nonlinear": "nonlinear_dynamics",
    "chaos": "chaos_theory", "statistical": "statistical_learning",
    "stochastic": "stochastic_process", "automata": "cellular_automata",
    "pattern": "pattern_formation", "regression": "regression",
    "optimization": "optimization", "self-reproducing": "self_reproduction",
    "computation": "computation_theory", "turing": "turing_machine",
}


def extract_concepts(text):
    text_lower = text.lower()
    return list(set(c for k, c in CONCEPTS.items() if k in text_lower))


def main(pdf_dir):
    if pdf_dir.startswith('/mnt/'):
        print("ERROR: Copy to Linux first!")
        sys.exit(1)

    import chromadb
    from sentence_transformers import SentenceTransformer
    from neo4j import GraphDatabase

    # Load progress
    processed = set()
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            processed = set(json.load(f).get('processed', []))

    pdfs = list(Path(pdf_dir).glob("*.pdf"))
    remaining = [p for p in pdfs if p.name not in processed]

    print(f"[{datetime.now():%H:%M}] {len(remaining)}/{len(pdfs)} PDFs to process")

    if not remaining:
        print("All done!")
        return

    # Init databases
    print("Loading embedding model...")
    embed_model = SentenceTransformer('all-mpnet-base-v2')

    print("Connecting to databases...")
    client = chromadb.PersistentClient(path=CHROMADB_PATH)
    coll = client.get_or_create_collection("polymath_corpus", metadata={"hnsw:space": "cosine"})
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    print(f"ChromaDB: {coll.count()} chunks\n")

    # Load Marker models once
    print("Loading Marker OCR models (GPU)...")
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    models = create_model_dict()
    converter = PdfConverter(artifact_dict=models)
    print("Models ready\n")

    stats = {"fitz": 0, "marker": 0, "failed": 0, "chunks": 0}

    for i, pdf in enumerate(remaining, 1):
        name = pdf.name[:50] + "..." if len(pdf.name) > 53 else pdf.name
        print(f"[{i}/{len(remaining)}] {name}", flush=True)

        # Try fast extraction first
        text = extract_with_fitz(str(pdf))
        method = "fitz"

        if not text:
            try:
                print("  → Running Marker OCR...", flush=True)
                text = extract_with_marker(str(pdf), models, converter)
                method = "marker"
            except Exception as e:
                print(f"  ERROR: {e}")
                stats["failed"] += 1
                processed.add(pdf.name)
                with open(PROGRESS_FILE, 'w') as f:
                    json.dump({"processed": list(processed)}, f)
                continue

        if not text or len(text.strip()) < 200:
            print("  SKIP (no text)")
            stats["failed"] += 1
            processed.add(pdf.name)
            continue

        stats[method] += 1
        title = extract_title(text, pdf.name)
        chunks = chunk_text(text)
        concepts = extract_concepts(text)
        file_hash = hashlib.md5(pdf.name.encode()).hexdigest()[:10]

        print(f"  Title: {title[:60]}...")
        print(f"  Chars: {len(text)}, Chunks: {len(chunks)}, Concepts: {concepts[:5]}")

        # Add to ChromaDB
        ids = [f"{file_hash}_{j}" for j in range(len(chunks))]
        try:
            existing = coll.get(ids=ids[:1])
            if not existing['ids']:
                embeddings = embed_model.encode(chunks).tolist()
                coll.add(
                    ids=ids,
                    documents=chunks,
                    embeddings=embeddings,
                    metadatas=[{"source": pdf.name, "title": title[:200], "method": method} for _ in chunks]
                )
                stats["chunks"] += len(chunks)
                print(f"  → Added {len(chunks)} chunks")
        except Exception as e:
            print(f"  DB error: {e}")

        # Add to Neo4j
        try:
            with driver.session() as session:
                session.run("MERGE (p:Paper {title: $t}) SET p.source='ocr_ingest'", {"t": title[:200]})
                for c in concepts:
                    session.run("""
                        MATCH (p:Paper {title: $t})
                        MERGE (c:CONCEPT {name: $c})
                        MERGE (p)-[:MENTIONS]->(c)
                    """, {"t": title[:200], "c": c})
        except Exception as e:
            print(f"  Neo4j error: {e}")

        processed.add(pdf.name)
        print(f"  ✓ Done [{method}]")

        # Save progress and clear GPU memory
        with open(PROGRESS_FILE, 'w') as f:
            json.dump({"processed": list(processed)}, f)
        gc.collect()
        torch.cuda.empty_cache()

    driver.close()

    print(f"\n{'='*55}")
    print(f"COMPLETE:")
    print(f"  Text (fitz): {stats['fitz']}")
    print(f"  OCR (marker): {stats['marker']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Chunks added: {stats['chunks']}")
    print(f"  ChromaDB total: {coll.count()}")
    print("="*55)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 ingest_ocr_sequential.py /path/to/pdfs/")
        sys.exit(1)
    main(sys.argv[1])
