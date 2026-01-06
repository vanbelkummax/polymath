#!/usr/bin/env python3
"""
Enhanced PDF ingestion with Marker OCR fallback
Uses GPU for fast extraction of scanned/image PDFs

Usage:
    python3 ingest_with_ocr.py /path/to/pdfs/
"""

import os
import sys
import json
import hashlib
import gc
from datetime import datetime
from pathlib import Path

# FAST PATHS ONLY
CHROMADB_PATH = "/home/user/work/polymax/chromadb/polymath_v2"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "polymathic2026"
PROGRESS_FILE = "/home/user/work/polymax/ingest_ocr_progress.json"

def extract_with_fitz(pdf_path, max_pages=50):
    """Fast extraction with PyMuPDF (for text PDFs)."""
    import fitz
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for i in range(min(len(doc), max_pages)):
            text += doc[i].get_text()
            if len(text) > 100000:
                break
        doc.close()
        return text if len(text.strip()) > 200 else None
    except:
        return None


def extract_with_marker(pdf_path):
    """GPU-accelerated OCR with Marker (for scanned PDFs)."""
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered

        # Load models (cached after first call)
        models = create_model_dict()
        converter = PdfConverter(artifact_dict=models)

        # Convert PDF to markdown
        rendered = converter(str(pdf_path))
        text, _, _ = text_from_rendered(rendered)

        return text if len(text.strip()) > 200 else None
    except Exception as e:
        print(f"Marker error: {e}")
        return None


def extract_pdf_text(pdf_path, use_ocr=True):
    """Extract text, falling back to OCR if needed."""
    # Try fast extraction first
    text = extract_with_fitz(pdf_path)

    if text:
        return text, "fitz"

    # Fall back to Marker OCR
    if use_ocr:
        text = extract_with_marker(pdf_path)
        if text:
            return text, "marker"

    return None, None


def chunk_text(text, size=1500, overlap=200, max_chunks=25):
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
    for line in text.split('\n')[:15]:
        line = line.strip()
        # Skip short lines, URLs, numbers only
        if 20 < len(line) < 250 and not line.startswith('http') and not line.isdigit():
            return line
    return Path(filename).stem.replace('_', ' ').replace('-', ' ')


# Concept keywords
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
    "self-organization": "self_organization", "emergence": "emergence",
    "bayesian": "bayesian_inference", "quantum": "quantum",
    "nonlinear": "nonlinear_dynamics", "chaos": "chaos_theory",
    "pattern": "pattern_formation", "statistical": "statistical_learning",
    "optimization": "optimization", "stochastic": "stochastic_process",
}


def extract_concepts(text):
    """Extract matching concepts."""
    text_lower = text.lower()
    return list(set(c for k, c in CONCEPTS.items() if k in text_lower))


def main(pdf_dir):
    if pdf_dir.startswith('/mnt/'):
        print("ERROR: Copy files to Linux first!")
        print("  mkdir -p /home/user/work/polymax/ingest_staging")
        print("  cp '/mnt/c/...'/*.pdf /home/user/work/polymax/ingest_staging/")
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

    if not remaining:
        print("All PDFs already processed!")
        return

    # Init
    print("Loading embedding model...")
    model = SentenceTransformer('all-mpnet-base-v2')

    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMADB_PATH)
    coll = client.get_or_create_collection("polymath_corpus", metadata={"hnsw:space": "cosine"})

    print("Connecting to Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    print(f"ChromaDB: {coll.count()} chunks\n")

    # Stats
    stats = {"fitz": 0, "marker": 0, "failed": 0, "chunks": 0}

    for i, pdf in enumerate(remaining, 1):
        name_short = pdf.name[:45] + "..." if len(pdf.name) > 48 else pdf.name
        print(f"[{i}/{len(remaining)}] {name_short}", end=" ", flush=True)

        text, method = extract_pdf_text(str(pdf))

        if not text:
            print("SKIP (no text)")
            stats["failed"] += 1
            processed.add(pdf.name)
            continue

        stats[method] += 1
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
                    metadatas=[{"source": pdf.name, "title": title[:200], "method": method} for _ in chunks]
                )
                stats["chunks"] += len(chunks)
        except Exception as e:
            print(f"DB error: {e}")

        # Add to Neo4j
        try:
            with driver.session() as session:
                session.run("MERGE (p:Paper {title: $t}) SET p.source='ingest_ocr'", {"t": title[:200]})
                for c in concepts:
                    session.run("""
                        MATCH (p:Paper {title: $t})
                        MERGE (c:CONCEPT {name: $c})
                        MERGE (p)-[:MENTIONS]->(c)
                    """, {"t": title[:200], "c": c})
        except Exception as e:
            print(f"Neo4j error: {e}")

        processed.add(pdf.name)
        print(f"OK [{method}] ({len(chunks)} chunks, {len(concepts)} concepts)")

        # Save progress every 5
        if i % 5 == 0:
            with open(PROGRESS_FILE, 'w') as f:
                json.dump({"processed": list(processed)}, f)
            gc.collect()

    driver.close()

    # Final save
    with open(PROGRESS_FILE, 'w') as f:
        json.dump({"processed": list(processed)}, f)

    print(f"\n{'='*55}")
    print(f"DONE:")
    print(f"  Fast (fitz): {stats['fitz']} papers")
    print(f"  OCR (marker): {stats['marker']} papers")
    print(f"  Failed: {stats['failed']} papers")
    print(f"  Chunks added: {stats['chunks']}")
    print(f"  ChromaDB total: {coll.count()}")
    print("="*55)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 ingest_with_ocr.py /path/to/pdfs/")
        print("\nThis script uses Marker (GPU) for OCR on scanned PDFs.")
        sys.exit(1)
    main(sys.argv[1])
