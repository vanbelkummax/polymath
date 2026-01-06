#!/usr/bin/env python3
"""
MinerU (Magic-PDF) based ingestion - SOTA for textbooks
Best for complex layouts, tables, equations
"""

import os
import sys
import json
import hashlib
import gc
import tempfile
import shutil
from datetime import datetime
from pathlib import Path

# FAST PATHS ONLY
CHROMADB_PATH = "/home/user/work/polymax/chromadb/polymath_v2"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "polymathic2026"
PROGRESS_FILE = "/home/user/work/polymax/ingest_mineru_progress.json"


def extract_with_fitz(pdf_path):
    """Try fast text extraction first."""
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


def extract_with_mineru(pdf_path, output_dir):
    """Extract with MinerU - handles complex layouts."""
    from magic_pdf.pipe.UNIPipe import UNIPipe
    from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter

    # Read PDF bytes
    with open(pdf_path, 'rb') as f:
        pdf_bytes = f.read()

    # Setup output
    image_writer = DiskReaderWriter(output_dir)

    # Process with auto mode (detects if OCR needed)
    pipe = UNIPipe(pdf_bytes, {"_pdf_type": "", "model_list": []}, image_writer)
    pipe.pipe_classify()
    pipe.pipe_analyze()
    pipe.pipe_parse()

    # Get markdown output
    md_content = pipe.pipe_mk_markdown(output_dir, drop_mode="none")

    return md_content


def chunk_text(text, size=1500, overlap=200, max_chunks=40):
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
    """Get title from text or filename."""
    for line in text.split('\n')[:25]:
        line = line.strip().strip('#').strip()
        if 15 < len(line) < 250 and not line.startswith('http') and not line.startswith('!'):
            return line
    return Path(filename).stem.replace('_', ' ')


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
    "automata": "cellular_automata", "computation": "computation",
    "regression": "regression", "classification": "classification",
}


def extract_concepts(text):
    text_lower = text.lower()
    return list(set(c for k, c in CONCEPTS.items() if k in text_lower))


def main(pdf_dir):
    if pdf_dir.startswith('/mnt/'):
        print("ERROR: Copy files to Linux first!")
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

    print("Loading embedding model...")
    embed_model = SentenceTransformer('all-mpnet-base-v2')

    print("Connecting to databases...")
    client = chromadb.PersistentClient(path=CHROMADB_PATH)
    coll = client.get_or_create_collection("polymath_corpus", metadata={"hnsw:space": "cosine"})
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    print(f"ChromaDB: {coll.count()} chunks\n")

    stats = {"fitz": 0, "mineru": 0, "failed": 0, "chunks": 0}

    for i, pdf in enumerate(remaining, 1):
        name = pdf.name[:50] + "..." if len(pdf.name) > 53 else pdf.name
        print(f"[{i}/{len(remaining)}] {name}", end=" ", flush=True)

        # Try fast extraction first
        text = extract_with_fitz(str(pdf))
        method = "fitz"

        if not text:
            # Use MinerU
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    print("(MinerU)", end=" ", flush=True)
                    text = extract_with_mineru(str(pdf), tmpdir)
                    method = "mineru"
            except Exception as e:
                print(f"ERROR: {e}")
                stats["failed"] += 1
                processed.add(pdf.name)
                continue

        if not text or len(text.strip()) < 200:
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
                embeddings = embed_model.encode(chunks).tolist()
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
                session.run("MERGE (p:Paper {title: $t}) SET p.source='mineru_ingest'", {"t": title[:200]})
                for c in concepts:
                    session.run("""
                        MATCH (p:Paper {title: $t})
                        MERGE (c:CONCEPT {name: $c})
                        MERGE (p)-[:MENTIONS]->(c)
                    """, {"t": title[:200], "c": c})
        except Exception as e:
            print(f"Neo4j error: {e}")

        processed.add(pdf.name)
        print(f"OK [{method}] {len(chunks)} chunks, {len(concepts)} concepts")

        # Save progress
        with open(PROGRESS_FILE, 'w') as f:
            json.dump({"processed": list(processed)}, f)
        gc.collect()

    driver.close()

    print(f"\n{'='*55}")
    print(f"DONE:")
    print(f"  Fast (fitz): {stats['fitz']}")
    print(f"  OCR (mineru): {stats['mineru']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Chunks: {stats['chunks']}")
    print(f"  Total: {coll.count()}")
    print("="*55)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 ingest_mineru.py /path/to/pdfs/")
        sys.exit(1)
    main(sys.argv[1])
