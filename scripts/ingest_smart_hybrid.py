#!/usr/bin/env python3
"""
Smart Hybrid Ingestion - Fast for text, Marker for scans
Optimized for RTX 5090
"""

import os
import sys
import json
import hashlib
import gc
import torch
from datetime import datetime
from pathlib import Path

os.environ["MARKER_BATCH_MULTIPLIER"] = "4"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

CHROMADB_PATH = "/home/user/work/polymax/chromadb/polymath_v2"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "polymathic2026"
PROGRESS_FILE = "/home/user/work/polymax/ingest_hybrid_progress.json"


def extract_with_fitz(pdf_path, min_chars=500):
    """Fast text extraction. Returns (text, success)"""
    import fitz
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        if len(text.strip()) > min_chars:
            return text, True
        return None, False
    except:
        return None, False


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
    "optimization": "optimization", "computation": "computation_theory",
    "linear algebra": "linear_algebra", "matrix": "matrix_theory",
    "probability": "probability_theory", "markov": "markov_chain",
}


def extract_concepts(text):
    text_lower = text.lower()
    return list(set(c for k, c in CONCEPTS.items() if k in text_lower))


def process_pdf(pdf, method, text, embed_model, coll, driver, stats):
    """Process a single PDF after text extraction."""
    title = extract_title(text, pdf.name)
    chunks = chunk_text(text)
    concepts = extract_concepts(text)
    file_hash = hashlib.md5(pdf.name.encode()).hexdigest()[:10]

    print(f"  Title: {title[:50]}...")
    print(f"  Chars: {len(text):,}, Chunks: {len(chunks)}")

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
        print(f"  DB error: {e}")

    # Add to Neo4j
    try:
        with driver.session() as session:
            session.run("MERGE (p:Paper {title: $t}) SET p.source='hybrid_ingest'", {"t": title[:200]})
            for c in concepts[:10]:
                session.run("""
                    MATCH (p:Paper {title: $t})
                    MERGE (c:CONCEPT {name: $c})
                    MERGE (p)-[:MENTIONS]->(c)
                """, {"t": title[:200], "c": c})
    except Exception as e:
        print(f"  Neo4j error: {e}")


def main(pdf_dir):
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

    print(f"{'='*60}")
    print(f"SMART HYBRID INGESTION")
    print(f"{'='*60}")
    print(f"[{datetime.now():%H:%M}] {len(remaining)}/{len(pdfs)} PDFs to process\n")

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

    # PHASE 1: Fast extraction for text-based PDFs
    print("=" * 60)
    print("PHASE 1: Fast Text Extraction (fitz)")
    print("=" * 60)

    stats = {"fitz": 0, "marker": 0, "failed": 0, "chunks": 0}
    needs_ocr = []
    start_time = datetime.now()

    for i, pdf in enumerate(remaining, 1):
        name = pdf.name[:40] + "..." if len(pdf.name) > 43 else pdf.name
        text, success = extract_with_fitz(str(pdf))

        if success:
            print(f"[{i}/{len(remaining)}] ✓ {name} (text)")
            stats["fitz"] += 1
            process_pdf(pdf, "fitz", text, embed_model, coll, driver, stats)
            processed.add(pdf.name)
        else:
            print(f"[{i}/{len(remaining)}] → {name} (needs OCR)")
            needs_ocr.append(pdf)

        # Save progress periodically
        if i % 20 == 0:
            with open(PROGRESS_FILE, 'w') as f:
                json.dump({"processed": list(processed)}, f)

    # Save after phase 1
    with open(PROGRESS_FILE, 'w') as f:
        json.dump({"processed": list(processed)}, f)

    phase1_time = (datetime.now() - start_time).total_seconds()
    print(f"\nPhase 1 complete: {stats['fitz']} PDFs in {phase1_time:.1f}s")
    print(f"Remaining for OCR: {len(needs_ocr)}\n")

    # PHASE 2: OCR for scanned PDFs (if any)
    if needs_ocr:
        print("=" * 60)
        print("PHASE 2: Marker OCR (GPU TURBO)")
        print("=" * 60)

        print("Loading Marker models...")
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered

        models = create_model_dict()
        converter = PdfConverter(artifact_dict=models)
        print("Models ready\n")

        for i, pdf in enumerate(needs_ocr, 1):
            name = pdf.name[:40] + "..." if len(pdf.name) > 43 else pdf.name
            print(f"[{i}/{len(needs_ocr)}] OCR: {name}", flush=True)

            try:
                rendered = converter(str(pdf))
                text, _, _ = text_from_rendered(rendered)

                if text and len(text.strip()) > 200:
                    stats["marker"] += 1
                    process_pdf(pdf, "marker", text, embed_model, coll, driver, stats)
                else:
                    print("  SKIP (no text)")
                    stats["failed"] += 1
            except Exception as e:
                print(f"  ERROR: {e}")
                stats["failed"] += 1

            processed.add(pdf.name)
            with open(PROGRESS_FILE, 'w') as f:
                json.dump({"processed": list(processed)}, f)

            gc.collect()
            torch.cuda.empty_cache()

    driver.close()

    total_time = (datetime.now() - start_time).total_seconds()
    print(f"\n{'='*60}")
    print(f"COMPLETE")
    print(f"{'='*60}")
    print(f"  Fast (fitz): {stats['fitz']}")
    print(f"  OCR (marker): {stats['marker']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Chunks added: {stats['chunks']}")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  ChromaDB total: {coll.count()}")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 ingest_smart_hybrid.py /path/to/pdfs/")
        sys.exit(1)
    main(sys.argv[1])
