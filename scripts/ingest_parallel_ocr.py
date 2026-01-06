#!/usr/bin/env python3
"""
Fast parallel OCR ingestion using Surya
Processes multiple pages in batch on GPU for maximum speed
"""

import os
import sys
import json
import hashlib
import gc
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import fitz  # PyMuPDF

# FAST PATHS ONLY
CHROMADB_PATH = "/home/user/work/polymax/chromadb/polymath_v2"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "polymathic2026"
PROGRESS_FILE = "/home/user/work/polymax/ingest_parallel_progress.json"

# Batch size for GPU processing
BATCH_SIZE = 8  # Process 8 pages at once on GPU
MAX_PAGES = None  # None = all pages, or set limit like 50


def pdf_to_images(pdf_path, dpi=150, max_pages=None):
    """Convert PDF pages to PIL images for OCR."""
    from PIL import Image
    import io

    doc = fitz.open(pdf_path)
    images = []

    n_pages = len(doc) if max_pages is None else min(len(doc), max_pages)

    for i in range(n_pages):
        page = doc[i]
        pix = page.get_pixmap(dpi=dpi)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        images.append(img)

    doc.close()
    return images


def ocr_batch_surya(images, det_predictor, rec_predictor):
    """OCR a batch of images using Surya."""
    # Detect text regions
    det_results = det_predictor.batch_detection(images)

    # Recognize text - process each image
    texts = []
    for img, det_result in zip(images, det_results):
        try:
            # Get text from detected bboxes
            text_result = rec_predictor.get_bboxes_text([img], [det_result])
            if text_result and len(text_result) > 0:
                # Extract text lines
                page_lines = []
                for bbox_result in text_result[0]:
                    if hasattr(bbox_result, 'text'):
                        page_lines.append(bbox_result.text)
                texts.append("\n".join(page_lines))
            else:
                texts.append("")
        except Exception as e:
            print(f"    OCR error: {e}")
            texts.append("")

    return texts


def extract_with_fitz(pdf_path):
    """Try fast text extraction first."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text if len(text.strip()) > 500 else None
    except:
        return None


def chunk_text(text, size=1500, overlap=200, max_chunks=30):
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
    for line in text.split('\n')[:20]:
        line = line.strip()
        if 15 < len(line) < 250 and not line.startswith('http'):
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
}


def extract_concepts(text):
    text_lower = text.lower()
    return list(set(c for k, c in CONCEPTS.items() if k in text_lower))


def process_pdf_with_ocr(pdf_path, det_predictor, rec_predictor, max_pages=None):
    """Process a single PDF with OCR."""
    print(f"  Converting to images...", end=" ", flush=True)
    images = pdf_to_images(str(pdf_path), dpi=150, max_pages=max_pages)
    print(f"{len(images)} pages", flush=True)

    all_text = []

    # Process in batches
    for i in range(0, len(images), BATCH_SIZE):
        batch = images[i:i+BATCH_SIZE]
        print(f"  OCR batch {i//BATCH_SIZE + 1}/{(len(images)-1)//BATCH_SIZE + 1}...", end=" ", flush=True)
        texts = ocr_batch_surya(batch, det_predictor, rec_predictor)
        all_text.extend(texts)
        print("done", flush=True)

    return "\n\n".join(all_text)


def main(pdf_dir, max_pages=None):
    if pdf_dir.startswith('/mnt/'):
        print("ERROR: Copy files to Linux first!")
        sys.exit(1)

    import chromadb
    from sentence_transformers import SentenceTransformer
    from neo4j import GraphDatabase
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor
    from surya.foundation import FoundationPredictor

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

    # Initialize models
    print("Loading Surya OCR models (GPU)...")
    det_predictor = DetectionPredictor(device="cuda")
    foundation = FoundationPredictor(device="cuda")
    rec_predictor = RecognitionPredictor(foundation)

    print("Loading embedding model...")
    embed_model = SentenceTransformer('all-mpnet-base-v2')

    print("Connecting to databases...")
    client = chromadb.PersistentClient(path=CHROMADB_PATH)
    coll = client.get_or_create_collection("polymath_corpus", metadata={"hnsw:space": "cosine"})
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    print(f"ChromaDB: {coll.count()} chunks\n")

    stats = {"fitz": 0, "ocr": 0, "failed": 0, "chunks": 0}

    for i, pdf in enumerate(remaining, 1):
        name = pdf.name[:50] + "..." if len(pdf.name) > 53 else pdf.name
        print(f"[{i}/{len(remaining)}] {name}")

        # Try fast extraction first
        text = extract_with_fitz(str(pdf))
        method = "fitz"

        if not text:
            # Use OCR
            try:
                text = process_pdf_with_ocr(pdf, det_predictor, rec_predictor, max_pages)
                method = "ocr"
            except Exception as e:
                print(f"  ERROR: {e}")
                stats["failed"] += 1
                processed.add(pdf.name)
                continue

        if not text or len(text.strip()) < 200:
            print(f"  SKIP (no text)")
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
        print(f"  OK [{method}] {len(chunks)} chunks, {len(concepts)} concepts")

        # Save progress
        with open(PROGRESS_FILE, 'w') as f:
            json.dump({"processed": list(processed)}, f)
        gc.collect()

    driver.close()

    print(f"\n{'='*55}")
    print(f"DONE:")
    print(f"  Fast (fitz): {stats['fitz']}")
    print(f"  OCR (surya): {stats['ocr']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Chunks: {stats['chunks']}")
    print(f"  Total: {coll.count()}")
    print("="*55)


if __name__ == "__main__":
    max_pages = None
    if len(sys.argv) >= 3:
        max_pages = int(sys.argv[2])

    if len(sys.argv) < 2:
        print("Usage: python3 ingest_parallel_ocr.py /path/to/pdfs/ [max_pages]")
        print("\nExample: python3 ingest_parallel_ocr.py ./staging/ 50")
        sys.exit(1)

    main(sys.argv[1], max_pages)
