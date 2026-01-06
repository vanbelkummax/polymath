#!/usr/bin/env python3
"""
Smart Multi-Tool PDF Ingestion
- Fast: PyMuPDF (text PDFs)
- Batch: Marker with --workers (most scanned PDFs)
- Complex: MinerU (wild layouts)
- Math: Nougat (equations, LaTeX)

Usage:
    python3 ingest_smart.py /path/to/pdfs/
    python3 ingest_smart.py /path/to/pdfs/ --workers 8
"""

import os
import sys
import json
import hashlib
import subprocess
import tempfile
import shutil
from datetime import datetime
from pathlib import Path

CHROMADB_PATH = "/home/user/work/polymax/chromadb/polymath_v2"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "polymathic2026"
PROGRESS_FILE = "/home/user/work/polymax/ingest_smart_progress.json"


def has_text(pdf_path, min_chars=500):
    """Check if PDF has extractable text."""
    import fitz
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for i, page in enumerate(doc):
            text += page.get_text()
            if len(text) > min_chars:
                doc.close()
                return True
            if i > 5:  # Check first 5 pages
                break
        doc.close()
        return len(text.strip()) > min_chars
    except:
        return False


def extract_fitz(pdf_path):
    """Fast extraction with PyMuPDF."""
    import fitz
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    doc.close()
    return text


def extract_marker_batch(pdf_dir, output_dir, workers=8):
    """
    Use Marker CLI for batch processing.
    marker IN_FOLDER --output_dir OUT_FOLDER --workers N
    """
    cmd = [
        "marker", str(pdf_dir),
        "--output_dir", str(output_dir),
        "--workers", str(workers),
        "--output_format", "markdown",
        "--disable_tqdm"
    ]

    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

    if result.returncode != 0:
        print(f"Marker error: {result.stderr[:500]}")
        return {}

    # Read all markdown outputs (marker creates subfolders per PDF)
    outputs = {}
    for md_file in Path(output_dir).rglob("*.md"):
        # Extract PDF name from path or filename
        pdf_name = md_file.parent.name + ".pdf" if md_file.name == "output.md" else md_file.stem + ".pdf"
        outputs[pdf_name] = md_file.read_text()

    return outputs


def classify_pdf(pdf_path):
    """Classify PDF to pick best extraction method."""
    import fitz

    doc = fitz.open(pdf_path)
    n_pages = len(doc)

    # Check for equations (look for math-like patterns in first pages)
    sample_text = ""
    for i in range(min(5, n_pages)):
        sample_text += doc[i].get_text()
    doc.close()

    has_equations = any(x in sample_text for x in ['∫', '∑', '∂', '∞', 'equation', 'theorem', 'proof'])
    has_text_content = len(sample_text.strip()) > 500

    if has_text_content:
        return "fitz"  # Fast path
    elif has_equations:
        return "nougat"  # Math specialist
    else:
        return "marker"  # General OCR


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
    for line in text.split('\n')[:25]:
        line = line.strip().strip('#').strip()
        if 15 < len(line) < 250 and not line.startswith(('http', '!', '|', '-')):
            return line
    return Path(filename).stem.replace('_', ' ')


CONCEPTS = {
    "deep learning": "deep_learning", "neural network": "neural_network",
    "machine learning": "machine_learning", "transformer": "transformer",
    "information theory": "information_theory", "entropy": "entropy",
    "compressed sensing": "compressed_sensing", "game theory": "game_theory",
    "evolution": "evolution", "natural selection": "natural_selection",
    "cybernetics": "cybernetics", "complexity": "complexity",
    "thermodynamics": "thermodynamics", "causal": "causal_inference",
    "self-organization": "self_organization", "emergence": "emergence",
    "bayesian": "bayesian_inference", "quantum": "quantum",
    "nonlinear": "nonlinear_dynamics", "chaos": "chaos_theory",
    "statistical": "statistical_learning", "stochastic": "stochastic_process",
    "automata": "cellular_automata", "pattern": "pattern_formation",
    "regression": "regression", "optimization": "optimization",
}


def extract_concepts(text):
    text_lower = text.lower()
    return list(set(c for k, c in CONCEPTS.items() if k in text_lower))


def main(pdf_dir, workers=8):
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
    print(f"Workers: {workers}")

    if not remaining:
        print("All done!")
        return

    # Classify PDFs
    print("\nClassifying PDFs...")
    fitz_pdfs = []
    ocr_pdfs = []

    for pdf in remaining:
        method = classify_pdf(str(pdf))
        if method == "fitz":
            fitz_pdfs.append(pdf)
        else:
            ocr_pdfs.append(pdf)

    print(f"  Text PDFs (fast): {len(fitz_pdfs)}")
    print(f"  OCR needed: {len(ocr_pdfs)}")

    # Initialize
    print("\nLoading models...")
    embed_model = SentenceTransformer('all-mpnet-base-v2')
    client = chromadb.PersistentClient(path=CHROMADB_PATH)
    coll = client.get_or_create_collection("polymath_corpus", metadata={"hnsw:space": "cosine"})
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    print(f"ChromaDB: {coll.count()} chunks\n")

    stats = {"fitz": 0, "marker": 0, "failed": 0, "chunks": 0}

    # Process text PDFs first (fast)
    if fitz_pdfs:
        print(f"=== Processing {len(fitz_pdfs)} text PDFs (fast) ===")
        for i, pdf in enumerate(fitz_pdfs, 1):
            name = pdf.name[:45] + "..." if len(pdf.name) > 48 else pdf.name
            print(f"[{i}/{len(fitz_pdfs)}] {name}", end=" ", flush=True)

            try:
                text = extract_fitz(str(pdf))
                if len(text.strip()) < 300:
                    ocr_pdfs.append(pdf)  # Move to OCR queue
                    print("-> OCR queue")
                    continue

                title = extract_title(text, pdf.name)
                chunks = chunk_text(text)
                concepts = extract_concepts(text)
                file_hash = hashlib.md5(pdf.name.encode()).hexdigest()[:10]

                ids = [f"{file_hash}_{j}" for j in range(len(chunks))]
                existing = coll.get(ids=ids[:1])
                if not existing['ids']:
                    embeddings = embed_model.encode(chunks).tolist()
                    coll.add(ids=ids, documents=chunks, embeddings=embeddings,
                            metadatas=[{"source": pdf.name, "title": title[:200], "method": "fitz"} for _ in chunks])
                    stats["chunks"] += len(chunks)

                with driver.session() as session:
                    session.run("MERGE (p:Paper {title: $t})", {"t": title[:200]})
                    for c in concepts:
                        session.run("MATCH (p:Paper {title: $t}) MERGE (c:CONCEPT {name: $c}) MERGE (p)-[:MENTIONS]->(c)",
                                   {"t": title[:200], "c": c})

                stats["fitz"] += 1
                processed.add(pdf.name)
                print(f"OK ({len(chunks)} chunks)")

            except Exception as e:
                print(f"ERROR: {e}")
                stats["failed"] += 1

            # Save progress
            with open(PROGRESS_FILE, 'w') as f:
                json.dump({"processed": list(processed)}, f)

    # Process OCR PDFs with Marker batch
    if ocr_pdfs:
        print(f"\n=== Processing {len(ocr_pdfs)} scanned PDFs with Marker (workers={workers}) ===")

        with tempfile.TemporaryDirectory(dir="/home/user/work/polymax") as tmpdir:
            # Create input dir with symlinks
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()
            output_dir.mkdir()

            for pdf in ocr_pdfs:
                (input_dir / pdf.name).symlink_to(pdf)

            print(f"Running Marker on {len(ocr_pdfs)} PDFs...")
            marker_outputs = extract_marker_batch(input_dir, output_dir, workers)
            print(f"Marker produced {len(marker_outputs)} outputs")

            for pdf in ocr_pdfs:
                text = marker_outputs.get(pdf.name, "")
                if not text or len(text.strip()) < 200:
                    stats["failed"] += 1
                    processed.add(pdf.name)
                    continue

                title = extract_title(text, pdf.name)
                chunks = chunk_text(text)
                concepts = extract_concepts(text)
                file_hash = hashlib.md5(pdf.name.encode()).hexdigest()[:10]

                ids = [f"{file_hash}_{j}" for j in range(len(chunks))]
                try:
                    existing = coll.get(ids=ids[:1])
                    if not existing['ids']:
                        embeddings = embed_model.encode(chunks).tolist()
                        coll.add(ids=ids, documents=chunks, embeddings=embeddings,
                                metadatas=[{"source": pdf.name, "title": title[:200], "method": "marker"} for _ in chunks])
                        stats["chunks"] += len(chunks)

                    with driver.session() as session:
                        session.run("MERGE (p:Paper {title: $t})", {"t": title[:200]})
                        for c in concepts:
                            session.run("MATCH (p:Paper {title: $t}) MERGE (c:CONCEPT {name: $c}) MERGE (p)-[:MENTIONS]->(c)",
                                       {"t": title[:200], "c": c})

                    stats["marker"] += 1
                    print(f"  {pdf.name[:40]}: {len(chunks)} chunks")
                except Exception as e:
                    print(f"  {pdf.name[:40]}: ERROR {e}")
                    stats["failed"] += 1

                processed.add(pdf.name)

            with open(PROGRESS_FILE, 'w') as f:
                json.dump({"processed": list(processed)}, f)

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
    workers = 8
    if "--workers" in sys.argv:
        idx = sys.argv.index("--workers")
        workers = int(sys.argv[idx + 1])
        sys.argv.pop(idx)
        sys.argv.pop(idx)

    if len(sys.argv) < 2:
        print("Usage: python3 ingest_smart.py /path/to/pdfs/ [--workers N]")
        sys.exit(1)

    main(sys.argv[1], workers)
