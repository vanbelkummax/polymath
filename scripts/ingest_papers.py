#!/usr/bin/env python3
"""
Polymath Paper Ingestion Script
Adds PDFs to ChromaDB (vectors) and Neo4j (knowledge graph)

Usage:
    python3 ingest_papers.py /path/to/pdfs/
    python3 ingest_papers.py /path/to/single.pdf
"""

import os
import sys
import re
import hashlib
from pathlib import Path
from tqdm import tqdm

# Fast paths only!
CHROMADB_PATH = "/home/user/work/polymax/chromadb/polymath_v2"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "polymathic2026"

# Concept extraction keywords
CONCEPT_KEYWORDS = {
    # Methods
    "deep learning": "deep_learning",
    "neural network": "neural_network",
    "transformer": "transformer",
    "attention": "attention",
    "machine learning": "machine_learning",
    "reinforcement learning": "reinforcement_learning",
    "bayesian": "bayesian",
    "regression": "regression",
    "classification": "classification",
    "clustering": "clustering",

    # Biology
    "spatial transcriptomics": "spatial_transcriptomics",
    "single-cell": "single_cell",
    "single cell": "single_cell",
    "gene expression": "gene_expression",
    "pathology": "pathology",
    "histology": "histology",
    "cancer": "cancer",
    "tumor": "tumor",
    "cell type": "cell_type",

    # Cross-domain (from our seeded concepts)
    "information theory": "information_theory",
    "entropy": "entropy",
    "compressed sensing": "compressed_sensing",
    "sparse": "sparse_coding",
    "game theory": "game_theory",
    "evolution": "evolution",
    "natural selection": "natural_selection",
    "feedback": "feedback_control",
    "cybernetics": "cybernetics",
    "systems theory": "systems_theory",
    "complexity": "complexity",
    "emergence": "emergence",
    "self-organization": "self_organization",
    "autopoiesis": "autopoiesis",
    "thermodynamics": "thermodynamics",
    "free energy": "free_energy_principle",
    "causal": "causal_inference",
    "counterfactual": "counterfactual_reasoning",
    "analogy": "structure_mapping",
    "pattern formation": "pattern_formation",
    "reaction-diffusion": "reaction_diffusion",
    "turing pattern": "reaction_diffusion",
    "segregation": "segregation",
    "schelling": "schelling_model",
    "red queen": "red_queen",
    "co-evolution": "coevolution",
}


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"  Warning: Could not extract text from {pdf_path}: {e}")
        return ""


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list:
    """Split text into overlapping chunks."""
    if not text:
        return []

    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk) > 100:  # Skip tiny chunks
            chunks.append(chunk)

    return chunks


def extract_title_from_text(text: str, filename: str) -> str:
    """Try to extract title from first lines of PDF text."""
    lines = text.split('\n')[:10]
    for line in lines:
        line = line.strip()
        if len(line) > 20 and len(line) < 200:
            # Likely a title
            return line
    # Fallback to filename
    return Path(filename).stem.replace('_', ' ').replace('-', ' ')


def extract_concepts(text: str) -> list:
    """Extract concepts from text based on keywords."""
    text_lower = text.lower()
    concepts = set()

    for keyword, concept in CONCEPT_KEYWORDS.items():
        if keyword in text_lower:
            concepts.add(concept)

    return list(concepts)


def get_file_hash(filepath: str) -> str:
    """Get MD5 hash of file for deduplication."""
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def add_to_chromadb(chunks: list, metadata: dict, collection):
    """Add chunks to ChromaDB collection."""
    if not chunks:
        return 0

    ids = [f"{metadata['hash']}_{i}" for i in range(len(chunks))]
    metadatas = [metadata.copy() for _ in chunks]

    # Check for existing
    existing = collection.get(ids=ids[:1])
    if existing['ids']:
        return 0  # Already ingested

    collection.add(
        ids=ids,
        documents=chunks,
        metadatas=metadatas
    )
    return len(chunks)


def add_to_neo4j(title: str, concepts: list, driver):
    """Add paper and concept relationships to Neo4j."""
    if not concepts:
        return

    with driver.session() as session:
        # Create paper node
        session.run("""
            MERGE (p:Paper {title: $title})
            ON CREATE SET p.created = datetime(), p.source = 'ingest_script'
        """, {"title": title})

        # Link to concepts
        for concept in concepts:
            session.run("""
                MATCH (p:Paper {title: $title})
                MERGE (c:CONCEPT {name: $concept})
                MERGE (p)-[:MENTIONS]->(c)
            """, {"title": title, "concept": concept})


def ingest_pdfs(input_path: str):
    """Main ingestion function."""
    print("=" * 60)
    print("POLYMATH PAPER INGESTION")
    print("=" * 60)

    # Validate path is on Linux filesystem
    if input_path.startswith('/mnt/'):
        print("ERROR: Input path is on Windows filesystem (slow)!")
        print("Copy files to /home/user/work/ first.")
        sys.exit(1)

    # Get list of PDFs
    input_path = Path(input_path)
    if input_path.is_file():
        pdfs = [input_path]
    else:
        pdfs = list(input_path.glob("*.pdf"))

    print(f"Found {len(pdfs)} PDFs to process")

    # Initialize connections
    print("\nConnecting to databases...")
    import chromadb
    from sentence_transformers import SentenceTransformer
    from neo4j import GraphDatabase

    model = SentenceTransformer('all-mpnet-base-v2')
    client = chromadb.PersistentClient(path=CHROMADB_PATH)
    collection = client.get_or_create_collection(
        name="polymath_corpus",
        metadata={"hnsw:space": "cosine"}
    )
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    print(f"ChromaDB: {collection.count()} existing chunks")

    # Process each PDF
    total_chunks = 0
    total_concepts = 0

    for pdf in tqdm(pdfs, desc="Ingesting"):
        try:
            # Extract text
            text = extract_text_from_pdf(str(pdf))
            if not text:
                continue

            # Get metadata
            file_hash = get_file_hash(str(pdf))
            title = extract_title_from_text(text, pdf.name)
            concepts = extract_concepts(text)

            # Chunk and embed
            chunks = chunk_text(text)

            metadata = {
                "source": pdf.name,
                "title": title,
                "hash": file_hash,
                "concepts": ",".join(concepts[:5])
            }

            # Add to ChromaDB
            added = add_to_chromadb(chunks, metadata, collection)
            total_chunks += added

            # Add to Neo4j
            add_to_neo4j(title, concepts, driver)
            total_concepts += len(concepts)

        except Exception as e:
            print(f"\nError processing {pdf.name}: {e}")

    driver.close()

    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print(f"  Chunks added: {total_chunks}")
    print(f"  Concept links: {total_concepts}")
    print(f"  ChromaDB total: {collection.count()}")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 ingest_papers.py /path/to/pdfs/")
        print("\nNOTE: Path must be on Linux filesystem (/home/user/...)")
        print("      NOT on Windows (/mnt/...)")
        sys.exit(1)

    ingest_pdfs(sys.argv[1])
