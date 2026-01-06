#!/usr/bin/env python3
"""
Full Paper Ingestion Pipeline
Adds papers to both:
1. ChromaDB (RAG - semantic search)
2. Neo4j (KG - relationship graph)
"""

import os
import sys
import json
import time
from pathlib import Path
from tqdm import tqdm
import chromadb
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF

# Configuration
CHROMADB_PATH = "/mnt/z/chromadb_polymax_full"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "polymathic2026"
CHUNK_SIZE = 1000  # chars per chunk
CHUNK_OVERLAP = 200

# Entity extraction keywords (expanded for classic papers)
METHODS = [
    # Foundation models
    'UNI', 'CLAM', 'TransMIL', 'Cell2location', 'Phikon', 'RCTD',
    'Prov-GigaPath', 'H-optimus', 'Virchow2', 'SegDecon', 'stTransfer',
    'STdeconvolve', 'SWOT', 'OmicsTweezer', 'Hist2ST', 'Img2ST',
    # Classic methods
    'neural network', 'perceptron', 'backpropagation', 'gradient descent',
    'reaction-diffusion', 'Turing pattern', 'morphogen', 'chemotaxis',
    'integral feedback', 'network motif', 'feed-forward loop',
    # Modern AI
    'transformer', 'attention mechanism', 'self-attention', 'BERT', 'GPT',
    'diffusion model', 'variational autoencoder', 'VAE', 'GAN',
    'graph neural network', 'GNN', 'message passing',
    # RAG/Knowledge
    'RAG', 'retrieval augmented generation', 'knowledge graph', 'GraphRAG',
    'vector database', 'embedding', 'semantic search',
]

CONCEPTS = [
    # Spatial/Genomics
    'spatial transcriptomics', 'spatial deconvolution', 'foundation model',
    'tumor microenvironment', 'scRNA-seq', 'H&E', 'attention mechanism',
    'cell type annotation', 'gene expression', 'single-cell',
    # Systems Biology
    'morphogenesis', 'pattern formation', 'developmental biology',
    'gene regulatory network', 'signaling pathway', 'feedback loop',
    'robustness', 'adaptation', 'homeostasis', 'bistability',
    # Information Theory
    'information theory', 'entropy', 'mutual information',
    'channel capacity', 'noise', 'signal transduction',
    # AI/ML
    'large language model', 'LLM', 'agentic AI', 'autonomous agent',
    'retrieval', 'reasoning', 'emergence', 'self-improvement',
    # Polymathic
    'cross-domain', 'interdisciplinary', 'systems thinking',
    'complexity', 'emergence', 'self-organization',
]

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using PyMuPDF"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"Error extracting {pdf_path}: {e}")
        return ""

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    return chunks

def extract_entities(text: str, title: str) -> dict:
    """Extract entities using keyword matching (fast, no API needed)"""
    text_lower = text.lower()
    title_lower = title.lower()

    found_methods = []
    for method in METHODS:
        if method.lower() in text_lower or method.lower() in title_lower:
            found_methods.append(method)

    found_concepts = []
    for concept in CONCEPTS:
        if concept.lower() in text_lower or concept.lower() in title_lower:
            found_concepts.append(concept)

    # Create relationships between methods and concepts found together
    relationships = []
    for method in found_methods:
        for concept in found_concepts[:3]:  # Limit to top 3 concepts
            relationships.append({
                "source": method,
                "target": concept,
                "type": "RELATES_TO",
                "confidence": 0.6
            })

    return {
        "entities": [{"name": m, "type": "method"} for m in found_methods] +
                   [{"name": c, "type": "concept"} for c in found_concepts],
        "relationships": relationships,
        "paper_title": title
    }

def add_to_neo4j(driver, extraction: dict, paper_id: str):
    """Add paper and entities to Neo4j"""
    with driver.session() as session:
        # Create paper node
        session.run("""
            MERGE (p:Paper {id: $id})
            SET p.title = $title,
                p.ingestion_time = timestamp()
        """, id=paper_id, title=extraction['paper_title'])

        # Create entity nodes and MENTIONS relationships
        for entity in extraction.get('entities', []):
            entity_type = entity['type'].upper()
            session.run(f"""
                MERGE (e:{entity_type} {{name: $name}})
                WITH e
                MATCH (p:Paper {{id: $paper_id}})
                MERGE (p)-[:MENTIONS]->(e)
            """, name=entity['name'], paper_id=paper_id)

        # Create inter-entity relationships
        for rel in extraction.get('relationships', []):
            session.run("""
                MATCH (s {name: $source})
                MATCH (t {name: $target})
                MERGE (s)-[r:RELATES_TO]->(t)
                SET r.confidence = $confidence,
                    r.paper = $paper_title
            """, source=rel['source'], target=rel['target'],
                 confidence=rel['confidence'], paper_title=extraction['paper_title'])

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ingest papers to RAG + KG")
    parser.add_argument('paper_dir', help='Directory containing PDFs')
    parser.add_argument('--collection', default='polymax_full', help='ChromaDB collection name')
    args = parser.parse_args()

    paper_dir = Path(args.paper_dir)
    pdfs = list(paper_dir.glob('*.pdf')) + list(paper_dir.glob('*.htm'))

    if not pdfs:
        print(f"No PDFs found in {paper_dir}")
        return

    print(f"\n{'='*60}")
    print(f"POLYMAX FULL INGESTION PIPELINE")
    print(f"{'='*60}")
    print(f"Papers to process: {len(pdfs)}")
    print(f"ChromaDB: {CHROMADB_PATH}")
    print(f"Neo4j: {NEO4J_URI}")
    print(f"{'='*60}\n")

    # Initialize clients
    print("Initializing embedding model...")
    model = SentenceTransformer('all-mpnet-base-v2')

    print("Connecting to ChromaDB...")
    os.makedirs(CHROMADB_PATH, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=CHROMADB_PATH)

    # Get or create collection
    try:
        collection = chroma_client.get_collection(args.collection)
        print(f"Using existing collection: {args.collection}")
    except:
        collection = chroma_client.create_collection(
            name=args.collection,
            metadata={"description": "PolyMax full paper corpus"}
        )
        print(f"Created new collection: {args.collection}")

    print("Connecting to Neo4j...")
    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    # Process papers
    total_chunks = 0
    total_entities = 0

    for pdf_path in tqdm(pdfs, desc="Processing papers"):
        try:
            # Extract text
            if pdf_path.suffix == '.htm':
                with open(pdf_path, 'r', errors='ignore') as f:
                    text = f.read()
            else:
                text = extract_text_from_pdf(str(pdf_path))

            if not text or len(text) < 100:
                print(f"\nSkipping {pdf_path.name} (no text extracted)")
                continue

            # Get title from filename
            title = pdf_path.stem.replace('_', ' ').replace('-', ' ')
            paper_id = pdf_path.stem

            # Chunk text for RAG
            chunks = chunk_text(text)

            if chunks:
                # Embed chunks
                embeddings = model.encode(chunks).tolist()

                # Add to ChromaDB
                chunk_ids = [f"{paper_id}_chunk_{i}" for i in range(len(chunks))]
                metadatas = [{"paper": title, "chunk": i, "source": pdf_path.name}
                            for i in range(len(chunks))]

                collection.add(
                    documents=chunks,
                    embeddings=embeddings,
                    ids=chunk_ids,
                    metadatas=metadatas
                )

                total_chunks += len(chunks)

            # Extract entities for KG
            extraction = extract_entities(text, title)
            extraction['paper_id'] = paper_id

            # Add to Neo4j
            add_to_neo4j(neo4j_driver, extraction, paper_id)
            total_entities += len(extraction.get('entities', []))

        except Exception as e:
            print(f"\nError processing {pdf_path.name}: {e}")
            continue

    neo4j_driver.close()

    # Summary
    print(f"\n{'='*60}")
    print(f"INGESTION COMPLETE")
    print(f"{'='*60}")
    print(f"Papers processed: {len(pdfs)}")
    print(f"RAG chunks added: {total_chunks}")
    print(f"KG entities extracted: {total_entities}")
    print(f"ChromaDB collection: {args.collection}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
