#!/usr/bin/env python3
"""
Paper Ingestion Orchestrator with Haiku Concept Extraction

This script is designed to be RUN BY Claude Code, which spawns fresh Haiku subagents
for each paper's concept extraction. This ensures clean context and avoids compaction.

Architecture:
1. Claude Code runs this script
2. Script identifies papers to process
3. For each paper, returns control to Claude Code
4. Claude Code spawns fresh Haiku subagent for concept extraction
5. Results flow back to complete ingestion

This is a "coordinator" script that works WITH Claude Code, not standalone.
"""

import sys
import json
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.unified_ingest import UnifiedIngestor


def get_papers_to_process(staging_dir: Path, limit: int = None) -> List[Path]:
    """Get list of PDF files to process."""
    pdfs = list(staging_dir.glob("*.pdf"))

    if limit:
        pdfs = pdfs[:limit]

    print(f"Found {len(pdfs)} PDFs to process")
    return pdfs


def prepare_paper_for_haiku(pdf_path: Path) -> Dict:
    """
    Prepare paper data for Haiku subagent processing.

    Returns dict with:
    - pdf_path: Path to PDF
    - context: Paper metadata (title, authors, year, DOI)
    - passages: List of passage texts
    """
    ingestor = UnifiedIngestor()

    # Extract PDF metadata and passages
    metadata = ingestor._extract_pdf_metadata(str(pdf_path))
    passages = ingestor._pdf_parser.extract_with_provenance(str(pdf_path))

    # Build paper context
    paper_context = {
        'title': metadata.get('title', 'Unknown'),
        'authors': metadata.get('authors', []),
        'year': metadata.get('year'),
        'doi': metadata.get('doi'),
        'pmid': metadata.get('pmid'),
    }

    # Get passage texts (all passages for full-text extraction)
    passage_texts = [p.passage_text for p in passages if p.passage_text]

    return {
        'pdf_path': str(pdf_path),
        'context': paper_context,
        'passages': passage_texts,
        'passage_objects': passages,  # Keep full Passage objects for later
    }


def complete_ingestion_with_concepts(
    pdf_path: Path,
    concepts: List[str]
) -> Dict:
    """
    Complete paper ingestion using Haiku-extracted concepts.

    This is called AFTER Haiku subagent has extracted concepts.

    Args:
        pdf_path: Path to PDF file
        concepts: List of concepts extracted by Haiku

    Returns:
        Ingestion result dict
    """
    ingestor = UnifiedIngestor()

    # Re-extract passages and metadata
    metadata = ingestor._extract_pdf_metadata(str(pdf_path))
    passages = ingestor._pdf_parser.extract_with_provenance(str(pdf_path))

    # Compute doc_id
    doc_id = ingestor.compute_doc_id(
        title=metadata.get('title'),
        authors=metadata.get('authors'),
        year=metadata.get('year'),
        doi=metadata.get('doi'),
        pmid=metadata.get('pmid'),
        arxiv_id=metadata.get('arxiv_id')
    )

    # Sync to Postgres (REQUIRED - fail-closed)
    ingestor._sync_postgres_passages(
        doc_id=doc_id,
        passages=passages,
        metadata=metadata
    )

    # Normalize concepts
    from lib.kb_derived import normalize_concept_name
    normalized_concepts = [normalize_concept_name(c) for c in concepts if c]

    # Embed and index in ChromaDB
    ingestor._index_passages_chromadb(
        passages=passages,
        doc_id=doc_id,
        metadata=metadata,
        concepts=normalized_concepts
    )

    # Create Neo4j graph
    if ingestor.neo4j_session:
        ingestor._create_neo4j_nodes(
            doc_id=doc_id,
            metadata=metadata,
            concepts=normalized_concepts
        )

    print(f"✓ Ingested: {metadata.get('title', 'Unknown')[:80]}")
    print(f"  Concepts: {', '.join(normalized_concepts[:5])}...")

    return {
        'doc_id': str(doc_id),
        'title': metadata.get('title'),
        'concepts': normalized_concepts,
        'passages': len(passages),
    }


def main():
    """
    Main orchestration function.

    NOTE: This is meant to be called BY Claude Code in a loop:

    1. Run this script to get list of papers
    2. For each paper batch (3 at a time):
       a. Call prepare_paper_for_haiku() to get paper data
       b. Spawn fresh Haiku subagent with Task tool
       c. Call complete_ingestion_with_concepts() with results
    """
    staging_dir = Path("/home/user/polymath-repo/ingest_staging")

    papers = get_papers_to_process(staging_dir, limit=10)

    # Return paper list for Claude Code to orchestrate
    print("\nReady for Haiku orchestration")
    print(f"Papers to process: {len(papers)}")
    print("\nNext step: Claude Code should:")
    print("1. Batch papers (3 at a time)")
    print("2. For each batch, spawn fresh Haiku subagents")
    print("3. Call complete_ingestion_with_concepts() with results")

    return papers


if __name__ == "__main__":
    main()
