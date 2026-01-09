#!/usr/bin/env python3
"""
Backfill passage_concepts using Haiku subagents.

ARCHITECTURE: This script provides helper functions for orchestration.
The actual extraction is done by Claude Code spawning fresh Haiku subagents
via the Task tool - this script CANNOT do that directly.

Usage (from Claude Code session):
    1. Import functions from this script
    2. Fetch batches of passages
    3. Spawn Haiku subagents for concept extraction
    4. Write results back to database

Performance: ~100-200 passages/min with Haiku (vs 27/min with local LLM)
Target: 543,867 eligible passages
Estimated time: ~2.5 days with Haiku
"""

import sys
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.kb_derived import (
    upsert_passage_concept,
    update_migration_checkpoint,
    get_migration_checkpoint
)
from lib.db import get_db_connection

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
JOB_NAME = "backfill_passage_concepts_haiku_v1"
EXTRACTOR_MODEL = "claude-haiku"
EXTRACTOR_VERSION = "haiku_v1"
MIN_TEXT_LENGTH = 100  # Skip very short passages


def fetch_passages_batch(
    conn,
    start_after_id: Optional[str] = None,
    batch_size: int = 100
) -> List[Tuple[str, str, str, Optional[int]]]:
    """
    Fetch batch of passages needing concept extraction.

    Args:
        conn: Database connection
        start_after_id: UUID to continue from (for pagination)
        batch_size: Number of passages to fetch

    Returns:
        List of (passage_id, passage_text, doc_title, doc_year)
    """
    cursor = conn.cursor()

    try:
        if start_after_id:
            query = """
            SELECT p.passage_id::text, p.passage_text, d.title, d.year
            FROM passages p
            JOIN documents d ON p.doc_id = d.doc_id
            WHERE p.passage_id > %s::uuid
              AND p.passage_id NOT IN (
                  SELECT passage_id FROM passage_concepts
                  WHERE extractor_version = %s
              )
              AND length(p.passage_text) >= %s
            ORDER BY p.passage_id
            LIMIT %s
            """
            cursor.execute(query, (start_after_id, EXTRACTOR_VERSION, MIN_TEXT_LENGTH, batch_size))
        else:
            query = """
            SELECT p.passage_id::text, p.passage_text, d.title, d.year
            FROM passages p
            JOIN documents d ON p.doc_id = d.doc_id
            WHERE p.passage_id NOT IN (
                SELECT passage_id FROM passage_concepts
                WHERE extractor_version = %s
            )
              AND length(p.passage_text) >= %s
            ORDER BY p.passage_id
            LIMIT %s
            """
            cursor.execute(query, (EXTRACTOR_VERSION, MIN_TEXT_LENGTH, batch_size))

        return cursor.fetchall()

    finally:
        cursor.close()


def count_remaining_passages(conn) -> int:
    """Count passages still needing concept extraction."""
    cursor = conn.cursor()
    try:
        cursor.execute("""
        SELECT COUNT(*) FROM passages p
        WHERE p.passage_id NOT IN (
            SELECT passage_id FROM passage_concepts
            WHERE extractor_version = %s
        )
        AND length(p.passage_text) >= %s
        """, (EXTRACTOR_VERSION, MIN_TEXT_LENGTH))
        return cursor.fetchone()[0]
    finally:
        cursor.close()


def build_extraction_prompt(
    passage_text: str,
    doc_title: str,
    doc_year: Optional[int]
) -> str:
    """
    Build prompt for Haiku concept extraction.

    Args:
        passage_text: Text to extract concepts from
        doc_title: Paper title for context
        doc_year: Publication year

    Returns:
        Formatted prompt string
    """
    # Truncate very long passages
    text = passage_text[:3500]

    prompt = f"""Extract scientific concepts from this paper passage.

Paper: {doc_title} ({doc_year or 'Unknown'})

Passage:
{text}

Instructions:
1. Extract key concepts: methods, techniques, models, datasets, domains
2. Use snake_case normalization (e.g., "Graph Neural Network" → "graph_neural_network")
3. Include cross-domain concepts that connect fields
4. Skip generic terms (analysis, method, data, result, study)
5. Include 3-15 concepts per passage

Return ONLY a JSON array of concept strings, no explanations:
["concept_one", "concept_two", "concept_three"]

Example output:
["spatial_transcriptomics", "graph_neural_network", "optimal_transport", "deconvolution", "foundation_model"]"""

    return prompt


def parse_haiku_response(response: str) -> List[str]:
    """
    Parse Haiku response to extract concept list.

    Args:
        response: Raw Haiku response text

    Returns:
        List of normalized concept names
    """
    try:
        # Find JSON array in response
        json_match = re.search(r'\[.*?\]', response, re.DOTALL)
        if json_match:
            concepts_raw = json.loads(json_match.group(0))
        else:
            # Fallback: split by lines/commas
            lines = response.strip().split('\n')
            concepts_raw = []
            for line in lines:
                line = line.strip().strip('-').strip('•').strip('"\'')
                if line and not line.startswith('[') and not line.startswith('{'):
                    concepts_raw.append(line)

        # Normalize each concept
        concepts = []
        for concept in concepts_raw:
            if isinstance(concept, str):
                normalized = normalize_concept(concept)
                if normalized and normalized not in concepts:
                    concepts.append(normalized)

        return concepts

    except Exception as e:
        logger.error(f"Failed to parse response: {e}")
        logger.debug(f"Raw response: {response[:500]}")
        return []


def normalize_concept(name: str) -> str:
    """
    Normalize concept name to snake_case.

    Args:
        name: Raw concept name

    Returns:
        Normalized snake_case concept name (or empty string if invalid)
    """
    if not name:
        return ""

    # Common alias mappings
    alias_map = {
        "gnn": "graph_neural_network",
        "gnns": "graph_neural_network",
        "graph neural network": "graph_neural_network",
        "graph neural networks": "graph_neural_network",
        "transformer": "transformer",
        "attention mechanism": "attention",
        "self-attention": "attention",
        "visium": "spatial_transcriptomics",
        "10x": "spatial_transcriptomics",
        "10x genomics": "spatial_transcriptomics",
        "spatial transcriptomics": "spatial_transcriptomics",
        "single cell": "single_cell",
        "single-cell": "single_cell",
        "scrna-seq": "single_cell_rna_seq",
        "scrnaseq": "single_cell_rna_seq",
        "optimal transport": "optimal_transport",
        "wasserstein": "optimal_transport",
        "compressed sensing": "compressed_sensing",
        "foundation model": "foundation_model",
        "foundation models": "foundation_model",
        "causal inference": "causal_inference",
        "manifold learning": "manifold_learning",
        "topological data analysis": "topological_data_analysis",
        "tda": "topological_data_analysis",
        "active inference": "active_inference",
        "free energy": "free_energy",
        "deconvolution": "deconvolution",
        "image segmentation": "image_segmentation",
        "cell segmentation": "cell_segmentation",
        "gene expression": "gene_expression",
    }

    # Clean and lowercase
    name = name.strip().strip('"\'').lower()

    # Check alias map first
    if name in alias_map:
        return alias_map[name]

    # Convert to snake_case
    name = re.sub(r'[\s\-]+', '_', name)
    name = re.sub(r'[^a-z0-9_]', '', name)
    name = name.strip('_')

    # Remove stopwords
    stopwords = {'the', 'a', 'an', 'and', 'or', 'of', 'in', 'for', 'to', 'on', 'with', 'by'}
    parts = [p for p in name.split('_') if p not in stopwords]
    name = '_'.join(parts)

    # Skip if too short or generic
    generic = {'method', 'analysis', 'data', 'result', 'study', 'approach', 'model',
               'system', 'based', 'using', 'paper', 'work'}
    if len(name) < 3 or name in generic:
        return ""

    return name


def save_concepts_for_passage(
    conn,
    passage_id: str,
    concepts: List[str],
    evidence: Optional[Dict] = None
) -> int:
    """
    Save extracted concepts to database.

    Args:
        conn: Database connection
        passage_id: Passage UUID
        concepts: List of concept names
        evidence: Optional evidence dict

    Returns:
        Number of concepts saved
    """
    saved = 0
    for concept in concepts:
        if not concept:
            continue

        try:
            upsert_passage_concept(
                conn,
                passage_id=passage_id,
                concept_name=concept,
                concept_type="extracted",
                aliases=[],
                confidence=0.85,  # Haiku baseline confidence
                extractor_model=EXTRACTOR_MODEL,
                extractor_version=EXTRACTOR_VERSION,
                evidence=evidence
            )
            saved += 1
        except Exception as e:
            logger.error(f"Failed to save concept {concept} for {passage_id}: {e}")

    return saved


def checkpoint(
    conn,
    cursor_position: str,
    processed: int,
    failed: int,
    status: str = "running"
) -> None:
    """Save checkpoint for resumability."""
    update_migration_checkpoint(
        conn,
        job_name=JOB_NAME,
        cursor_position=cursor_position,
        status=status,
        items_processed=processed,
        items_failed=failed,
        cursor_type="passage_id"
    )


def get_checkpoint(conn) -> Optional[str]:
    """Get last checkpoint position."""
    result = get_migration_checkpoint(conn, JOB_NAME)
    if result and result["status"] != "completed":
        return result["cursor_position"]
    return None


def get_stats(conn) -> Dict[str, Any]:
    """Get current backfill statistics."""
    cursor = conn.cursor()
    try:
        # Total passages
        cursor.execute("SELECT COUNT(*) FROM passages WHERE length(passage_text) >= %s", (MIN_TEXT_LENGTH,))
        total = cursor.fetchone()[0]

        # With concepts
        cursor.execute("""
        SELECT COUNT(DISTINCT passage_id) FROM passage_concepts
        WHERE extractor_version = %s
        """, (EXTRACTOR_VERSION,))
        with_concepts = cursor.fetchone()[0]

        # Checkpoint info
        checkpoint_info = get_migration_checkpoint(conn, JOB_NAME)

        return {
            "total_eligible": total,
            "with_concepts": with_concepts,
            "remaining": total - with_concepts,
            "percent_complete": round(100 * with_concepts / total, 2) if total > 0 else 0,
            "checkpoint": checkpoint_info
        }
    finally:
        cursor.close()


# For testing
if __name__ == "__main__":
    conn = get_db_connection()

    print("=== Passage Concept Backfill Status ===")
    stats = get_stats(conn)
    print(f"Total eligible: {stats['total_eligible']:,}")
    print(f"With concepts:  {stats['with_concepts']:,}")
    print(f"Remaining:      {stats['remaining']:,}")
    print(f"Progress:       {stats['percent_complete']}%")

    if stats['checkpoint']:
        print(f"\nCheckpoint:")
        print(f"  Status: {stats['checkpoint']['status']}")
        print(f"  Position: {stats['checkpoint']['cursor_position']}")
        print(f"  Processed: {stats['checkpoint']['items_processed']}")

    # Show sample batch
    print("\n=== Sample Batch (first 3) ===")
    batch = fetch_passages_batch(conn, batch_size=3)
    for passage_id, text, title, year in batch:
        print(f"\nPassage: {passage_id}")
        print(f"  Paper: {title[:60]}... ({year})")
        print(f"  Text: {text[:100]}...")

    conn.close()
