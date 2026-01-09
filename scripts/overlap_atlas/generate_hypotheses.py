#!/usr/bin/env python3
"""
Step 3: Evidence-Bound Hypothesis Generation for Overlap Atlas

Generates novel hypotheses from field overlaps and bridge concepts,
grounded in actual evidence from the corpus.

Usage:
    python scripts/overlap_atlas/generate_hypotheses.py
    python scripts/overlap_atlas/generate_hypotheses.py --top 30
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional
import random

import pandas as pd

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lib.db import get_db_connection

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
INPUT_DIR = Path("/home/user/polymath-repo/var/overlap_atlas")
OUTPUT_DIR = INPUT_DIR
EXTRACTOR_VERSION = "gemini_batch_v1"
MIN_EVIDENCE_PASSAGES = 2
MAX_EVIDENCE_PASSAGES = 5


def load_overlap_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load overlap scoring outputs from Step 2."""
    field_overlap = pd.read_csv(INPUT_DIR / "field_field_overlap.csv")
    bridge_concepts = pd.read_csv(INPUT_DIR / "bridge_concepts.csv")
    rare_pairs = pd.read_csv(INPUT_DIR / "rare_pairs.csv")

    logger.info(f"Loaded {len(field_overlap)} field overlaps")
    logger.info(f"Loaded {len(bridge_concepts)} bridge concepts")
    logger.info(f"Loaded {len(rare_pairs)} rare pairs")

    return field_overlap, bridge_concepts, rare_pairs


def get_evidence_passages(conn, concept: str, field: Optional[str] = None,
                          limit: int = MAX_EVIDENCE_PASSAGES) -> List[Dict]:
    """Retrieve passages with evidence for a concept, optionally filtered by field."""
    cur = conn.cursor()

    if field:
        # Get passages that have both the concept and the field
        cur.execute("""
            SELECT DISTINCT ON (pc.passage_id)
                pc.passage_id::text,
                pc.concept_name,
                pc.confidence,
                pc.evidence->>'source_text' as source_text,
                pc.evidence->'quality'->>'confidence' as evidence_confidence,
                LEFT(p.passage_text, 500) as passage_preview
            FROM passage_concepts pc
            JOIN passages p ON p.passage_id = pc.passage_id
            WHERE pc.extractor_version = %s
              AND pc.concept_name ILIKE %s
              AND EXISTS (
                  SELECT 1 FROM passage_concepts pc2
                  WHERE pc2.passage_id = pc.passage_id
                    AND pc2.extractor_version = %s
                    AND pc2.concept_name ILIKE %s
              )
              AND pc.confidence >= 0.7
            ORDER BY pc.passage_id, pc.confidence DESC
            LIMIT %s
        """, (EXTRACTOR_VERSION, f"%{concept}%", EXTRACTOR_VERSION, f"%{field}%", limit))
    else:
        # Get passages for concept only
        cur.execute("""
            SELECT DISTINCT ON (pc.passage_id)
                pc.passage_id::text,
                pc.concept_name,
                pc.confidence,
                pc.evidence->>'source_text' as source_text,
                pc.evidence->'quality'->>'confidence' as evidence_confidence,
                LEFT(p.passage_text, 500) as passage_preview
            FROM passage_concepts pc
            JOIN passages p ON p.passage_id = pc.passage_id
            WHERE pc.extractor_version = %s
              AND pc.concept_name ILIKE %s
              AND pc.confidence >= 0.7
            ORDER BY pc.passage_id, pc.confidence DESC
            LIMIT %s
        """, (EXTRACTOR_VERSION, f"%{concept}%", limit))

    rows = cur.fetchall()
    cur.close()

    passages = []
    for row in rows:
        passages.append({
            'passage_id': row[0],
            'concept': row[1],
            'confidence': float(row[2]) if row[2] else 0.8,
            'source_text': row[3],
            'evidence_confidence': row[4],
            'passage_preview': row[5]
        })

    return passages


def generate_field_overlap_hypotheses(conn, field_overlap: pd.DataFrame,
                                      top_n: int = 10) -> List[Dict]:
    """Generate hypotheses from top field-field overlaps."""
    logger.info(f"Generating hypotheses from top {top_n} field overlaps...")

    hypotheses = []
    top_overlaps = field_overlap.head(top_n * 2)  # Get extra to handle failures

    for _, row in top_overlaps.iterrows():
        field_a = row['field_a']
        field_b = row['field_b']
        pmi = row['pmi']
        count = row['count']

        # Get evidence passages for each field's intersection
        evidence_a = get_evidence_passages(conn, field_a, field_b, MAX_EVIDENCE_PASSAGES)
        evidence_b = get_evidence_passages(conn, field_b, field_a, MAX_EVIDENCE_PASSAGES)

        all_evidence = evidence_a + evidence_b
        if len(all_evidence) < MIN_EVIDENCE_PASSAGES:
            continue

        # Compute hypothesis score
        mean_confidence = sum(e['confidence'] for e in all_evidence) / len(all_evidence)
        evidence_coverage = min(1.0, len(all_evidence) / (2 * MAX_EVIDENCE_PASSAGES))
        score = pmi * mean_confidence * (0.5 + 0.5 * evidence_coverage)

        # Generate hypothesis text
        hypothesis = {
            'id': f"FO_{field_a[:20]}_{field_b[:20]}",
            'type': 'field_overlap',
            'title': f"Cross-pollination between {field_a.replace('_', ' ')} and {field_b.replace('_', ' ')}",
            'fields': [field_a, field_b],
            'bridge_concepts': [],
            'hypothesis_text': (
                f"The strong co-occurrence of {field_a.replace('_', ' ')} and {field_b.replace('_', ' ')} "
                f"(PMI={pmi:.2f}, n={count}) suggests potential for methodological transfer. "
                f"Techniques developed in one domain may address challenges in the other."
            ),
            'pmi': pmi,
            'cooccurrence_count': count,
            'evidence_passages': all_evidence[:MAX_EVIDENCE_PASSAGES],
            'mean_confidence': mean_confidence,
            'evidence_coverage': evidence_coverage,
            'score': score,
            'generated_at': datetime.now().isoformat()
        }

        hypotheses.append(hypothesis)

        if len(hypotheses) >= top_n:
            break

    logger.info(f"Generated {len(hypotheses)} field overlap hypotheses")
    return hypotheses


def generate_bridge_hypotheses(conn, bridge_concepts: pd.DataFrame,
                               field_overlap: pd.DataFrame,
                               top_n: int = 10) -> List[Dict]:
    """Generate hypotheses from bridge concepts."""
    logger.info(f"Generating hypotheses from top {top_n} bridge concepts...")

    hypotheses = []
    top_bridges = bridge_concepts.head(top_n * 2)

    # Get top fields for context
    top_fields = set(field_overlap['field_a'].head(20)) | set(field_overlap['field_b'].head(20))

    for _, row in top_bridges.iterrows():
        concept = row['concept']
        betweenness = row['betweenness']
        degree = row['degree']
        connected_fields = row.get('connected_fields', '')

        if isinstance(connected_fields, str):
            fields_list = [f.strip() for f in connected_fields.split(',') if f.strip()]
        else:
            fields_list = []

        # Filter to top fields for relevance
        relevant_fields = [f for f in fields_list if f in top_fields][:4]

        if len(relevant_fields) < 2:
            # Use any fields if no overlap with top
            relevant_fields = fields_list[:4]

        if len(relevant_fields) < 2:
            continue

        # Get evidence passages for the bridge concept
        evidence = get_evidence_passages(conn, concept, None, MAX_EVIDENCE_PASSAGES)

        if len(evidence) < MIN_EVIDENCE_PASSAGES:
            continue

        mean_confidence = sum(e['confidence'] for e in evidence) / len(evidence)
        evidence_coverage = min(1.0, len(evidence) / MAX_EVIDENCE_PASSAGES)

        # Score: betweenness * field diversity * confidence
        field_diversity = min(1.0, len(relevant_fields) / 4)
        score = betweenness * 1000 * mean_confidence * field_diversity

        # Generate hypothesis text
        field_str = ', '.join([f.replace('_', ' ') for f in relevant_fields[:3]])
        hypothesis = {
            'id': f"BC_{concept[:30]}",
            'type': 'bridge_concept',
            'title': f"'{concept.replace('_', ' ')}' as interdisciplinary bridge",
            'fields': relevant_fields,
            'bridge_concepts': [concept],
            'hypothesis_text': (
                f"The concept '{concept.replace('_', ' ')}' bridges {degree} fields "
                f"including {field_str}. "
                f"Its high betweenness centrality ({betweenness:.4f}) suggests it may serve as "
                f"a translation layer for insights between these domains."
            ),
            'betweenness': betweenness,
            'connected_fields_count': degree,
            'evidence_passages': evidence[:MAX_EVIDENCE_PASSAGES],
            'mean_confidence': mean_confidence,
            'evidence_coverage': evidence_coverage,
            'score': score,
            'generated_at': datetime.now().isoformat()
        }

        hypotheses.append(hypothesis)

        if len(hypotheses) >= top_n:
            break

    logger.info(f"Generated {len(hypotheses)} bridge concept hypotheses")
    return hypotheses


def generate_rare_pair_hypotheses(conn, rare_pairs: pd.DataFrame,
                                  top_n: int = 10) -> List[Dict]:
    """Generate hypotheses from rare but strong cross-field concept pairs."""
    logger.info(f"Generating hypotheses from top {top_n} rare pairs...")

    hypotheses = []
    top_pairs = rare_pairs.head(top_n * 2)

    for _, row in top_pairs.iterrows():
        concept_a = row['concept_a']
        concept_b = row['concept_b']
        count = row['count']
        avg_conf = row['avg_confidence']
        n_field_pairs = row['n_field_pairs']
        pair_score = row['score']

        # Get evidence for both concepts together
        evidence_a = get_evidence_passages(conn, concept_a, concept_b, 3)
        evidence_b = get_evidence_passages(conn, concept_b, concept_a, 3)

        all_evidence = evidence_a + evidence_b
        if len(all_evidence) < MIN_EVIDENCE_PASSAGES:
            continue

        mean_confidence = sum(e['confidence'] for e in all_evidence) / len(all_evidence)
        evidence_coverage = min(1.0, len(all_evidence) / MAX_EVIDENCE_PASSAGES)

        # Score combines rarity, confidence, and diversity
        score = pair_score * mean_confidence * (1 + evidence_coverage)

        hypothesis = {
            'id': f"RP_{concept_a[:15]}_{concept_b[:15]}",
            'type': 'rare_pair',
            'title': f"Unexpected connection: {concept_a.replace('_', ' ')} meets {concept_b.replace('_', ' ')}",
            'fields': [],  # Inferred from field pairs
            'bridge_concepts': [concept_a, concept_b],
            'hypothesis_text': (
                f"The rare co-occurrence of '{concept_a.replace('_', ' ')}' and "
                f"'{concept_b.replace('_', ' ')}' across {n_field_pairs} different field pairs "
                f"(n={count}, avg confidence={avg_conf:.2f}) suggests an underexplored connection. "
                f"This combination may reveal novel mechanisms or methodological opportunities."
            ),
            'cooccurrence_count': count,
            'n_field_pairs': n_field_pairs,
            'evidence_passages': all_evidence[:MAX_EVIDENCE_PASSAGES],
            'mean_confidence': mean_confidence,
            'evidence_coverage': evidence_coverage,
            'score': score,
            'generated_at': datetime.now().isoformat()
        }

        hypotheses.append(hypothesis)

        if len(hypotheses) >= top_n:
            break

    logger.info(f"Generated {len(hypotheses)} rare pair hypotheses")
    return hypotheses


def rank_and_dedupe_hypotheses(all_hypotheses: List[Dict]) -> List[Dict]:
    """Rank hypotheses by score and remove duplicates."""
    # Sort by score descending
    all_hypotheses.sort(key=lambda h: h['score'], reverse=True)

    # Remove hypotheses with overlapping concepts/fields
    seen_concepts = set()
    seen_fields = set()
    deduplicated = []

    for h in all_hypotheses:
        # Get key concepts/fields
        key_items = set(h.get('bridge_concepts', [])) | set(h.get('fields', []))

        # Check overlap
        overlap = key_items & (seen_concepts | seen_fields)
        if len(overlap) > 0 and len(deduplicated) > 5:
            # Allow some overlap in top 5, then filter
            continue

        deduplicated.append(h)
        seen_concepts.update(h.get('bridge_concepts', []))
        seen_fields.update(h.get('fields', []))

    return deduplicated


def save_hypotheses(hypotheses: List[Dict], output_dir: Path):
    """Save hypotheses to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "hypotheses.json"

    output = {
        'generated_at': datetime.now().isoformat(),
        'total_hypotheses': len(hypotheses),
        'by_type': {
            'field_overlap': len([h for h in hypotheses if h['type'] == 'field_overlap']),
            'bridge_concept': len([h for h in hypotheses if h['type'] == 'bridge_concept']),
            'rare_pair': len([h for h in hypotheses if h['type'] == 'rare_pair'])
        },
        'hypotheses': hypotheses
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"Saved {len(hypotheses)} hypotheses to {output_path}")

    # Print summary
    print("\n" + "="*60)
    print("HYPOTHESIS GENERATION SUMMARY")
    print("="*60)
    print(f"Total hypotheses: {len(hypotheses)}")
    print(f"  Field overlap: {output['by_type']['field_overlap']}")
    print(f"  Bridge concept: {output['by_type']['bridge_concept']}")
    print(f"  Rare pair: {output['by_type']['rare_pair']}")
    print(f"\nTop 5 hypotheses by score:")
    for i, h in enumerate(hypotheses[:5], 1):
        print(f"  {i}. [{h['type']}] {h['title'][:50]}...")
        print(f"     Score: {h['score']:.3f}, Evidence: {len(h['evidence_passages'])} passages")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Generate evidence-bound hypotheses")
    parser.add_argument("--top", type=int, default=10,
                        help="Number of hypotheses per type (default: 10)")
    parser.add_argument("--total", type=int, default=20,
                        help="Total hypotheses to output (default: 20)")
    args = parser.parse_args()

    logger.info("Starting hypothesis generation...")

    # Load overlap data
    field_overlap, bridge_concepts, rare_pairs = load_overlap_data()

    # Connect to database
    conn = get_db_connection()

    try:
        # Generate hypotheses from each source
        field_hypotheses = generate_field_overlap_hypotheses(conn, field_overlap, args.top)
        bridge_hypotheses = generate_bridge_hypotheses(conn, bridge_concepts, field_overlap, args.top)
        rare_hypotheses = generate_rare_pair_hypotheses(conn, rare_pairs, args.top)

        # Combine and rank
        all_hypotheses = field_hypotheses + bridge_hypotheses + rare_hypotheses
        ranked = rank_and_dedupe_hypotheses(all_hypotheses)

        # Take top N
        final_hypotheses = ranked[:args.total]

        # Save
        save_hypotheses(final_hypotheses, OUTPUT_DIR)

        logger.info("Hypothesis generation complete!")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
