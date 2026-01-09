#!/usr/bin/env python3
"""
Step 1: Data Extraction & Field Labeling for Overlap Atlas

Extracts high-confidence concepts from passage_concepts and assigns
field labels to each passage based on domain-type concepts.

Usage:
    python scripts/overlap_atlas/build_overlap_tables.py
    python scripts/overlap_atlas/build_overlap_tables.py --min-confidence 0.8
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Set, Tuple

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
EXTRACTOR_VERSION = "gemini_batch_v1"
DEFAULT_MIN_CONFIDENCE = 0.7
OUTPUT_DIR = Path("/home/user/polymath-repo/var/overlap_atlas")

# Field-type concepts are used as primary field labels
FIELD_TYPES = {'field', 'domain'}


def extract_concepts(conn, min_confidence: float) -> pd.DataFrame:
    """Extract all high-confidence concepts from passage_concepts."""
    logger.info(f"Extracting concepts with confidence >= {min_confidence}")

    cur = conn.cursor()
    cur.execute("""
        SELECT
            passage_id::text,
            concept_name,
            concept_type,
            confidence,
            evidence->>'source_text' as source_text,
            evidence->'quality'->>'confidence' as evidence_confidence
        FROM passage_concepts
        WHERE extractor_version = %s
          AND confidence >= %s
        ORDER BY passage_id, confidence DESC
    """, (EXTRACTOR_VERSION, min_confidence))

    rows = cur.fetchall()
    cur.close()

    df = pd.DataFrame(rows, columns=[
        'passage_id', 'concept_name', 'concept_type',
        'confidence', 'source_text', 'evidence_confidence'
    ])

    logger.info(f"Extracted {len(df):,} concept associations")
    logger.info(f"Unique passages: {df['passage_id'].nunique():,}")
    logger.info(f"Unique concepts: {df['concept_name'].nunique():,}")

    return df


def assign_field_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Assign field labels to each passage based on domain/field type concepts."""
    logger.info("Assigning field labels to passages...")

    # Get field/domain type concepts for each passage
    field_concepts = df[df['concept_type'].isin(FIELD_TYPES)].copy()

    # Group by passage and aggregate field labels
    passage_fields = field_concepts.groupby('passage_id').agg({
        'concept_name': lambda x: list(set(x)),
        'confidence': 'mean'
    }).reset_index()
    passage_fields.columns = ['passage_id', 'field_labels', 'field_confidence']

    logger.info(f"Passages with explicit field labels: {len(passage_fields):,}")

    # For passages without field labels, use highest-confidence concept
    all_passages = df['passage_id'].unique()
    passages_with_fields = set(passage_fields['passage_id'])
    passages_without_fields = set(all_passages) - passages_with_fields

    if passages_without_fields:
        logger.info(f"Inferring labels for {len(passages_without_fields):,} passages without explicit fields")

        # Get top concept per passage (already sorted by confidence)
        top_concepts = df[df['passage_id'].isin(passages_without_fields)].groupby('passage_id').first().reset_index()

        inferred_fields = pd.DataFrame({
            'passage_id': top_concepts['passage_id'],
            'field_labels': top_concepts['concept_name'].apply(lambda x: [x]),
            'field_confidence': top_concepts['confidence']
        })

        passage_fields = pd.concat([passage_fields, inferred_fields], ignore_index=True)

    return passage_fields


def build_passage_summary(df: pd.DataFrame, passage_fields: pd.DataFrame) -> pd.DataFrame:
    """Build summary table with all concepts per passage."""
    logger.info("Building passage summary...")

    # Aggregate all concepts per passage
    passage_concepts = df.groupby('passage_id').agg({
        'concept_name': lambda x: list(x),
        'concept_type': lambda x: list(x),
        'confidence': lambda x: list(x),
        'source_text': 'first'  # Keep one source text sample
    }).reset_index()

    passage_concepts.columns = [
        'passage_id', 'concepts', 'concept_types',
        'confidences', 'sample_text'
    ]

    # Merge with field labels
    summary = passage_concepts.merge(passage_fields, on='passage_id', how='left')

    # Fill missing field labels
    summary['field_labels'] = summary['field_labels'].apply(
        lambda x: x if isinstance(x, list) else []
    )

    return summary


def compute_statistics(df: pd.DataFrame, summary: pd.DataFrame) -> Dict:
    """Compute dataset statistics for the report."""
    stats = {
        'timestamp': datetime.now().isoformat(),
        'extractor_version': EXTRACTOR_VERSION,
        'total_concept_associations': len(df),
        'unique_passages': df['passage_id'].nunique(),
        'unique_concepts': df['concept_name'].nunique(),
        'concept_type_distribution': df['concept_type'].value_counts().to_dict(),
        'confidence_stats': {
            'mean': float(df['confidence'].mean()),
            'median': float(df['confidence'].median()),
            'min': float(df['confidence'].min()),
            'max': float(df['confidence'].max())
        },
        'passages_with_explicit_fields': len(summary[summary['field_labels'].apply(len) > 0]),
        'top_fields': df[df['concept_type'].isin(FIELD_TYPES)]['concept_name'].value_counts().head(30).to_dict(),
        'top_concepts': df['concept_name'].value_counts().head(50).to_dict()
    }
    return stats


def save_outputs(df: pd.DataFrame, summary: pd.DataFrame, stats: Dict, output_dir: Path):
    """Save all outputs to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full concepts table (parquet for efficiency)
    concepts_path = output_dir / "passage_concepts.parquet"
    df.to_parquet(concepts_path, index=False)
    logger.info(f"Saved concepts to {concepts_path}")

    # Save passage summary (parquet)
    summary_path = output_dir / "passage_summary.parquet"
    summary.to_parquet(summary_path, index=False)
    logger.info(f"Saved summary to {summary_path}")

    # Save field labels as CSV for easy inspection
    fields_df = summary[['passage_id', 'field_labels', 'field_confidence']].copy()
    fields_df['field_labels'] = fields_df['field_labels'].apply(lambda x: ','.join(x) if x else '')
    fields_path = output_dir / "passage_fields.csv"
    fields_df.to_csv(fields_path, index=False)
    logger.info(f"Saved field labels to {fields_path}")

    # Save statistics
    stats_path = output_dir / "extraction_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved statistics to {stats_path}")

    # Print summary
    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    print(f"Total concept associations: {stats['total_concept_associations']:,}")
    print(f"Unique passages: {stats['unique_passages']:,}")
    print(f"Unique concepts: {stats['unique_concepts']:,}")
    print(f"Confidence: mean={stats['confidence_stats']['mean']:.3f}, median={stats['confidence_stats']['median']:.3f}")
    print(f"\nTop 10 fields/domains:")
    for field, count in list(stats['top_fields'].items())[:10]:
        print(f"  {field}: {count:,}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Build overlap atlas data tables")
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE,
                        help=f"Minimum confidence threshold (default: {DEFAULT_MIN_CONFIDENCE})")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help=f"Output directory (default: {OUTPUT_DIR})")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    logger.info("Starting Overlap Atlas data extraction...")
    logger.info(f"Min confidence: {args.min_confidence}")
    logger.info(f"Output directory: {output_dir}")

    # Connect to database
    conn = get_db_connection()

    try:
        # Step 1: Extract concepts
        df = extract_concepts(conn, args.min_confidence)

        if len(df) == 0:
            logger.error("No concepts found! Check database connection and filters.")
            sys.exit(1)

        # Step 2: Assign field labels
        passage_fields = assign_field_labels(df)

        # Step 3: Build passage summary
        summary = build_passage_summary(df, passage_fields)

        # Step 4: Compute statistics
        stats = compute_statistics(df, summary)

        # Step 5: Save outputs
        save_outputs(df, summary, stats, output_dir)

        logger.info("Data extraction complete!")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
