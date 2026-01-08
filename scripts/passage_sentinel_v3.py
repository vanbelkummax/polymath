#!/usr/bin/env python3
"""
Passage Sentinel v3: Quality-Gated Evidence Extraction Monitor

Samples passages and reports:
- Text quality distribution
- Extraction coverage (% with concepts)
- Concept counts by support type (literal/normalized/inferred/none)
- Grant-grade citation rate (literal + high confidence + clean text)
- Top concepts by support type

Usage:
    python3 scripts/passage_sentinel_v3.py --samples 200
    python3 scripts/passage_sentinel_v3.py --samples 50 --dry-run
"""

import sys
import os
import logging
import argparse
import random
from collections import Counter, defaultdict
from typing import Dict, List, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.db import get_db_connection
from lib.local_extractor import LocalEntityExtractor
from lib.text_quality import text_quality

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def sample_passages(conn, n: int = 200) -> List[tuple]:
    """Sample N random passages from database."""
    cursor = conn.cursor()
    try:
        # Get total count
        cursor.execute("SELECT COUNT(*) FROM passages WHERE length(passage_text) >= 50")
        total = cursor.fetchone()[0]

        # Calculate sampling stride
        stride = max(1, total // n)

        # Sample with stride
        cursor.execute("""
            SELECT passage_id::text, passage_text
            FROM passages
            WHERE length(passage_text) >= 50
            ORDER BY passage_id
        """)

        all_passages = cursor.fetchall()
        sampled = [all_passages[i] for i in range(0, len(all_passages), stride)][:n]

        logger.info(f"Sampled {len(sampled)} passages from {total} total")
        return sampled
    finally:
        cursor.close()


def analyze_passage_quality(passages: List[tuple]) -> Dict[str, Any]:
    """Analyze text quality distribution."""
    quality_labels = Counter()
    quality_scores = []

    for pid, text in passages:
        q = text_quality(text)
        quality_labels[q['label']] += 1
        quality_scores.append(q['score'])

    return {
        'labels': dict(quality_labels),
        'avg_score': sum(quality_scores) / len(quality_scores) if quality_scores else 0,
        'min_score': min(quality_scores) if quality_scores else 0,
        'max_score': max(quality_scores) if quality_scores else 0
    }


def extract_and_analyze(
    passages: List[tuple],
    extractor: LocalEntityExtractor
) -> Dict[str, Any]:
    """
    Extract concepts from passages and analyze results.

    Returns:
        Dict with metrics:
        - coverage: % of passages with >= 1 concept
        - support_counts: Counter by support type
        - grant_grade_count: concepts with literal support + high conf + clean text
        - top_concepts_by_support: dict of {support_type: [top concepts]}
    """
    coverage_count = 0
    support_counts = Counter()
    grant_grade_count = 0
    concepts_by_support = defaultdict(list)

    for i, (pid, text) in enumerate(passages):
        if (i + 1) % 50 == 0:
            logger.info(f"Processed {i + 1}/{len(passages)} passages")

        try:
            concepts = extractor.extract_concepts_with_evidence(text)

            if concepts:
                coverage_count += 1

            for concept in concepts:
                evidence = concept.get('evidence', {})
                support = evidence.get('support', 'unknown')
                quality = evidence.get('quality', {})
                confidence = concept.get('confidence', 0.0)

                support_counts[support] += 1

                # Track for top concepts
                concepts_by_support[support].append({
                    'canonical': concept['canonical'],
                    'confidence': confidence,
                    'quality_score': quality.get('score', 0.0)
                })

                # Check grant-grade criteria
                if (support == 'literal' and
                    confidence >= 0.8 and
                    quality.get('score', 0) >= 0.5):
                    grant_grade_count += 1

        except Exception as e:
            logger.warning(f"Failed to extract from passage {pid}: {e}")

    # Calculate top concepts by support type
    top_concepts_by_support = {}
    for support_type, concept_list in concepts_by_support.items():
        # Sort by confidence, take top 10
        sorted_concepts = sorted(
            concept_list,
            key=lambda x: x['confidence'],
            reverse=True
        )[:10]
        top_concepts_by_support[support_type] = sorted_concepts

    return {
        'coverage': coverage_count / len(passages) if passages else 0,
        'support_counts': dict(support_counts),
        'grant_grade_count': grant_grade_count,
        'top_concepts_by_support': top_concepts_by_support
    }


def print_report(
    quality_analysis: Dict[str, Any],
    extraction_analysis: Dict[str, Any],
    n_passages: int
):
    """Print formatted sentinel report."""
    print("\n" + "="*70)
    print("PASSAGE SENTINEL V3 REPORT")
    print("="*70)

    print(f"\n📊 Sample Size: {n_passages} passages")

    print("\n" + "-"*70)
    print("TEXT QUALITY DISTRIBUTION")
    print("-"*70)
    for label, count in sorted(quality_analysis['labels'].items(), key=lambda x: -x[1]):
        pct = 100 * count / n_passages
        print(f"  {label:12s}: {count:4d} ({pct:5.1f}%)")

    print(f"\n  Average Score: {quality_analysis['avg_score']:.3f}")
    print(f"  Range: [{quality_analysis['min_score']:.3f}, {quality_analysis['max_score']:.3f}]")

    print("\n" + "-"*70)
    print("EXTRACTION COVERAGE")
    print("-"*70)
    coverage_pct = 100 * extraction_analysis['coverage']
    print(f"  Passages with ≥1 concept: {coverage_pct:.1f}%")

    print("\n" + "-"*70)
    print("CONCEPTS BY SUPPORT TYPE")
    print("-"*70)
    support_counts = extraction_analysis['support_counts']
    total_concepts = sum(support_counts.values())

    if total_concepts > 0:
        for support_type in ['literal', 'normalized', 'inferred', 'none']:
            count = support_counts.get(support_type, 0)
            pct = 100 * count / total_concepts
            print(f"  {support_type:12s}: {count:5d} ({pct:5.1f}%)")

        print(f"\n  Total Concepts: {total_concepts}")
    else:
        print("  No concepts extracted")

    print("\n" + "-"*70)
    print("GRANT-GRADE CITATIONS")
    print("-"*70)
    grant_grade = extraction_analysis['grant_grade_count']
    if total_concepts > 0:
        grant_pct = 100 * grant_grade / total_concepts
        print(f"  Grant-grade concepts: {grant_grade} ({grant_pct:.1f}%)")
        print("  Criteria: literal support + confidence ≥ 0.8 + quality ≥ 0.5")
    else:
        print("  No concepts to evaluate")

    print("\n" + "-"*70)
    print("TOP CONCEPTS BY SUPPORT TYPE")
    print("-"*70)
    for support_type in ['literal', 'normalized', 'inferred', 'none']:
        concepts = extraction_analysis['top_concepts_by_support'].get(support_type, [])
        if concepts:
            print(f"\n  {support_type.upper()}:")
            for i, c in enumerate(concepts[:5], 1):
                print(f"    {i}. {c['canonical']:30s} (conf={c['confidence']:.2f}, q={c['quality_score']:.2f})")

    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description="Passage Sentinel v3: Quality monitoring")
    parser.add_argument('--samples', type=int, default=200, help='Number of passages to sample')
    parser.add_argument('--dry-run', action='store_true', help='Quick test mode')
    args = parser.parse_args()

    n_samples = 50 if args.dry_run else args.samples

    logger.info("Starting Passage Sentinel v3")

    # Connect to database
    conn = get_db_connection()
    if not conn:
        logger.error("Failed to connect to database")
        return 1

    try:
        # Sample passages
        logger.info(f"Sampling {n_samples} passages...")
        passages = sample_passages(conn, n_samples)

        if not passages:
            logger.error("No passages found")
            return 1

        # Analyze text quality
        logger.info("Analyzing text quality...")
        quality_analysis = analyze_passage_quality(passages)

        # Extract concepts and analyze
        logger.info("Extracting concepts with quality gating...")
        extractor = LocalEntityExtractor()
        extraction_analysis = extract_and_analyze(passages, extractor)

        # Print report
        print_report(quality_analysis, extraction_analysis, len(passages))

        return 0

    finally:
        conn.close()


if __name__ == '__main__':
    sys.exit(main())
