#!/usr/bin/env python3
"""
Step 2: Overlap Scoring for Overlap Atlas

Computes:
- Field-field co-occurrence matrix with PMI scores
- Bridge concepts via betweenness centrality
- Rare but strong cross-field concept pairs

Usage:
    python scripts/overlap_atlas/score_overlaps.py
    python scripts/overlap_atlas/score_overlaps.py --top-fields 30
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import math

import pandas as pd
import numpy as np
import networkx as nx

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
INPUT_DIR = Path("/home/user/polymath-repo/var/overlap_atlas")
OUTPUT_DIR = INPUT_DIR
MIN_COOCCUR = 5  # Minimum co-occurrences for significance
SMOOTHING = 0.5  # Laplace smoothing for PMI


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load extracted data from Step 1."""
    concepts_path = INPUT_DIR / "passage_concepts.parquet"
    summary_path = INPUT_DIR / "passage_summary.parquet"

    logger.info(f"Loading data from {INPUT_DIR}")

    concepts = pd.read_parquet(concepts_path)
    summary = pd.read_parquet(summary_path)

    logger.info(f"Loaded {len(concepts):,} concept associations")
    logger.info(f"Loaded {len(summary):,} passage summaries")

    return concepts, summary


def compute_field_cooccurrence(summary: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """Compute field-field co-occurrence matrix."""
    logger.info("Computing field co-occurrence matrix...")

    # Explode field labels to get one row per passage-field
    exploded = summary[['passage_id', 'field_labels']].explode('field_labels')
    exploded = exploded[exploded['field_labels'].notna()]
    exploded.columns = ['passage_id', 'field']

    # Get top N fields by frequency
    field_counts = exploded['field'].value_counts()
    top_fields = field_counts.head(top_n).index.tolist()
    logger.info(f"Top {top_n} fields selected (max count: {field_counts.iloc[0]:,})")

    # Filter to top fields
    exploded = exploded[exploded['field'].isin(top_fields)]

    # Self-join to get all field pairs per passage
    pairs = exploded.merge(exploded, on='passage_id', suffixes=('_a', '_b'))
    pairs = pairs[pairs['field_a'] < pairs['field_b']]  # Avoid duplicates

    # Count co-occurrences
    cooccur = pairs.groupby(['field_a', 'field_b']).size().reset_index(name='count')

    logger.info(f"Found {len(cooccur):,} field pairs with co-occurrences")

    return cooccur, field_counts[top_fields].to_dict()


def compute_pmi(cooccur: pd.DataFrame, field_counts: Dict[str, int],
                total_passages: int) -> pd.DataFrame:
    """Compute Pointwise Mutual Information for field pairs."""
    logger.info("Computing PMI scores...")

    results = []

    for _, row in cooccur.iterrows():
        field_a, field_b = row['field_a'], row['field_b']
        count_ab = row['count']
        count_a = field_counts.get(field_a, 0)
        count_b = field_counts.get(field_b, 0)

        if count_a == 0 or count_b == 0:
            continue

        # Probabilities with Laplace smoothing
        p_ab = (count_ab + SMOOTHING) / (total_passages + SMOOTHING * len(field_counts))
        p_a = (count_a + SMOOTHING) / (total_passages + SMOOTHING * len(field_counts))
        p_b = (count_b + SMOOTHING) / (total_passages + SMOOTHING * len(field_counts))

        # PMI = log2(P(a,b) / (P(a) * P(b)))
        pmi = math.log2(p_ab / (p_a * p_b)) if p_a * p_b > 0 else 0

        # Normalized PMI (NPMI) = PMI / -log2(P(a,b))
        npmi = pmi / (-math.log2(p_ab)) if p_ab > 0 and p_ab < 1 else 0

        # Lift = P(a,b) / (P(a) * P(b))
        lift = p_ab / (p_a * p_b) if p_a * p_b > 0 else 0

        results.append({
            'field_a': field_a,
            'field_b': field_b,
            'count': count_ab,
            'count_a': count_a,
            'count_b': count_b,
            'pmi': pmi,
            'npmi': npmi,
            'lift': lift,
            'p_ab': p_ab,
            'p_a': p_a,
            'p_b': p_b
        })

    df = pd.DataFrame(results)

    # Filter for significance
    df = df[df['count'] >= MIN_COOCCUR]

    # Sort by PMI (most interesting overlaps)
    df = df.sort_values('pmi', ascending=False)

    logger.info(f"Computed PMI for {len(df):,} significant field pairs")

    return df


def build_concept_bridge_graph(concepts: pd.DataFrame, summary: pd.DataFrame) -> nx.Graph:
    """Build bipartite graph of Fields <-> Concepts for bridge detection."""
    logger.info("Building field-concept bipartite graph...")

    # Get field labels per passage
    passage_fields = {}
    for _, row in summary.iterrows():
        labels = row['field_labels']
        if labels is not None and len(labels) > 0:
            passage_fields[row['passage_id']] = set(labels)

    # Build edges: (field, concept) weighted by co-occurrence * confidence
    edge_weights = defaultdict(float)
    concept_field_counts = defaultdict(lambda: defaultdict(int))

    for _, row in concepts.iterrows():
        passage_id = row['passage_id']
        concept = row['concept_name']
        confidence = row['confidence'] or 0.8

        fields = passage_fields.get(passage_id, set())
        for field in fields:
            edge_weights[(field, concept)] += confidence
            concept_field_counts[concept][field] += 1

    # Create graph
    G = nx.Graph()

    # Add edges with weight threshold
    min_weight = 2.0
    for (field, concept), weight in edge_weights.items():
        if weight >= min_weight:
            G.add_edge(f"FIELD:{field}", f"CONCEPT:{concept}", weight=weight)

    logger.info(f"Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    return G, concept_field_counts


def find_bridge_concepts(G: nx.Graph, concept_field_counts: Dict,
                         top_n: int = 100) -> pd.DataFrame:
    """Find concepts that bridge distant fields using betweenness centrality."""
    logger.info("Computing betweenness centrality for bridge detection...")

    # Get concept nodes only
    concept_nodes = [n for n in G.nodes() if n.startswith("CONCEPT:")]

    if len(concept_nodes) == 0:
        logger.warning("No concept nodes found in graph")
        return pd.DataFrame()

    # Compute betweenness centrality (only for concept nodes)
    # This measures how often a concept lies on shortest paths between fields
    betweenness = nx.betweenness_centrality(G, k=min(500, len(G.nodes())))

    results = []
    for node in concept_nodes:
        concept = node.replace("CONCEPT:", "")
        bc = betweenness.get(node, 0)

        # Get connected fields
        neighbors = list(G.neighbors(node))
        connected_fields = [n.replace("FIELD:", "") for n in neighbors if n.startswith("FIELD:")]

        # Degree (number of connected fields)
        degree = len(connected_fields)

        # Field diversity (unique fields)
        field_counts = concept_field_counts.get(concept, {})

        results.append({
            'concept': concept,
            'betweenness': bc,
            'degree': degree,
            'connected_fields': connected_fields[:10],  # Top 10 for storage
            'field_count': len(field_counts),
            'total_occurrences': sum(field_counts.values())
        })

    df = pd.DataFrame(results)
    df = df.sort_values('betweenness', ascending=False)

    # Filter: must connect at least 2 fields
    df = df[df['degree'] >= 2]

    logger.info(f"Found {len(df):,} bridge concepts")

    return df.head(top_n)


def find_rare_pairs(concepts: pd.DataFrame, summary: pd.DataFrame,
                    top_n: int = 200) -> pd.DataFrame:
    """Find rare but strong cross-field concept pairs."""
    logger.info("Finding rare cross-field concept pairs...")

    # Get field labels per passage
    passage_fields = {}
    for _, row in summary.iterrows():
        labels = row['field_labels']
        if labels is not None and len(labels) > 0:
            passage_fields[row['passage_id']] = set(labels)

    # Group concepts by passage
    passage_concepts = concepts.groupby('passage_id').agg({
        'concept_name': list,
        'confidence': list
    }).reset_index()

    # Find concept pairs that appear in passages with different primary fields
    pair_data = defaultdict(lambda: {'count': 0, 'confidence_sum': 0, 'field_pairs': set()})

    for _, row in passage_concepts.iterrows():
        passage_id = row['passage_id']
        concept_list = row['concept_name']
        confidence_list = row['confidence']
        fields = passage_fields.get(passage_id, set())

        if len(fields) < 2 or len(concept_list) < 2:
            continue

        # Create concept pairs within this passage
        for i in range(len(concept_list)):
            for j in range(i + 1, len(concept_list)):
                c1, c2 = sorted([concept_list[i], concept_list[j]])
                conf = (confidence_list[i] + confidence_list[j]) / 2

                key = (c1, c2)
                pair_data[key]['count'] += 1
                pair_data[key]['confidence_sum'] += conf
                pair_data[key]['field_pairs'].update(
                    [(f1, f2) for f1 in fields for f2 in fields if f1 < f2]
                )

    # Convert to DataFrame
    results = []
    for (c1, c2), data in pair_data.items():
        if data['count'] < 3:  # Minimum occurrences
            continue

        avg_conf = data['confidence_sum'] / data['count']
        n_field_pairs = len(data['field_pairs'])

        if n_field_pairs < 2:  # Must appear in multiple field pairs
            continue

        # Score: rarity (inverse frequency) * confidence * field diversity
        # Lower count = more rare = higher score
        rarity_score = 1 / (1 + math.log(data['count']))
        diversity_score = math.log(1 + n_field_pairs)
        score = rarity_score * avg_conf * diversity_score

        results.append({
            'concept_a': c1,
            'concept_b': c2,
            'count': data['count'],
            'avg_confidence': avg_conf,
            'n_field_pairs': n_field_pairs,
            'field_pairs': list(data['field_pairs'])[:5],  # Top 5 for storage
            'score': score
        })

    df = pd.DataFrame(results)
    df = df.sort_values('score', ascending=False)

    logger.info(f"Found {len(df):,} rare cross-field concept pairs")

    return df.head(top_n)


def save_outputs(field_overlap: pd.DataFrame, bridge_concepts: pd.DataFrame,
                 rare_pairs: pd.DataFrame, stats: Dict):
    """Save all outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save field overlap
    overlap_path = OUTPUT_DIR / "field_field_overlap.csv"
    field_overlap.to_csv(overlap_path, index=False)
    logger.info(f"Saved {len(field_overlap)} field overlaps to {overlap_path}")

    # Save bridge concepts
    bridge_path = OUTPUT_DIR / "bridge_concepts.csv"
    # Convert list columns to strings for CSV
    bridge_concepts['connected_fields'] = bridge_concepts['connected_fields'].apply(
        lambda x: ','.join(x) if isinstance(x, list) else ''
    )
    bridge_concepts.to_csv(bridge_path, index=False)
    logger.info(f"Saved {len(bridge_concepts)} bridge concepts to {bridge_path}")

    # Save rare pairs
    rare_path = OUTPUT_DIR / "rare_pairs.csv"
    rare_pairs['field_pairs'] = rare_pairs['field_pairs'].apply(
        lambda x: ';'.join([f"{a}-{b}" for a, b in x]) if isinstance(x, list) else ''
    )
    rare_pairs.to_csv(rare_path, index=False)
    logger.info(f"Saved {len(rare_pairs)} rare pairs to {rare_path}")

    # Save stats
    stats_path = OUTPUT_DIR / "overlap_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved stats to {stats_path}")

    # Print summary
    print("\n" + "="*60)
    print("OVERLAP SCORING SUMMARY")
    print("="*60)
    print(f"Field-field pairs with PMI: {len(field_overlap):,}")
    print(f"Bridge concepts identified: {len(bridge_concepts):,}")
    print(f"Rare cross-field pairs: {len(rare_pairs):,}")
    print(f"\nTop 5 field overlaps by PMI:")
    for _, row in field_overlap.head(5).iterrows():
        print(f"  {row['field_a']} <-> {row['field_b']}: PMI={row['pmi']:.3f}, count={row['count']}")
    print(f"\nTop 5 bridge concepts:")
    for _, row in bridge_concepts.head(5).iterrows():
        print(f"  {row['concept']}: betweenness={row['betweenness']:.4f}, fields={row['degree']}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Compute overlap scores for Overlap Atlas")
    parser.add_argument("--top-fields", type=int, default=50,
                        help="Number of top fields to analyze (default: 50)")
    parser.add_argument("--top-bridges", type=int, default=100,
                        help="Number of bridge concepts to output (default: 100)")
    parser.add_argument("--top-rare", type=int, default=200,
                        help="Number of rare pairs to output (default: 200)")
    args = parser.parse_args()

    logger.info("Starting overlap scoring...")

    # Load data
    concepts, summary = load_data()
    total_passages = len(summary)

    # Step 1: Field-field co-occurrence and PMI
    cooccur, field_counts = compute_field_cooccurrence(summary, args.top_fields)
    field_overlap = compute_pmi(cooccur, field_counts, total_passages)

    # Step 2: Bridge concept detection
    G, concept_field_counts = build_concept_bridge_graph(concepts, summary)
    bridge_concepts = find_bridge_concepts(G, concept_field_counts, args.top_bridges)

    # Step 3: Rare cross-field pairs
    rare_pairs = find_rare_pairs(concepts, summary, args.top_rare)

    # Compute stats
    stats = {
        'total_passages': total_passages,
        'total_concepts': concepts['concept_name'].nunique(),
        'total_concept_associations': len(concepts),
        'top_fields_analyzed': args.top_fields,
        'significant_field_pairs': len(field_overlap),
        'bridge_concepts_found': len(bridge_concepts),
        'rare_pairs_found': len(rare_pairs),
        'graph_nodes': G.number_of_nodes(),
        'graph_edges': G.number_of_edges(),
        'top_pmi_pairs': field_overlap.head(10)[['field_a', 'field_b', 'pmi', 'count']].to_dict('records'),
        'top_bridges': bridge_concepts.head(10)[['concept', 'betweenness', 'degree']].to_dict('records')
    }

    # Save outputs
    save_outputs(field_overlap, bridge_concepts, rare_pairs, stats)

    logger.info("Overlap scoring complete!")


if __name__ == "__main__":
    main()
