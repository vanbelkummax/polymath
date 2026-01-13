#!/usr/bin/env python3
"""
Phase 5: Rank and Diversify Bridges using MMR
Select Top 20 bridges with diversity across buckets.
"""

import json
import os
import sys
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, '/home/user/polymath-repo')

OUTPUT_DIR = os.environ.get('OUTPUT_DIR', '/home/user/work/polymax/reports/bridge_mine_v3_20260112_2126')


def compute_mmr_selection(bridges: list, k: int = 20, lambda_param: float = 0.7) -> list:
    """
    Maximal Marginal Relevance selection.
    Balances relevance (evidence strength) with diversity (bucket coverage).
    """
    if len(bridges) <= k:
        return bridges

    selected = []
    candidates = bridges.copy()

    # First, select highest scoring bridge
    candidates.sort(key=lambda x: x['score'], reverse=True)
    selected.append(candidates.pop(0))

    # Then use MMR
    while len(selected) < k and candidates:
        best_mmr = -float('inf')
        best_idx = 0

        for i, cand in enumerate(candidates):
            # Relevance component
            relevance = cand['score']

            # Diversity component - penalty for same bucket
            max_sim = 0
            for sel in selected:
                if sel['bucket'] == cand['bucket']:
                    # Same bucket = high similarity
                    max_sim = max(max_sim, 0.8)
                else:
                    max_sim = max(max_sim, 0.2)

            # MMR score
            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim

            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = i

        selected.append(candidates.pop(best_idx))

    return selected


def main():
    # Load validated evidence
    validated_file = Path(OUTPUT_DIR) / "EVIDENCE" / "VALIDATED_EVIDENCE.jsonl"
    if not validated_file.exists():
        print(f"ERROR: {validated_file} not found. Run Phase 4 first.")
        return

    # Load bridge info
    aliases_file = Path(OUTPUT_DIR) / "INTERMEDIATE" / "ALIASES.jsonl"
    bridge_info = {}
    with open(aliases_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            bridge_info[data['bridge_id']] = data

    # Aggregate evidence stats
    stats = defaultdict(lambda: {
        'supports': 0,
        'tangential': 0,
        'garbage': 0,
        'docs': set(),
        'evidence_spans': []
    })

    with open(validated_file, 'r') as f:
        for line in f:
            record = json.loads(line)
            bridge_id = record['bridge_id']
            classification = record.get('classification', 'GARBAGE')

            stats[bridge_id][classification.lower()] += 1
            if classification == 'SUPPORTS':
                if record.get('doc_id'):
                    stats[bridge_id]['docs'].add(record['doc_id'])
                stats[bridge_id]['evidence_spans'].append(record)

    # Filter passing bridges
    passing = []
    for bridge_id, s in stats.items():
        if s['supports'] >= 2 and len(s['docs']) >= 2:
            info = bridge_info.get(bridge_id, {})
            passing.append({
                'bridge_id': bridge_id,
                'bucket': info.get('bucket', 'unknown'),
                'source_concept': info.get('source_concept', ''),
                'target_concept': info.get('target_concept', ''),
                'bucket_description': info.get('bucket_description', ''),
                'supports_count': s['supports'],
                'distinct_docs': len(s['docs']),
                'tangential_count': s['tangential'],
                'garbage_count': s['garbage'],
                # Composite score: supports * sqrt(docs) for diminishing returns on docs
                'score': s['supports'] * (len(s['docs']) ** 0.5),
                'evidence_spans': s['evidence_spans'][:5]  # Keep top 5 for writing
            })

    print(f"Bridges passing filter: {len(passing)}")

    if not passing:
        print("ERROR: No bridges passed the filter!")
        return

    # Rank with MMR
    top20 = compute_mmr_selection(passing, k=20, lambda_param=0.7)
    print(f"Selected top {len(top20)} with MMR")

    # Save to JSON
    top20_file = Path(OUTPUT_DIR) / "INTERMEDIATE" / "TOP20.json"
    with open(top20_file, 'w') as f:
        json.dump(top20, f, indent=2, default=str)

    # Save to CSV
    csv_file = Path(OUTPUT_DIR) / "BRIDGE_TABLE.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'rank', 'bridge_id', 'source_concept', 'target_concept', 'bucket',
            'supports', 'distinct_docs', 'score'
        ])
        writer.writeheader()
        for i, b in enumerate(top20):
            writer.writerow({
                'rank': i + 1,
                'bridge_id': b['bridge_id'],
                'source_concept': b['source_concept'],
                'target_concept': b['target_concept'],
                'bucket': b['bucket'],
                'supports': b['supports_count'],
                'distinct_docs': b['distinct_docs'],
                'score': round(b['score'], 2)
            })

    print(f"\n✓ Ranking complete")
    print(f"  TOP20.json: {top20_file}")
    print(f"  BRIDGE_TABLE.csv: {csv_file}")

    # Print summary
    print(f"\nTop 20 Bridges:")
    print("-" * 80)
    bucket_counts = defaultdict(int)
    for i, b in enumerate(top20):
        print(f"{i+1:2}. [{b['bucket'][:8]}] {b['source_concept']} → {b['target_concept']} "
              f"({b['supports_count']} SUPPORTS, {b['distinct_docs']} docs)")
        bucket_counts[b['bucket']] += 1

    print(f"\nBucket diversity:")
    for bucket, count in sorted(bucket_counts.items()):
        print(f"  {bucket}: {count}")


if __name__ == "__main__":
    main()
