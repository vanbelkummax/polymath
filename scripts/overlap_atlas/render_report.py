#!/usr/bin/env python3
"""
Step 6: Report Generation for Overlap Atlas

Generates a comprehensive markdown report summarizing the Overlap Atlas findings.

Usage:
    python scripts/overlap_atlas/render_report.py
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
INPUT_DIR = Path("/home/user/polymath-repo/var/overlap_atlas")
OUTPUT_DIR = Path("/home/user/polymath-repo/docs/runlogs")
FIGURES_DIR = OUTPUT_DIR / "figures"


def load_all_data():
    """Load all generated data."""
    field_overlap = pd.read_csv(INPUT_DIR / "field_field_overlap.csv")
    bridge_concepts = pd.read_csv(INPUT_DIR / "bridge_concepts.csv")
    rare_pairs = pd.read_csv(INPUT_DIR / "rare_pairs.csv")

    with open(INPUT_DIR / "hypotheses.json") as f:
        hypotheses = json.load(f)

    with open(INPUT_DIR / "extraction_stats.json") as f:
        extraction_stats = json.load(f)

    with open(INPUT_DIR / "overlap_stats.json") as f:
        overlap_stats = json.load(f)

    return {
        'field_overlap': field_overlap,
        'bridge_concepts': bridge_concepts,
        'rare_pairs': rare_pairs,
        'hypotheses': hypotheses,
        'extraction_stats': extraction_stats,
        'overlap_stats': overlap_stats
    }


def generate_report(data: dict) -> str:
    """Generate markdown report."""
    today = datetime.now().strftime("%Y-%m-%d")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []

    # Title
    lines.append(f"# Polymath Overlap Atlas Report")
    lines.append(f"")
    lines.append(f"**Generated:** {timestamp}")
    lines.append(f"")

    # Executive Summary
    lines.append(f"## Executive Summary")
    lines.append(f"")
    lines.append(f"This report presents the **Polymath Overlap Atlas**, a systematic analysis of cross-domain ")
    lines.append(f"connections within the Polymath knowledge base. By analyzing {data['extraction_stats']['total_concept_associations']:,} ")
    lines.append(f"concept-passage associations across {data['extraction_stats']['unique_passages']:,} passages, ")
    lines.append(f"we identify unexpected field overlaps, bridge concepts that connect distant domains, ")
    lines.append(f"and generate evidence-bound hypotheses for potential research directions.")
    lines.append(f"")
    lines.append(f"### Key Findings")
    lines.append(f"")
    lines.append(f"- **{len(data['field_overlap'])} significant field-field overlaps** discovered (PMI > 0)")
    lines.append(f"- **{len(data['bridge_concepts'])} bridge concepts** identified via betweenness centrality")
    lines.append(f"- **{data['hypotheses']['total_hypotheses']} evidence-bound hypotheses** generated")
    lines.append(f"- **Top bridge concepts**: machine_learning, artificial_intelligence, gene_expression, deep_learning, cancer")
    lines.append(f"")

    # Dataset Summary
    lines.append(f"## Dataset Summary")
    lines.append(f"")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total Passages | {data['extraction_stats']['unique_passages']:,} |")
    lines.append(f"| Total Concept Associations | {data['extraction_stats']['total_concept_associations']:,} |")
    lines.append(f"| Unique Concepts | {data['extraction_stats']['unique_concepts']:,} |")
    lines.append(f"| Confidence Threshold | >= 0.7 |")
    lines.append(f"| Mean Confidence | {data['extraction_stats']['confidence_stats']['mean']:.3f} |")
    lines.append(f"| Extractor | gemini_batch_v1 |")
    lines.append(f"")

    # Concept Type Distribution
    lines.append(f"### Concept Type Distribution")
    lines.append(f"")
    lines.append(f"| Type | Count | Percentage |")
    lines.append(f"|------|-------|------------|")
    total = sum(data['extraction_stats']['concept_type_distribution'].values())
    for ctype, count in sorted(data['extraction_stats']['concept_type_distribution'].items(),
                                key=lambda x: x[1], reverse=True)[:10]:
        pct = count / total * 100
        lines.append(f"| {ctype} | {count:,} | {pct:.1f}% |")
    lines.append(f"")

    # Field Overlap Analysis
    lines.append(f"## Field Overlap Analysis")
    lines.append(f"")
    lines.append(f"We computed Pointwise Mutual Information (PMI) for all field pairs that co-occur ")
    lines.append(f"in at least 5 passages. PMI measures how much more often two fields appear together ")
    lines.append(f"than would be expected by chance.")
    lines.append(f"")
    lines.append(f"![Field Overlap Heatmap](figures/field_overlap_heatmap.png)")
    lines.append(f"")

    lines.append(f"### Top 20 Field Overlaps by PMI")
    lines.append(f"")
    lines.append(f"| Field A | Field B | PMI | Co-occurrences |")
    lines.append(f"|---------|---------|-----|----------------|")
    for _, row in data['field_overlap'].head(20).iterrows():
        lines.append(f"| {row['field_a'].replace('_', ' ')} | {row['field_b'].replace('_', ' ')} | {row['pmi']:.3f} | {row['count']:,} |")
    lines.append(f"")

    # Bridge Concepts
    lines.append(f"## Bridge Concepts")
    lines.append(f"")
    lines.append(f"Bridge concepts are ideas that connect multiple distant fields, potentially enabling ")
    lines.append(f"cross-domain knowledge transfer. We identified these using betweenness centrality ")
    lines.append(f"in the field-concept bipartite graph.")
    lines.append(f"")
    lines.append(f"![Bridge Concept Network](figures/bridge_concept_network.png)")
    lines.append(f"")

    lines.append(f"### Top 20 Bridge Concepts")
    lines.append(f"")
    lines.append(f"| Concept | Betweenness | Fields Connected |")
    lines.append(f"|---------|-------------|------------------|")
    for _, row in data['bridge_concepts'].head(20).iterrows():
        lines.append(f"| {row['concept'].replace('_', ' ')} | {row['betweenness']:.5f} | {row['degree']} |")
    lines.append(f"")

    # Novel Hypotheses
    lines.append(f"## Novel Hypotheses")
    lines.append(f"")
    lines.append(f"We generated {data['hypotheses']['total_hypotheses']} evidence-bound hypotheses from ")
    lines.append(f"three sources: field overlaps, bridge concepts, and rare cross-field pairs. Each ")
    lines.append(f"hypothesis is grounded in actual passage evidence from the corpus.")
    lines.append(f"")
    lines.append(f"![Hypothesis Cards](figures/hypothesis_cards.png)")
    lines.append(f"")

    lines.append(f"### Hypothesis Details")
    lines.append(f"")
    for i, hyp in enumerate(data['hypotheses']['hypotheses'][:10], 1):
        lines.append(f"#### {i}. {hyp['title']}")
        lines.append(f"")
        lines.append(f"**Type:** {hyp['type'].replace('_', ' ').title()}")
        lines.append(f"")
        lines.append(f"> {hyp['hypothesis_text']}")
        lines.append(f"")
        lines.append(f"**Score:** {hyp['score']:.3f} | **Evidence passages:** {len(hyp.get('evidence_passages', []))}")
        lines.append(f"")
        if hyp.get('fields'):
            lines.append(f"**Fields:** {', '.join([f.replace('_', ' ') for f in hyp['fields'][:4]])}")
            lines.append(f"")
        if hyp.get('bridge_concepts'):
            lines.append(f"**Bridge concepts:** {', '.join([c.replace('_', ' ') for c in hyp['bridge_concepts'][:3]])}")
            lines.append(f"")

        # Sample evidence
        if hyp.get('evidence_passages'):
            lines.append(f"**Sample evidence:**")
            lines.append(f"")
            for ev in hyp['evidence_passages'][:2]:
                evidence_text = ev.get('evidence') or ev.get('source_text') or ev.get('passage_preview', '')
                if evidence_text:
                    evidence_text = evidence_text[:200] + ('...' if len(evidence_text) > 200 else '')
                    lines.append(f"- `{ev['passage_id'][:8]}...`: \"{evidence_text}\"")
            lines.append(f"")
        lines.append(f"---")
        lines.append(f"")

    # Retrieval Synthesis Demo
    lines.append(f"## Retrieval Synthesis Demo")
    lines.append(f"")
    lines.append(f"The new `atlas_search()` method in `lib/hybrid_search_v2.py` provides tri-modal ")
    lines.append(f"search with an explainability trace. Here's an example:")
    lines.append(f"")
    lines.append(f"```python")
    lines.append(f"from lib.hybrid_search_v2 import HybridSearcherV2")
    lines.append(f"")
    lines.append(f"hs = HybridSearcherV2()")
    lines.append(f"result = hs.atlas_search('spatial transcriptomics optimal transport', n=10)")
    lines.append(f"")
    lines.append(f"# Results with explainability")
    lines.append(f"print(result['explain'])")
    lines.append(f"# {{")
    lines.append(f"#   'sql': {{'matched_concepts': ['spatial', 'transcriptomics', 'optimal', 'transport'], 'count': 45}},")
    lines.append(f"#   'vector': {{'top_neighbors': [...], 'scores': [...], 'count': 10}},")
    lines.append(f"#   'graph': {{'expanded_concepts': ['gene_expression', 'single_cell', ...], 'paths_found': 12}}")
    lines.append(f"# }}")
    lines.append(f"```")
    lines.append(f"")
    lines.append(f"The explain trace shows which retrieval channel contributed what, enabling ")
    lines.append(f"transparent and debuggable polymathic search.")
    lines.append(f"")

    # Methodology
    lines.append(f"## Methodology")
    lines.append(f"")
    lines.append(f"### Data Extraction (Step 1)")
    lines.append(f"- Extracted concept-passage associations from `passage_concepts` table")
    lines.append(f"- Filtered to confidence >= 0.7 and extractor_version = 'gemini_batch_v1'")
    lines.append(f"- Assigned field labels using domain/field concept types")
    lines.append(f"")
    lines.append(f"### Overlap Scoring (Step 2)")
    lines.append(f"- **PMI**: log2(P(a,b) / (P(a) * P(b))) with Laplace smoothing")
    lines.append(f"- **Bridge detection**: Betweenness centrality in field-concept bipartite graph")
    lines.append(f"- Minimum co-occurrence threshold: 5 passages")
    lines.append(f"")
    lines.append(f"### Hypothesis Generation (Step 3)")
    lines.append(f"- Three sources: field overlaps, bridge concepts, rare pairs")
    lines.append(f"- Each hypothesis requires >= 2 evidence passages")
    lines.append(f"- Score = overlap_strength * mean_confidence * evidence_coverage")
    lines.append(f"")

    # Appendix
    lines.append(f"## Appendix")
    lines.append(f"")
    lines.append(f"### All Hypotheses")
    lines.append(f"")
    lines.append(f"| # | Type | Title | Score |")
    lines.append(f"|---|------|-------|-------|")
    for i, hyp in enumerate(data['hypotheses']['hypotheses'], 1):
        title = hyp['title'][:50] + ('...' if len(hyp['title']) > 50 else '')
        lines.append(f"| {i} | {hyp['type']} | {title} | {hyp['score']:.3f} |")
    lines.append(f"")

    lines.append(f"### Top Fields by Frequency")
    lines.append(f"")
    lines.append(f"| Field | Count |")
    lines.append(f"|-------|-------|")
    for field, count in list(data['extraction_stats']['top_fields'].items())[:20]:
        lines.append(f"| {field.replace('_', ' ')} | {count:,} |")
    lines.append(f"")

    lines.append(f"---")
    lines.append(f"")
    lines.append(f"*Report generated by Polymath Overlap Atlas pipeline*")
    lines.append(f"")
    lines.append(f"*Scripts: `scripts/overlap_atlas/`*")

    return '\n'.join(lines)


def main():
    logger.info("Loading data...")
    data = load_all_data()

    logger.info("Generating report...")
    report = generate_report(data)

    # Save report
    today = datetime.now().strftime("%Y%m%d")
    output_path = OUTPUT_DIR / f"polymath_overlap_atlas_{today}.md"

    with open(output_path, 'w') as f:
        f.write(report)

    logger.info(f"Report saved to {output_path}")

    # Also save as latest
    latest_path = OUTPUT_DIR / "polymath_overlap_atlas_latest.md"
    with open(latest_path, 'w') as f:
        f.write(report)

    print(f"\n{'='*60}")
    print(f"REPORT GENERATED")
    print(f"{'='*60}")
    print(f"Main report: {output_path}")
    print(f"Latest link: {latest_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
