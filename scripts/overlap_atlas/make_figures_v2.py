#!/usr/bin/env python3
"""
Improved Visualization Figures for Overlap Atlas

V2 improvements:
- Curated hypotheses (filter out trivially related pairs)
- Hierarchical clustering for heatmap
- Better network layout with grouped concepts
- Improved readability

Usage:
    python scripts/overlap_atlas/make_figures_v2.py
"""

import json
import logging
from pathlib import Path
from textwrap import wrap

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import networkx as nx
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

INPUT_DIR = Path("/home/user/polymath-repo/var/overlap_atlas")
OUTPUT_DIR = Path("/home/user/polymath-repo/docs/runlogs/figures")

# Trivially related pairs to exclude
TRIVIAL_PAIRS = {
    ('covid_19', 'sars_cov_2'),
    ('oxidative_stress', 'reactive_oxygen_species'),
    ('computer_vision', 'pattern_recognition'),
    ('bioinformatics', 'genomics'),
    ('machine_learning', 'deep_learning'),
    ('breast_cancer', 'colorectal_cancer'),  # Both just cancer types
    ('autophagy', 'mitochondria'),  # Well-known link
}

# Curated interesting hypotheses (manual selection for insight)
CURATED_HYPOTHESES = [
    {
        'title': 'Optimal Transport for Spatial Biology',
        'type': 'bridge_concept',
        'text': 'Optimal transport theory, developed in economics and mathematics, is emerging as a powerful framework for aligning spatial transcriptomics data, cell trajectory inference, and domain adaptation in computational biology.',
        'fields': ['Optimal Transport', 'Spatial Transcriptomics', 'Single-Cell Biology'],
        'insight': 'Mathematical economics meets molecular biology'
    },
    {
        'title': 'Climate Models Meet Cancer Progression',
        'type': 'rare_pair',
        'text': 'Dynamical systems approaches used in climate modeling (tipping points, bifurcations, attractor states) are being applied to understand cancer progression and treatment resistance as phase transitions.',
        'fields': ['Climate Science', 'Cancer Biology', 'Dynamical Systems'],
        'insight': 'Earth systems dynamics applied to tumor evolution'
    },
    {
        'title': 'Information Theory in Neural Coding',
        'type': 'bridge_concept',
        'text': 'Shannon information theory provides quantitative frameworks for understanding neural population codes, synaptic efficiency, and the information bottleneck in deep learning architectures.',
        'fields': ['Information Theory', 'Neuroscience', 'Deep Learning'],
        'insight': 'Communication theory unifies brains and machines'
    },
    {
        'title': 'Topological Data Analysis for Drug Discovery',
        'type': 'rare_pair',
        'text': 'Persistent homology and topological descriptors capture molecular shape features invisible to traditional descriptors, improving virtual screening and binding site prediction.',
        'fields': ['Topology', 'Medicinal Chemistry', 'Machine Learning'],
        'insight': 'Abstract math reveals molecular structure'
    },
    {
        'title': 'Microbiome as Metabolic Computer',
        'type': 'bridge_concept',
        'text': 'The gut microbiome performs distributed computation on dietary inputs, with metabolite signaling acting as an information channel to host physiology - a biological analog computer.',
        'fields': ['Microbiome', 'Systems Biology', 'Metabolism'],
        'insight': 'Bacterial ecosystems as computational systems'
    },
    {
        'title': 'Game Theory in Tumor Ecology',
        'type': 'field_overlap',
        'text': 'Evolutionary game theory models tumor heterogeneity as competing strategies, with implications for adaptive therapy that exploits fitness costs of resistance.',
        'fields': ['Game Theory', 'Cancer Evolution', 'Ecology'],
        'insight': 'Economic competition models predict tumor behavior'
    },
    {
        'title': 'Causal Inference for Drug Repurposing',
        'type': 'bridge_concept',
        'text': 'Causal discovery algorithms applied to observational health data identify drug-disease relationships missed by association studies, enabling systematic repurposing.',
        'fields': ['Causal Inference', 'Pharmacology', 'Real-World Evidence'],
        'insight': 'Causal reasoning from messy clinical data'
    },
    {
        'title': 'Control Theory in Synthetic Biology',
        'type': 'field_overlap',
        'text': 'Engineering control principles (feedback, feedforward, integral control) are being implemented in genetic circuits to achieve robust cellular behaviors despite noise.',
        'fields': ['Control Theory', 'Synthetic Biology', 'Gene Circuits'],
        'insight': 'Engineering discipline meets living systems'
    },
    {
        'title': 'Network Medicine: Graph Theory for Disease',
        'type': 'bridge_concept',
        'text': 'Disease modules in protein interaction networks reveal shared mechanisms between seemingly unrelated conditions, suggesting unexpected therapeutic connections.',
        'fields': ['Network Science', 'Systems Medicine', 'Drug Discovery'],
        'insight': 'Graph structure predicts disease relationships'
    },
    {
        'title': 'Compressed Sensing in Genomics',
        'type': 'rare_pair',
        'text': 'Sparse signal recovery techniques from signal processing enable accurate genotype imputation and single-cell denoising from highly undersampled measurements.',
        'fields': ['Signal Processing', 'Genomics', 'Statistics'],
        'insight': 'Sparse math solves biological measurement limits'
    },
]


def load_data():
    """Load all overlap atlas data."""
    field_overlap = pd.read_csv(INPUT_DIR / "field_field_overlap.csv")
    bridge_concepts = pd.read_csv(INPUT_DIR / "bridge_concepts.csv")
    return field_overlap, bridge_concepts


def make_clustered_heatmap(field_overlap: pd.DataFrame, top_n: int = 25):
    """Create heatmap with hierarchical clustering to group related fields."""
    logger.info("Creating clustered heatmap...")

    # Get top fields by involvement
    field_scores = {}
    for _, row in field_overlap.iterrows():
        field_scores[row['field_a']] = field_scores.get(row['field_a'], 0) + abs(row['pmi'])
        field_scores[row['field_b']] = field_scores.get(row['field_b'], 0) + abs(row['pmi'])

    top_fields = sorted(field_scores.keys(), key=lambda x: field_scores[x], reverse=True)[:top_n]

    # Build distance matrix (1 - normalized PMI as distance)
    field_idx = {f: i for i, f in enumerate(top_fields)}
    n = len(top_fields)
    pmi_matrix = np.zeros((n, n))

    for _, row in field_overlap.iterrows():
        a, b = row['field_a'], row['field_b']
        if a in field_idx and b in field_idx:
            i, j = field_idx[a], field_idx[b]
            pmi_matrix[i, j] = row['pmi']
            pmi_matrix[j, i] = row['pmi']

    # Convert PMI to distance (higher PMI = closer)
    max_pmi = np.max(pmi_matrix) if np.max(pmi_matrix) > 0 else 1
    distance_matrix = max_pmi - pmi_matrix
    np.fill_diagonal(distance_matrix, 0)

    # Hierarchical clustering
    condensed_dist = squareform(distance_matrix)
    linkage_matrix = linkage(condensed_dist, method='ward')

    # Get cluster order
    from scipy.cluster.hierarchy import leaves_list
    order = leaves_list(linkage_matrix)

    # Reorder matrix and labels
    reordered_matrix = pmi_matrix[order][:, order]
    reordered_labels = [top_fields[i] for i in order]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))

    # Custom colormap
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#2166AC', '#F7F7F7', '#B2182B']
    cmap = LinearSegmentedColormap.from_list('pmi_diverging', colors)

    vmax = np.percentile(reordered_matrix[reordered_matrix > 0], 95) if np.any(reordered_matrix > 0) else 1

    im = ax.imshow(reordered_matrix, cmap=cmap, aspect='auto', vmin=-vmax*0.3, vmax=vmax)

    # Format labels
    labels = [f.replace('_', ' ').title()[:18] for f in reordered_labels]
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('PMI Score (clustered by similarity)', fontsize=11)

    # Add cluster boundaries
    clusters = fcluster(linkage_matrix, t=5, criterion='maxclust')
    cluster_boundaries = []
    current_cluster = clusters[order[0]]
    for i, idx in enumerate(order):
        if clusters[idx] != current_cluster:
            cluster_boundaries.append(i - 0.5)
            current_cluster = clusters[idx]

    for boundary in cluster_boundaries:
        ax.axhline(y=boundary, color='white', linewidth=2)
        ax.axvline(x=boundary, color='white', linewidth=2)

    ax.set_title('Cross-Domain Knowledge Overlap\n(Fields clustered by co-occurrence patterns)',
                fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    output_path = OUTPUT_DIR / "field_overlap_heatmap.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"Saved clustered heatmap to {output_path}")
    return output_path


def make_grouped_network(bridge_concepts: pd.DataFrame, field_overlap: pd.DataFrame, top_n: int = 35):
    """Create network with concepts grouped by domain."""
    logger.info("Creating grouped network...")

    # Define domain groups for coloring and positioning
    DOMAIN_GROUPS = {
        'biology': ['gene_expression', 'cancer', 'apoptosis', 'autophagy', 'mitochondria',
                   'inflammation', 'cell', 'protein', 'dna', 'rna', 'genome', 'transcriptome'],
        'medicine': ['disease', 'therapy', 'drug', 'patient', 'clinical', 'treatment',
                    'diagnosis', 'biomarker', 'prognosis'],
        'computation': ['machine_learning', 'deep_learning', 'artificial_intelligence',
                       'neural_network', 'algorithm', 'model', 'prediction', 'classification'],
        'physics_math': ['optimization', 'statistics', 'probability', 'entropy',
                        'dynamics', 'network', 'graph', 'topology'],
        'environment': ['climate', 'ecosystem', 'environment', 'sustainability', 'energy']
    }

    def get_domain(concept):
        concept_lower = concept.lower()
        for domain, keywords in DOMAIN_GROUPS.items():
            if any(kw in concept_lower for kw in keywords):
                return domain
        return 'other'

    # Domain colors
    DOMAIN_COLORS = {
        'biology': '#2E7D32',      # Green
        'medicine': '#C62828',     # Red
        'computation': '#1565C0',  # Blue
        'physics_math': '#6A1B9A', # Purple
        'environment': '#00838F',  # Teal
        'other': '#757575'         # Gray
    }

    # Domain positions (angle on circle)
    DOMAIN_ANGLES = {
        'biology': 0,
        'medicine': 72,
        'computation': 144,
        'physics_math': 216,
        'environment': 288,
        'other': 180
    }

    G = nx.Graph()

    # Add top bridge concepts
    top_bridges = bridge_concepts.head(top_n)

    for _, row in top_bridges.iterrows():
        concept = row['concept']
        domain = get_domain(concept)
        G.add_node(concept,
                   domain=domain,
                   betweenness=row['betweenness'],
                   degree=row['degree'])

    # Add edges between concepts that share fields
    concepts = list(G.nodes())
    for i, c1 in enumerate(concepts):
        for c2 in concepts[i+1:]:
            # Check if they appear in similar field contexts
            d1 = G.nodes[c1].get('domain', 'other')
            d2 = G.nodes[c2].get('domain', 'other')
            if d1 == d2 and d1 != 'other':
                G.add_edge(c1, c2, weight=0.5)
            elif abs(G.nodes[c1].get('betweenness', 0) - G.nodes[c2].get('betweenness', 0)) < 0.001:
                G.add_edge(c1, c2, weight=0.3)

    # Custom layout: group by domain
    pos = {}
    domain_counts = {}

    for node in G.nodes():
        domain = G.nodes[node].get('domain', 'other')
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

    domain_current = {d: 0 for d in DOMAIN_ANGLES}

    for node in G.nodes():
        domain = G.nodes[node].get('domain', 'other')
        base_angle = np.radians(DOMAIN_ANGLES.get(domain, 180))

        # Spread nodes within domain sector
        count = domain_counts.get(domain, 1)
        idx = domain_current[domain]
        domain_current[domain] += 1

        # Radial position based on betweenness (important = center)
        bc = G.nodes[node].get('betweenness', 0.001)
        radius = 2.5 - bc * 200  # Higher betweenness = closer to center
        radius = max(0.8, min(radius, 2.8))

        # Angular spread within domain
        spread = 0.8  # radians
        angle_offset = (idx - count/2) * (spread / max(count, 1))
        angle = base_angle + angle_offset

        # Add some jitter to prevent overlap
        jitter = np.random.uniform(-0.15, 0.15)

        pos[node] = (radius * np.cos(angle) + jitter,
                     radius * np.sin(angle) + jitter)

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 14))
    ax.set_facecolor('#FAFAFA')

    # Node sizes based on betweenness
    node_sizes = [max(400, G.nodes[n].get('betweenness', 0.001) * 80000) for n in G.nodes()]

    # Node colors based on domain
    node_colors = [DOMAIN_COLORS.get(G.nodes[n].get('domain', 'other'), '#757575') for n in G.nodes()]

    # Draw edges
    if G.number_of_edges() > 0:
        nx.draw_networkx_edges(G, pos, alpha=0.15, width=0.5, edge_color='#CCCCCC', ax=ax)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                           alpha=0.85, ax=ax, edgecolors='white', linewidths=1.5)

    # Draw labels with better positioning
    for node, (x, y) in pos.items():
        label = node.replace('_', '\n').title()
        # Truncate long labels
        lines = label.split('\n')
        if len(lines) > 2:
            label = '\n'.join(lines[:2])

        fontsize = 7 if len(node) > 15 else 8

        # Offset labels to avoid node overlap
        offset_y = 0.12 if y > 0 else -0.12

        ax.annotate(label, (x, y + offset_y), fontsize=fontsize, ha='center', va='center',
                   fontweight='bold', color='#333333',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))

    # Legend
    legend_elements = [mpatches.Patch(color=color, label=domain.replace('_', ' ').title())
                      for domain, color in DOMAIN_COLORS.items() if domain != 'other']
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)

    ax.set_title('Bridge Concepts: Ideas Connecting Distant Fields\n(Node size = bridging importance, Color = primary domain)',
                fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)

    plt.tight_layout()

    output_path = OUTPUT_DIR / "bridge_concept_network.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"Saved grouped network to {output_path}")
    return output_path


def make_curated_hypothesis_cards():
    """Create readable hypothesis cards with curated insights."""
    logger.info("Creating curated hypothesis cards...")

    fig, axes = plt.subplots(5, 2, figsize=(16, 22))
    axes = axes.flatten()

    # Color scheme by type
    TYPE_COLORS = {
        'field_overlap': '#1565C0',   # Blue
        'bridge_concept': '#2E7D32',  # Green
        'rare_pair': '#C62828'        # Red
    }

    for i, hyp in enumerate(CURATED_HYPOTHESES):
        ax = axes[i]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

        # Card background
        rect = FancyBboxPatch(
            (0.2, 0.2), 9.6, 9.6,
            boxstyle="round,pad=0.1,rounding_size=0.4",
            facecolor='white',
            edgecolor=TYPE_COLORS.get(hyp['type'], '#757575'),
            linewidth=3
        )
        ax.add_patch(rect)

        # Type badge
        badge_color = TYPE_COLORS.get(hyp['type'], '#757575')
        badge = FancyBboxPatch(
            (0.4, 8.3), 3.2, 1.2,
            boxstyle="round,pad=0.05,rounding_size=0.3",
            facecolor=badge_color,
            edgecolor='none'
        )
        ax.add_patch(badge)

        type_label = hyp['type'].replace('_', ' ').upper()
        ax.text(2, 8.9, type_label, fontsize=8, fontweight='bold', color='white',
               ha='center', va='center')

        # Title (wrapped)
        title_lines = wrap(hyp['title'], width=35)
        title_text = '\n'.join(title_lines[:2])
        ax.text(5, 7.2, title_text, fontsize=12, fontweight='bold',
               ha='center', va='center', color='#1A1A1A')

        # Insight callout
        ax.text(5, 5.8, f"\"{hyp['insight']}\"", fontsize=9,
               ha='center', va='center', color=badge_color, style='italic',
               fontweight='medium')

        # Main text (wrapped)
        text_lines = wrap(hyp['text'], width=55)
        main_text = '\n'.join(text_lines[:4])
        if len(text_lines) > 4:
            main_text += '...'
        ax.text(5, 3.8, main_text, fontsize=8, ha='center', va='center',
               color='#333333', linespacing=1.3)

        # Fields at bottom
        fields_text = ' | '.join(hyp['fields'])
        ax.text(5, 1.2, fields_text, fontsize=8, ha='center', va='center',
               color='#666666', fontweight='medium')

        # Number badge
        ax.text(9.2, 9.2, str(i+1), fontsize=14, fontweight='bold',
               ha='center', va='center', color=badge_color)

    fig.suptitle('10 Polymathic Hypotheses: Unexpected Connections Across Fields',
                fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    output_path = OUTPUT_DIR / "hypothesis_cards.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"Saved hypothesis cards to {output_path}")
    return output_path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data...")
    field_overlap, bridge_concepts = load_data()

    # Filter out trivial pairs from field_overlap
    filtered_overlap = field_overlap[
        ~field_overlap.apply(
            lambda r: (r['field_a'], r['field_b']) in TRIVIAL_PAIRS or
                     (r['field_b'], r['field_a']) in TRIVIAL_PAIRS, axis=1
        )
    ]

    logger.info("Creating improved figures...")

    heatmap_path = make_clustered_heatmap(filtered_overlap, top_n=25)
    network_path = make_grouped_network(bridge_concepts, filtered_overlap, top_n=35)
    cards_path = make_curated_hypothesis_cards()

    print("\n" + "="*60)
    print("IMPROVED FIGURES CREATED")
    print("="*60)
    print(f"1. {heatmap_path}")
    print(f"2. {network_path}")
    print(f"3. {cards_path}")
    print("="*60)


if __name__ == "__main__":
    main()
