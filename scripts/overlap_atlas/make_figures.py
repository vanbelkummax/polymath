#!/usr/bin/env python3
"""
Step 5: Visualization Figures for Overlap Atlas

Creates:
1. Field-field overlap heatmap (PMI scores)
2. Bridge concept network graph
3. Hypothesis cards (top 10)

Usage:
    python scripts/overlap_atlas/make_figures.py
"""

import json
import logging
from pathlib import Path
from textwrap import wrap

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import networkx as nx

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
INPUT_DIR = Path("/home/user/polymath-repo/var/overlap_atlas")
OUTPUT_DIR = Path("/home/user/polymath-repo/docs/runlogs/figures")

# Color palette (Anthropic-inspired muted tones)
COLORS = {
    'primary': '#5B4B8A',      # Deep purple
    'secondary': '#7B68A6',    # Medium purple
    'accent': '#D4A574',       # Warm gold
    'positive': '#6B9080',     # Teal green
    'negative': '#A45A52',     # Muted red
    'neutral': '#8B8589',      # Gray
    'background': '#F8F6F4'    # Off-white
}


def load_data():
    """Load all overlap atlas data."""
    field_overlap = pd.read_csv(INPUT_DIR / "field_field_overlap.csv")
    bridge_concepts = pd.read_csv(INPUT_DIR / "bridge_concepts.csv")

    with open(INPUT_DIR / "hypotheses.json") as f:
        hypotheses = json.load(f)

    return field_overlap, bridge_concepts, hypotheses


def make_heatmap(field_overlap: pd.DataFrame, top_n: int = 20):
    """Create field-field overlap heatmap."""
    logger.info(f"Creating heatmap for top {top_n} field pairs...")

    # Get top fields by total PMI involvement
    field_scores = {}
    for _, row in field_overlap.iterrows():
        field_scores[row['field_a']] = field_scores.get(row['field_a'], 0) + row['pmi']
        field_scores[row['field_b']] = field_scores.get(row['field_b'], 0) + row['pmi']

    top_fields = sorted(field_scores.keys(), key=lambda x: field_scores[x], reverse=True)[:top_n]

    # Build matrix
    field_idx = {f: i for i, f in enumerate(top_fields)}
    matrix = np.zeros((top_n, top_n))

    for _, row in field_overlap.iterrows():
        a, b = row['field_a'], row['field_b']
        if a in field_idx and b in field_idx:
            i, j = field_idx[a], field_idx[b]
            matrix[i, j] = row['pmi']
            matrix[j, i] = row['pmi']  # Symmetric

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))

    # Custom colormap (diverging)
    from matplotlib.colors import LinearSegmentedColormap
    colors_list = ['#2166AC', '#FFFFFF', '#B2182B']  # Blue-White-Red
    cmap = LinearSegmentedColormap.from_list('pmi', colors_list)

    # Plot heatmap
    vmax = np.percentile(matrix[matrix > 0], 95) if np.any(matrix > 0) else 1
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=-vmax/2, vmax=vmax)

    # Labels
    labels = [f.replace('_', ' ')[:20] for f in top_fields]
    ax.set_xticks(np.arange(top_n))
    ax.set_yticks(np.arange(top_n))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('PMI (Pointwise Mutual Information)', fontsize=11)

    # Add text annotations for strong overlaps
    for i in range(top_n):
        for j in range(top_n):
            if matrix[i, j] > vmax * 0.5:  # Only annotate strong overlaps
                ax.text(j, i, f'{matrix[i,j]:.1f}', ha='center', va='center',
                       color='white', fontsize=7, fontweight='bold')

    ax.set_title('Field-Field Overlap Atlas: Cross-Domain Connections\n(PMI > 0 indicates stronger-than-random co-occurrence)',
                fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()

    output_path = OUTPUT_DIR / "field_overlap_heatmap.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"Saved heatmap to {output_path}")
    return output_path


def make_network(bridge_concepts: pd.DataFrame, field_overlap: pd.DataFrame, top_n: int = 40):
    """Create bridge concept network graph."""
    logger.info(f"Creating network graph for top {top_n} concepts...")

    # Build graph
    G = nx.Graph()

    # Add top bridge concepts as nodes
    top_bridges = bridge_concepts.head(top_n)

    for _, row in top_bridges.iterrows():
        concept = row['concept']
        G.add_node(concept,
                   type='concept',
                   betweenness=row['betweenness'],
                   degree=row['degree'])

    # Add edges between co-occurring concepts
    # Get top fields
    top_fields_a = set(field_overlap['field_a'].head(30))
    top_fields_b = set(field_overlap['field_b'].head(30))
    top_fields = top_fields_a | top_fields_b

    # Connect concepts that share fields
    connected_fields = {}
    for _, row in top_bridges.iterrows():
        concept = row['concept']
        fields_str = row.get('connected_fields', '')
        if isinstance(fields_str, str):
            fields = [f.strip() for f in fields_str.split(',') if f.strip()]
            connected_fields[concept] = set(fields) & top_fields

    # Add edges between concepts sharing fields
    concepts = list(connected_fields.keys())
    for i, c1 in enumerate(concepts):
        for c2 in concepts[i+1:]:
            shared = connected_fields.get(c1, set()) & connected_fields.get(c2, set())
            if shared:
                G.add_edge(c1, c2, weight=len(shared))

    if G.number_of_edges() == 0:
        # Fallback: connect by similarity in betweenness
        for i, row1 in top_bridges.head(20).iterrows():
            for j, row2 in top_bridges.head(20).iloc[i+1:].iterrows():
                if abs(row1['betweenness'] - row2['betweenness']) < 0.002:
                    G.add_edge(row1['concept'], row2['concept'], weight=0.5)

    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_facecolor('#FAFAFA')

    # Node sizes based on betweenness
    node_sizes = []
    for node in G.nodes():
        bc = G.nodes[node].get('betweenness', 0.001)
        node_sizes.append(max(300, bc * 50000))

    # Node colors based on degree (field diversity)
    node_colors = []
    for node in G.nodes():
        degree = G.nodes[node].get('degree', 1)
        if degree > 100:
            node_colors.append(COLORS['primary'])
        elif degree > 50:
            node_colors.append(COLORS['secondary'])
        elif degree > 20:
            node_colors.append(COLORS['positive'])
        else:
            node_colors.append(COLORS['neutral'])

    # Draw edges
    if G.number_of_edges() > 0:
        edges = G.edges(data=True)
        weights = [e[2].get('weight', 1) for e in edges]
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=[w*0.5 for w in weights],
                               edge_color=COLORS['neutral'], ax=ax)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                           alpha=0.8, ax=ax)

    # Draw labels
    labels = {n: n.replace('_', '\n')[:25] for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=7, font_weight='bold', ax=ax)

    # Legend
    legend_elements = [
        mpatches.Patch(color=COLORS['primary'], label='High connectivity (100+ fields)'),
        mpatches.Patch(color=COLORS['secondary'], label='Medium connectivity (50-100)'),
        mpatches.Patch(color=COLORS['positive'], label='Moderate (20-50)'),
        mpatches.Patch(color=COLORS['neutral'], label='Focused (<20)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    ax.set_title('Bridge Concept Network: Ideas That Connect Distant Fields\n(Node size = betweenness centrality, Color = field diversity)',
                fontsize=13, fontweight='bold', pad=20)
    ax.axis('off')

    plt.tight_layout()

    output_path = OUTPUT_DIR / "bridge_concept_network.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"Saved network to {output_path}")
    return output_path


def make_hypothesis_cards(hypotheses: dict, top_n: int = 10):
    """Create hypothesis cards visualization."""
    logger.info(f"Creating hypothesis cards for top {top_n}...")

    hyp_list = hypotheses['hypotheses'][:top_n]

    # Create figure with subplots (2 columns x 5 rows)
    fig, axes = plt.subplots(5, 2, figsize=(16, 20))
    axes = axes.flatten()

    for i, hyp in enumerate(hyp_list):
        ax = axes[i]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

        # Card background
        rect = mpatches.FancyBboxPatch(
            (0.2, 0.2), 9.6, 9.6,
            boxstyle="round,pad=0.1,rounding_size=0.3",
            facecolor='white',
            edgecolor=COLORS['primary'],
            linewidth=2
        )
        ax.add_patch(rect)

        # Type badge
        badge_colors = {
            'field_overlap': COLORS['primary'],
            'bridge_concept': COLORS['positive'],
            'rare_pair': COLORS['accent']
        }
        badge_color = badge_colors.get(hyp['type'], COLORS['neutral'])

        badge = mpatches.FancyBboxPatch(
            (0.5, 8.5), 3, 1,
            boxstyle="round,pad=0.05,rounding_size=0.2",
            facecolor=badge_color,
            edgecolor='none'
        )
        ax.add_patch(badge)
        ax.text(2, 9, hyp['type'].replace('_', ' ').upper(),
               fontsize=8, fontweight='bold', color='white',
               ha='center', va='center')

        # Title
        title = hyp['title'][:60] + ('...' if len(hyp['title']) > 60 else '')
        ax.text(5, 7.5, title, fontsize=10, fontweight='bold',
               ha='center', va='center', wrap=True,
               color=COLORS['primary'])

        # Hypothesis text (wrapped)
        hyp_text = hyp['hypothesis_text'][:200] + ('...' if len(hyp['hypothesis_text']) > 200 else '')
        wrapped = '\n'.join(wrap(hyp_text, width=50))
        ax.text(5, 4.5, wrapped, fontsize=8, ha='center', va='center',
               color='#333333', style='italic')

        # Score bar
        score = min(hyp['score'] / 10, 1.0)  # Normalize
        ax.barh(1.5, score * 8, height=0.5, left=1, color=COLORS['positive'], alpha=0.7)
        ax.barh(1.5, 8, height=0.5, left=1, color='#EEEEEE', alpha=0.3, zorder=0)
        ax.text(9.5, 1.5, f"Score: {hyp['score']:.2f}", fontsize=8, va='center')

        # Evidence count
        n_evidence = len(hyp.get('evidence_passages', []))
        ax.text(1, 0.7, f"Evidence: {n_evidence} passages", fontsize=7, color=COLORS['neutral'])

    # Hide unused subplots
    for i in range(len(hyp_list), len(axes)):
        axes[i].axis('off')

    fig.suptitle('Top 10 Evidence-Bound Hypotheses from Polymath Overlap Atlas',
                fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    output_path = OUTPUT_DIR / "hypothesis_cards.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"Saved hypothesis cards to {output_path}")
    return output_path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data...")
    field_overlap, bridge_concepts, hypotheses = load_data()

    logger.info("Creating figures...")

    # Figure 1: Heatmap
    heatmap_path = make_heatmap(field_overlap, top_n=25)

    # Figure 2: Network
    network_path = make_network(bridge_concepts, field_overlap, top_n=40)

    # Figure 3: Hypothesis cards
    cards_path = make_hypothesis_cards(hypotheses, top_n=10)

    print("\n" + "="*60)
    print("FIGURES CREATED")
    print("="*60)
    print(f"1. {heatmap_path}")
    print(f"2. {network_path}")
    print(f"3. {cards_path}")
    print("="*60)


if __name__ == "__main__":
    main()
