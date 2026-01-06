#!/usr/bin/env python3
"""
Visualize Knowledge Graph from Neo4j
Creates both interactive (PyVis) and static (NetworkX) visualizations
"""

from neo4j import GraphDatabase
from pyvis.network import Network
import networkx as nx
import matplotlib.pyplot as plt
import argparse
import os

# Configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "polymathic2026"
OUTPUT_DIR = "/home/user/work/polymax/visualizations"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def fetch_graph_data():
    """Fetch graph data from Neo4j"""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    nodes = []
    edges = []

    with driver.session() as session:
        # Fetch all nodes
        result = session.run("""
            MATCH (n)
            RETURN id(n) as id, labels(n) as labels, properties(n) as props
        """)
        for record in result:
            node_id = str(record['id'])
            labels = record['labels']
            props = record['props']

            # Determine node type and label
            node_type = labels[0] if labels else 'Unknown'
            node_label = props.get('name', props.get('title', f'Node {node_id}'))

            nodes.append({
                'id': node_id,
                'label': node_label[:50],  # Truncate long labels
                'type': node_type,
                'props': props
            })

        # Fetch all relationships
        result = session.run("""
            MATCH (a)-[r]->(b)
            RETURN id(a) as source, id(b) as target, type(r) as rel_type, properties(r) as props
        """)
        for record in result:
            edges.append({
                'source': str(record['source']),
                'target': str(record['target']),
                'type': record['rel_type'],
                'props': record['props']
            })

    driver.close()
    return nodes, edges

def create_interactive_viz(nodes, edges, output_file):
    """Create interactive PyVis visualization"""
    net = Network(height='900px', width='100%', bgcolor='#1a1a1a', font_color='white')

    # Configure physics
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 200,
          "springConstant": 0.08
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": {"iterations": 150}
      }
    }
    """)

    # Color scheme by node type
    colors = {
        'Paper': '#4A90E2',      # Blue
        'METHOD': '#E24A4A',     # Red
        'CONCEPT': '#4AE290',    # Green
        'GENE': '#E2D14A',       # Yellow
        'INSTITUTION': '#E24AE2', # Purple
        'AUTHOR': '#E2904A'      # Orange
    }

    # Sizes by node type
    sizes = {
        'Paper': 15,
        'METHOD': 25,
        'CONCEPT': 25,
        'GENE': 20,
        'INSTITUTION': 20,
        'AUTHOR': 20
    }

    # Add nodes
    for node in nodes:
        node_type = node['type']
        color = colors.get(node_type, '#CCCCCC')
        size = sizes.get(node_type, 15)

        # Create hover tooltip
        title = f"<b>{node['label']}</b><br>Type: {node_type}"
        if 'year' in node['props']:
            title += f"<br>Year: {node['props']['year']}"

        net.add_node(
            node['id'],
            label=node['label'],
            color=color,
            size=size,
            title=title,
            shape='dot'
        )

    # Add edges
    for edge in edges:
        edge_type = edge['type']
        confidence = edge['props'].get('confidence', 0.5)

        # Edge thickness by confidence
        width = 1 + confidence * 3

        # Edge color by type
        edge_colors = {
            'MENTIONS': '#666666',
            'USES': '#4AE290',
            'CITES': '#4A90E2',
            'PRODUCES': '#E2904A',
            'CAUSES': '#E24A4A'
        }
        color = edge_colors.get(edge_type, '#666666')

        title = f"{edge_type}"
        if 'evidence' in edge['props']:
            title += f"<br>{edge['props']['evidence'][:100]}"

        net.add_edge(
            edge['source'],
            edge['target'],
            title=title,
            color=color,
            width=width,
            arrows='to'
        )

    # Save
    net.save_graph(output_file)
    print(f"✓ Interactive visualization saved: {output_file}")

def create_static_viz(nodes, edges, output_file):
    """Create static NetworkX visualization"""
    G = nx.DiGraph()

    # Add nodes with attributes
    for node in nodes:
        G.add_node(node['id'], **node)

    # Add edges
    for edge in edges:
        G.add_edge(edge['source'], edge['target'], **edge)

    # Layout
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

    # Colors by type
    color_map = {
        'Paper': '#4A90E2',
        'METHOD': '#E24A4A',
        'CONCEPT': '#4AE290',
        'GENE': '#E2D14A'
    }

    node_colors = [color_map.get(G.nodes[node]['type'], '#CCCCCC') for node in G.nodes()]

    # Sizes by type
    size_map = {
        'Paper': 300,
        'METHOD': 500,
        'CONCEPT': 500,
        'GENE': 400
    }

    node_sizes = [size_map.get(G.nodes[node]['type'], 300) for node in G.nodes()]

    # Draw
    plt.figure(figsize=(20, 15), facecolor='#1a1a1a')
    ax = plt.gca()
    ax.set_facecolor('#1a1a1a')

    # Draw edges with varying alpha based on confidence
    for edge in G.edges(data=True):
        confidence = edge[2].get('props', {}).get('confidence', 0.5)
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[edge[:2]],
            alpha=0.3 + confidence * 0.5,
            edge_color='#666666',
            arrows=True,
            arrowsize=10,
            width=1 + confidence * 2
        )

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9
    )

    # Draw labels for METHOD and CONCEPT nodes only
    labels = {
        node: G.nodes[node]['label'][:15]
        for node in G.nodes()
        if G.nodes[node]['type'] in ['METHOD', 'CONCEPT']
    }

    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=8,
        font_color='white'
    )

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, facecolor='#1a1a1a', bbox_inches='tight')
    print(f"✓ Static visualization saved: {output_file}")
    plt.close()

def print_graph_stats(nodes, edges):
    """Print graph statistics"""
    print("\n" + "="*60)
    print("KNOWLEDGE GRAPH STATISTICS")
    print("="*60)

    # Count by node type
    from collections import Counter
    node_types = Counter(node['type'] for node in nodes)

    print("\nNODE COUNTS:")
    for node_type, count in sorted(node_types.items()):
        print(f"  {node_type}: {count}")

    print(f"\nTOTAL NODES: {len(nodes)}")
    print(f"TOTAL EDGES: {len(edges)}")

    # Edge types
    edge_types = Counter(edge['type'] for edge in edges)
    print("\nRELATIONSHIP COUNTS:")
    for edge_type, count in sorted(edge_types.items()):
        print(f"  {edge_type}: {count}")

    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Knowledge Graph")
    parser.add_argument('--interactive', action='store_true', help='Create interactive PyVis visualization')
    parser.add_argument('--static', action='store_true', help='Create static NetworkX visualization')
    parser.add_argument('--both', action='store_true', help='Create both visualizations')

    args = parser.parse_args()

    # Default to both if no option specified
    if not (args.interactive or args.static or args.both):
        args.both = True

    print("\n" + "="*60)
    print("KNOWLEDGE GRAPH VISUALIZATION")
    print("="*60)
    print("\nFetching graph data from Neo4j...")

    nodes, edges = fetch_graph_data()

    print(f"✓ Loaded {len(nodes)} nodes and {len(edges)} edges")

    print_graph_stats(nodes, edges)

    if args.interactive or args.both:
        interactive_file = os.path.join(OUTPUT_DIR, 'knowledge_graph_interactive.html')
        create_interactive_viz(nodes, edges, interactive_file)
        print(f"\n→ Open in browser: file://{interactive_file}")

    if args.static or args.both:
        static_file = os.path.join(OUTPUT_DIR, 'knowledge_graph_static.png')
        create_static_viz(nodes, edges, static_file)

    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60 + "\n")
