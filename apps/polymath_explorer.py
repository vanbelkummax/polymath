#!/usr/bin/env python3
"""
Polymath Overlap Atlas Explorer

Interactive Streamlit app for exploring the Polymath knowledge base,
field overlaps, bridge concepts, and evidence-bound hypotheses.

Usage:
    streamlit run apps/polymath_explorer.py
"""

import json
import sys
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

# Configuration
DATA_DIR = Path("/home/user/polymath-repo/var/overlap_atlas")
FIGURES_DIR = Path("/home/user/polymath-repo/docs/runlogs/figures")


@st.cache_data
def load_data():
    """Load all overlap atlas data."""
    field_overlap = pd.read_csv(DATA_DIR / "field_field_overlap.csv")
    bridge_concepts = pd.read_csv(DATA_DIR / "bridge_concepts.csv")
    rare_pairs = pd.read_csv(DATA_DIR / "rare_pairs.csv")

    with open(DATA_DIR / "hypotheses.json") as f:
        hypotheses = json.load(f)

    with open(DATA_DIR / "extraction_stats.json") as f:
        stats = json.load(f)

    return field_overlap, bridge_concepts, rare_pairs, hypotheses, stats


@st.cache_resource
def get_searcher():
    """Get hybrid searcher instance (cached)."""
    from lib.hybrid_search_v2 import HybridSearcherV2
    return HybridSearcherV2(use_reranker=False)


def search_tab():
    """Search tab with atlas_search."""
    st.header("Tri-Modal Search")
    st.markdown("""
    Search the Polymath corpus using three retrieval channels:
    - **SQL**: Direct concept matching in passage_concepts
    - **Vector**: Semantic embedding similarity (BGE-M3)
    - **Graph**: Concept expansion via Neo4j relationships
    """)

    query = st.text_input("Enter search query:", placeholder="spatial transcriptomics gene expression")

    col1, col2, col3 = st.columns(3)
    with col1:
        n_results = st.slider("Results", 5, 50, 20)
    with col2:
        min_conf = st.slider("Min confidence", 0.5, 0.95, 0.7)
    with col3:
        include_graph = st.checkbox("Include graph expansion", value=False)

    if st.button("Search", type="primary"):
        if query:
            with st.spinner("Searching..."):
                try:
                    searcher = get_searcher()
                    result = searcher.atlas_search(
                        query,
                        n=n_results,
                        min_confidence=min_conf,
                        include_graph=include_graph
                    )

                    # Show explain trace
                    st.subheader("Retrieval Explanation")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("SQL Matches", result['explain']['sql']['count'])
                        if result['explain']['sql']['matched_concepts']:
                            st.caption(f"Concepts: {', '.join(result['explain']['sql']['matched_concepts'])}")

                    with col2:
                        st.metric("Vector Matches", result['explain']['vector']['count'])
                        if result['explain']['vector']['scores']:
                            st.caption(f"Top score: {result['explain']['vector']['scores'][0]:.3f}")

                    with col3:
                        st.metric("Graph Paths", result['explain']['graph']['paths_found'])
                        if result['explain']['graph']['expanded_concepts']:
                            st.caption(f"Expanded: {', '.join(result['explain']['graph']['expanded_concepts'][:3])}")

                    # Show results
                    st.subheader(f"Results ({len(result['results'])} found)")

                    for i, r in enumerate(result['results'], 1):
                        with st.expander(f"{i}. [{r.source}] {r.title[:70]}... (score: {r.score:.3f})"):
                            st.markdown(f"**ID:** `{r.id[:20]}...`")
                            st.markdown(f"**Source:** {r.source}")
                            st.markdown(f"**Content:**")
                            st.text(r.content[:500])
                            if r.metadata:
                                st.markdown("**Metadata:**")
                                st.json({k: v for k, v in r.metadata.items() if v is not None})

                except Exception as e:
                    st.error(f"Search error: {e}")


def overlap_tab():
    """Field overlap exploration tab."""
    st.header("Field-Field Overlap Atlas")

    field_overlap, _, _, _, stats = load_data()

    # Stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Field Pairs", len(field_overlap))
    with col2:
        st.metric("Total Passages", f"{stats['unique_passages']:,}")
    with col3:
        st.metric("Unique Concepts", f"{stats['unique_concepts']:,}")

    # Heatmap image
    st.subheader("PMI Heatmap")
    heatmap_path = FIGURES_DIR / "field_overlap_heatmap.png"
    if heatmap_path.exists():
        st.image(str(heatmap_path), caption="Field-Field PMI Overlap Heatmap")
    else:
        st.warning("Heatmap not found. Run `make_figures.py` first.")

    # Data table
    st.subheader("Top Overlaps by PMI")

    # Filter controls
    col1, col2 = st.columns(2)
    with col1:
        min_count = st.slider("Minimum co-occurrences", 5, 200, 10)
    with col2:
        top_n = st.slider("Show top N", 10, 100, 30)

    filtered = field_overlap[field_overlap['count'] >= min_count].head(top_n)

    # Format for display
    display_df = filtered[['field_a', 'field_b', 'pmi', 'count', 'lift']].copy()
    display_df['field_a'] = display_df['field_a'].str.replace('_', ' ')
    display_df['field_b'] = display_df['field_b'].str.replace('_', ' ')
    display_df['pmi'] = display_df['pmi'].round(3)
    display_df['lift'] = display_df['lift'].round(3)
    display_df.columns = ['Field A', 'Field B', 'PMI', 'Co-occurrences', 'Lift']

    st.dataframe(display_df, use_container_width=True)

    # Field selector for detailed view
    st.subheader("Explore Specific Field")
    all_fields = set(field_overlap['field_a']) | set(field_overlap['field_b'])
    selected_field = st.selectbox(
        "Select a field:",
        sorted([f.replace('_', ' ') for f in all_fields])
    )

    if selected_field:
        field_key = selected_field.replace(' ', '_')
        related = field_overlap[
            (field_overlap['field_a'] == field_key) |
            (field_overlap['field_b'] == field_key)
        ].sort_values('pmi', ascending=False)

        st.markdown(f"**Fields related to '{selected_field}':**")

        for _, row in related.head(15).iterrows():
            other_field = row['field_b'] if row['field_a'] == field_key else row['field_a']
            st.markdown(f"- {other_field.replace('_', ' ')}: PMI={row['pmi']:.3f}, n={row['count']}")


def bridges_tab():
    """Bridge concepts exploration tab."""
    st.header("Bridge Concepts")

    _, bridge_concepts, rare_pairs, _, _ = load_data()

    st.markdown("""
    Bridge concepts connect distant fields and may enable cross-domain knowledge transfer.
    Identified via betweenness centrality in the field-concept graph.
    """)

    # Network image
    st.subheader("Bridge Concept Network")
    network_path = FIGURES_DIR / "bridge_concept_network.png"
    if network_path.exists():
        st.image(str(network_path), caption="Top Bridge Concepts by Betweenness Centrality")
    else:
        st.warning("Network graph not found. Run `make_figures.py` first.")

    # Stats
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Bridge Concepts", len(bridge_concepts))
    with col2:
        st.metric("Rare Pairs", len(rare_pairs))

    # Table
    st.subheader("Top Bridge Concepts")
    top_n = st.slider("Show top N", 10, 100, 30, key="bridge_n")

    display_df = bridge_concepts.head(top_n)[['concept', 'betweenness', 'degree', 'field_count']].copy()
    display_df['concept'] = display_df['concept'].str.replace('_', ' ')
    display_df['betweenness'] = display_df['betweenness'].apply(lambda x: f"{x:.5f}")
    display_df.columns = ['Concept', 'Betweenness', 'Fields Connected', 'Total Fields']

    st.dataframe(display_df, use_container_width=True)

    # Rare pairs
    st.subheader("Rare Cross-Field Pairs")
    st.markdown("Unusual concept combinations that appear across multiple fields:")

    rare_display = rare_pairs.head(20)[['concept_a', 'concept_b', 'count', 'avg_confidence', 'n_field_pairs']].copy()
    rare_display['concept_a'] = rare_display['concept_a'].str.replace('_', ' ')
    rare_display['concept_b'] = rare_display['concept_b'].str.replace('_', ' ')
    rare_display['avg_confidence'] = rare_display['avg_confidence'].round(3)
    rare_display.columns = ['Concept A', 'Concept B', 'Count', 'Avg Confidence', 'Field Pairs']

    st.dataframe(rare_display, use_container_width=True)


def hypotheses_tab():
    """Hypotheses exploration tab."""
    st.header("Evidence-Bound Hypotheses")

    _, _, _, hypotheses, _ = load_data()

    st.markdown("""
    These hypotheses are generated from field overlaps, bridge concepts, and rare pairs.
    Each is grounded in actual evidence from the corpus.
    """)

    # Stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Hypotheses", hypotheses['total_hypotheses'])
    with col2:
        st.metric("Field Overlap", hypotheses['by_type']['field_overlap'])
    with col3:
        st.metric("Bridge Concept", hypotheses['by_type']['bridge_concept'])

    # Cards image
    cards_path = FIGURES_DIR / "hypothesis_cards.png"
    if cards_path.exists():
        st.image(str(cards_path), caption="Top 10 Hypothesis Cards")

    # Filter
    st.subheader("Browse Hypotheses")
    type_filter = st.selectbox(
        "Filter by type:",
        ['All', 'field_overlap', 'bridge_concept', 'rare_pair']
    )

    hyp_list = hypotheses['hypotheses']
    if type_filter != 'All':
        hyp_list = [h for h in hyp_list if h['type'] == type_filter]

    for i, hyp in enumerate(hyp_list, 1):
        with st.expander(f"{i}. {hyp['title'][:70]}... (score: {hyp['score']:.3f})"):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"**Type:** {hyp['type'].replace('_', ' ').title()}")
                st.markdown(f"> {hyp['hypothesis_text']}")

                if hyp.get('fields'):
                    st.markdown(f"**Fields:** {', '.join([f.replace('_', ' ') for f in hyp['fields'][:4]])}")

                if hyp.get('bridge_concepts'):
                    st.markdown(f"**Bridge concepts:** {', '.join([c.replace('_', ' ') for c in hyp['bridge_concepts'][:3]])}")

            with col2:
                st.metric("Score", f"{hyp['score']:.3f}")
                st.metric("Evidence", len(hyp.get('evidence_passages', [])))

            # Evidence
            if hyp.get('evidence_passages'):
                st.markdown("**Sample Evidence:**")
                for ev in hyp['evidence_passages'][:3]:
                    evidence_text = ev.get('evidence') or ev.get('source_text') or ev.get('passage_preview', '')
                    if evidence_text:
                        evidence_text = evidence_text[:300] + ('...' if len(evidence_text) > 300 else '')
                        st.markdown(f"- `{ev['passage_id'][:12]}...`: *{evidence_text}*")


def main():
    st.set_page_config(
        page_title="Polymath Overlap Atlas Explorer",
        page_icon="🔬",
        layout="wide"
    )

    st.title("Polymath Overlap Atlas Explorer")
    st.markdown("*Discover cross-domain connections in the Polymath knowledge base*")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Search", "📊 Field Overlaps", "🌉 Bridges", "💡 Hypotheses"])

    with tab1:
        search_tab()

    with tab2:
        overlap_tab()

    with tab3:
        bridges_tab()

    with tab4:
        hypotheses_tab()

    # Footer
    st.markdown("---")
    st.markdown("*Polymath Overlap Atlas | Built with Streamlit*")


if __name__ == "__main__":
    main()
