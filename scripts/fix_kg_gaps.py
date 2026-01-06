#!/usr/bin/env python3
"""
Fix Knowledge Graph Gaps - Addresses all Priority issues from review

Priority 1: Add CO_OCCURS edges between concepts (based on paper co-mentions)
Priority 2: Add cross-domain concepts from other fields
Priority 3: Link orphan papers to concepts
"""

from neo4j import GraphDatabase
from tqdm import tqdm

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "polymathic2026"

# Priority 2: Cross-domain concepts to seed
CROSS_DOMAIN_CONCEPTS = {
    # Signal Processing / Information Theory
    "compressed_sensing": "Recovery of sparse signals from incomplete measurements (Candès, Tao, Donoho)",
    "sparse_coding": "Representing data as sparse linear combinations of basis elements",
    "dictionary_learning": "Learning overcomplete bases for sparse representation",
    "information_bottleneck": "Compressing input while preserving relevant information about output",
    "rate_distortion": "Fundamental limits of lossy compression",
    "maximum_entropy": "Principle of least commitment given constraints",

    # Physics / Thermodynamics
    "free_energy_principle": "Systems minimize variational free energy (Friston)",
    "thermodynamic_inference": "Using physics principles for statistical inference",
    "reaction_diffusion": "Pattern formation through local activation and lateral inhibition (Turing)",
    "phase_transition": "Qualitative changes in system behavior at critical points",

    # Causal Inference / Econometrics
    "counterfactual_reasoning": "Reasoning about what would happen under different conditions",
    "do_calculus": "Formal rules for causal reasoning (Pearl)",
    "instrumental_variables": "Using exogenous variation to identify causal effects",
    "potential_outcomes": "Rubin causal model framework",
    "difference_in_differences": "Quasi-experimental design for causal inference",

    # Chemistry / Materials Science
    "binding_affinity": "Strength of molecular interactions",
    "taphonomy": "Study of how organisms decay and become preserved",
    "spectroscopy": "Probing matter through light-matter interaction",
    "stoichiometry": "Quantitative relationships in chemical reactions",

    # Ecology / Evolution
    "niche_construction": "Organisms modify their environment, changing selection pressures",
    "evolutionary_game_theory": "Strategic interactions in evolving populations",
    "metabolic_scaling": "How metabolic rate scales with body size (Kleiber's law)",
    "trophic_cascade": "Effects propagating through food web levels",

    # Systems Theory / Cybernetics
    "feedback_control": "Using output to regulate input (Wiener, Ashby)",
    "autopoiesis": "Self-maintaining systems (Maturana, Varela)",
    "requisite_variety": "Controller must have variety >= system being controlled (Ashby)",
    "attractor_dynamics": "System states that trajectories converge toward",

    # Cognitive Science / Linguistics
    "structure_mapping": "Analogical reasoning through relational alignment (Gentner)",
    "conceptual_blending": "Creating new concepts by combining frames (Fauconnier, Turner)",
    "embodied_cognition": "Cognition grounded in bodily experience",
    "predictive_coding": "Brain as prediction machine minimizing surprise",

    # Computer Science Theory
    "kolmogorov_complexity": "Shortest program that produces a given output",
    "pac_learning": "Probably approximately correct learning framework",
    "no_free_lunch": "No algorithm is universally best across all problems",
    "occams_razor": "Prefer simpler explanations",
}


def add_co_occurs_edges(driver):
    """Priority 1: Add CO_OCCURS edges between concepts that appear in same papers."""
    print("\n=== Priority 1: Adding CO_OCCURS edges ===")

    with driver.session() as session:
        # First, count how many we'll add
        result = session.run("""
            MATCH (p:Paper)-[:MENTIONS]->(c1:CONCEPT)
            MATCH (p)-[:MENTIONS]->(c2:CONCEPT)
            WHERE id(c1) < id(c2)
            WITH c1, c2, count(p) as weight
            WHERE weight >= 2
            RETURN count(*) as total
        """)
        total = result.single()["total"]
        print(f"  Will create {total} CO_OCCURS edges (weight >= 2)")

        # Create the edges
        result = session.run("""
            MATCH (p:Paper)-[:MENTIONS]->(c1:CONCEPT)
            MATCH (p)-[:MENTIONS]->(c2:CONCEPT)
            WHERE id(c1) < id(c2)
            WITH c1, c2, count(p) as weight
            WHERE weight >= 2
            MERGE (c1)-[r:CO_OCCURS]->(c2)
            SET r.weight = weight, r.created = datetime()
            RETURN count(r) as created
        """)
        created = result.single()["created"]
        print(f"  Created {created} CO_OCCURS edges")

        # Show top co-occurrences
        result = session.run("""
            MATCH (c1:CONCEPT)-[r:CO_OCCURS]->(c2:CONCEPT)
            RETURN c1.name, c2.name, r.weight
            ORDER BY r.weight DESC LIMIT 10
        """)
        print("\n  Top co-occurrences:")
        for record in result:
            print(f"    {record['c1.name']} <-> {record['c2.name']}: {record['r.weight']}")


def add_cross_domain_concepts(driver):
    """Priority 2: Add cross-domain concepts from other fields."""
    print("\n=== Priority 2: Adding cross-domain concepts ===")

    added = 0
    with driver.session() as session:
        for name, description in tqdm(CROSS_DOMAIN_CONCEPTS.items(), desc="Adding concepts"):
            result = session.run("""
                MERGE (c:CONCEPT {name: $name})
                ON CREATE SET
                    c.description = $desc,
                    c.domain = 'cross_domain',
                    c.created = datetime()
                RETURN c.created IS NOT NULL as created
            """, {"name": name, "desc": description})
            if result.single()["created"]:
                added += 1

    print(f"  Added {added} new cross-domain concepts")
    print(f"  Total cross-domain concepts: {len(CROSS_DOMAIN_CONCEPTS)}")


def link_orphan_papers(driver):
    """Priority 3: Try to link papers that have no concept connections."""
    print("\n=== Priority 3: Linking orphan papers ===")

    # Keywords to search for in titles
    KEYWORD_CONCEPTS = {
        "deep learning": "deep_learning",
        "neural network": "neural_network",
        "transformer": "transformer",
        "attention": "attention",
        "segmentation": "segmentation",
        "classification": "classification",
        "detection": "detection",
        "pathology": "pathology",
        "histology": "histology",
        "cancer": "cancer",
        "tumor": "tumor",
        "spatial": "spatial_transcriptomics",
        "single-cell": "single_cell",
        "scRNA": "single_cell",
        "foundation model": "foundation_model",
        "self-supervised": "self_supervised",
        "contrastive": "contrastive_learning",
        "embedding": "embedding",
        "deconvolution": "deconvolution",
        "cell type": "cell_type",
    }

    with driver.session() as session:
        # Find orphan papers
        result = session.run("""
            MATCH (p:Paper)
            WHERE NOT (p)-[:MENTIONS]->()
            RETURN p.title as title, id(p) as pid
        """)
        orphans = [(r["title"], r["pid"]) for r in result]
        print(f"  Found {len(orphans)} orphan papers")

        linked = 0
        for title, pid in tqdm(orphans, desc="Linking"):
            if not title:
                continue
            title_lower = title.lower()

            for keyword, concept in KEYWORD_CONCEPTS.items():
                if keyword in title_lower:
                    session.run("""
                        MATCH (p:Paper) WHERE id(p) = $pid
                        MERGE (c:CONCEPT {name: $concept})
                        MERGE (p)-[:MENTIONS]->(c)
                    """, {"pid": pid, "concept": concept})
                    linked += 1

        print(f"  Created {linked} new MENTIONS edges for orphan papers")


def verify_fixes(driver):
    """Verify the fixes worked."""
    print("\n=== Verification ===")

    with driver.session() as session:
        # Count CO_OCCURS
        result = session.run("MATCH ()-[r:CO_OCCURS]->() RETURN count(r) as count")
        print(f"  CO_OCCURS edges: {result.single()['count']}")

        # Count cross-domain concepts
        result = session.run("MATCH (c:CONCEPT {domain: 'cross_domain'}) RETURN count(c) as count")
        print(f"  Cross-domain concepts: {result.single()['count']}")

        # Count orphan papers
        result = session.run("MATCH (p:Paper) WHERE NOT (p)-[:MENTIONS]->() RETURN count(p) as count")
        print(f"  Remaining orphan papers: {result.single()['count']}")

        # New: Find TRUE gaps (cross-domain concepts not connected to biology)
        result = session.run("""
            MATCH (c:CONCEPT {domain: 'cross_domain'})
            WHERE NOT (c)<-[:MENTIONS]-(:Paper)
            RETURN c.name as name, c.description as desc
            LIMIT 10
        """)
        print("\n  True unexplored cross-domain concepts:")
        for record in result:
            print(f"    • {record['name']}: {record['desc'][:50]}...")


def main():
    print("=" * 60)
    print("KNOWLEDGE GRAPH GAP FIXES")
    print("=" * 60)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    add_co_occurs_edges(driver)
    add_cross_domain_concepts(driver)
    link_orphan_papers(driver)
    verify_fixes(driver)

    driver.close()

    print("\n" + "=" * 60)
    print("FIXES COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
