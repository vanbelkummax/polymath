#!/usr/bin/env python3
"""
PolyMax Polymathic Discovery Engine

Multiple methods for generating cross-domain hypotheses:
1. Graph-based: Unexplored method-concept pairs
2. Analogy-based: "X is to Y as A is to B" patterns
3. Bridge-based: Papers connecting distant domains
4. Gap-based: Missing connections in citation network
5. Evolution-based: Predict next method from trajectory
"""

import json
from neo4j import GraphDatabase
from datetime import datetime

# Neo4j connection
DRIVER = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'polymathic2026'))

def method_1_unexplored_pairs(limit=10):
    """
    Find method-concept pairs never combined in any paper.
    Hypothesis: Applying method X to domain Y might yield novel results.
    """
    query = """
    MATCH (m:METHOD), (c:CONCEPT)
    WHERE NOT EXISTS {
        MATCH (p:Paper)-[:MENTIONS]->(m)
        MATCH (p)-[:MENTIONS]->(c)
    }
    WITH m.name AS method, c.name AS concept,
         [(p:Paper)-[:MENTIONS]->(m) | p.title][0..3] AS method_papers,
         [(p:Paper)-[:MENTIONS]->(c) | p.title][0..3] AS concept_papers
    WHERE size(method_papers) > 0 AND size(concept_papers) > 0
    RETURN method, concept, method_papers, concept_papers
    LIMIT $limit
    """
    with DRIVER.session() as session:
        result = session.run(query, limit=limit)
        hypotheses = []
        for record in result:
            hyp = {
                "method": "unexplored_pairs",
                "hypothesis": f"Apply {record['method']} to {record['concept']}",
                "method_name": record['method'],
                "concept_name": record['concept'],
                "rationale": f"{record['method']} successful in: {record['method_papers'][:2]}. "
                           f"{record['concept']} studied in: {record['concept_papers'][:2]}",
                "test": f"Benchmark {record['method']} on {record['concept']} task",
                "kill_shot": f"If no improvement over baseline in {record['concept']} domain"
            }
            hypotheses.append(hyp)
        return hypotheses

def method_2_analogy_patterns():
    """
    Find "X is to Y as A is to B" patterns.
    If method A improves concept X, maybe similar method B improves similar concept Y.
    """
    query = """
    // Find two methods that work on related concepts
    MATCH (m1:METHOD)<-[:MENTIONS]-(p1:Paper)-[:MENTIONS]->(c1:CONCEPT)
    MATCH (m2:METHOD)<-[:MENTIONS]-(p2:Paper)-[:MENTIONS]->(c2:CONCEPT)
    WHERE m1 <> m2 AND c1 <> c2
      AND (c1)-[:RELATES_TO]-(c2)
      AND NOT EXISTS {
        MATCH (px:Paper)-[:MENTIONS]->(m1)
        MATCH (px)-[:MENTIONS]->(c2)
      }
    RETURN m1.name AS method1, c1.name AS concept1,
           m2.name AS method2, c2.name AS concept2,
           p1.title AS paper1, p2.title AS paper2
    LIMIT 5
    """
    with DRIVER.session() as session:
        result = session.run(query)
        hypotheses = []
        for record in result:
            hyp = {
                "method": "analogy",
                "hypothesis": f"{record['method1']}:{record['concept1']} :: {record['method2']}:{record['concept2']}",
                "rationale": f"If {record['method1']} works for {record['concept1']}, "
                           f"try {record['method1']} for related {record['concept2']}",
                "test": f"Apply {record['method1']} to {record['concept2']} domain",
                "evidence": [record['paper1'], record['paper2']]
            }
            hypotheses.append(hyp)
        return hypotheses

def method_3_bridge_papers():
    """
    Find papers that bridge distant domains.
    These are candidates for breakthrough insights.
    """
    query = """
    MATCH (p:Paper)-[:MENTIONS]->(m:METHOD)
    MATCH (p)-[:MENTIONS]->(c:CONCEPT)
    WITH p, collect(DISTINCT m.name) AS methods, collect(DISTINCT c.name) AS concepts
    WHERE size(methods) >= 3 AND size(concepts) >= 3
    RETURN p.title AS paper, methods, concepts,
           size(methods) + size(concepts) AS breadth
    ORDER BY breadth DESC
    LIMIT 5
    """
    with DRIVER.session() as session:
        result = session.run(query)
        bridges = []
        for record in result:
            bridge = {
                "method": "bridge_paper",
                "paper": record['paper'],
                "methods": record['methods'],
                "concepts": record['concepts'],
                "breadth": record['breadth'],
                "insight": f"Study how this paper connects {len(record['methods'])} methods with {len(record['concepts'])} concepts"
            }
            bridges.append(bridge)
        return bridges

def method_4_hub_extension():
    """
    Find hub methods (highly connected) and suggest extensions to new domains.
    """
    query = """
    MATCH (m:METHOD)<-[:MENTIONS]-(p:Paper)-[:MENTIONS]->(c:CONCEPT)
    WITH m.name AS method, collect(DISTINCT c.name) AS connected_concepts, count(DISTINCT p) AS paper_count
    WHERE paper_count >= 5
    MATCH (isolated:CONCEPT)
    WHERE NOT isolated.name IN connected_concepts
    RETURN method, paper_count, connected_concepts[0..5] AS sample_concepts,
           collect(DISTINCT isolated.name)[0..3] AS unexplored_concepts
    ORDER BY paper_count DESC
    LIMIT 5
    """
    with DRIVER.session() as session:
        result = session.run(query)
        hypotheses = []
        for record in result:
            hyp = {
                "method": "hub_extension",
                "hub_method": record['method'],
                "current_reach": record['paper_count'],
                "sample_domains": record['sample_concepts'],
                "unexplored": record['unexplored_concepts'],
                "hypothesis": f"Extend {record['method']} to: {record['unexplored_concepts']}"
            }
            hypotheses.append(hyp)
        return hypotheses

def method_5_concept_trajectory():
    """
    Based on concept co-occurrence patterns, predict next evolution.
    """
    query = """
    // Find concept pairs that often appear together
    MATCH (c1:CONCEPT)<-[:MENTIONS]-(p:Paper)-[:MENTIONS]->(c2:CONCEPT)
    WHERE c1.name < c2.name
    WITH c1.name AS concept1, c2.name AS concept2, count(p) AS co_occurrence
    WHERE co_occurrence >= 2
    RETURN concept1, concept2, co_occurrence
    ORDER BY co_occurrence DESC
    LIMIT 10
    """
    with DRIVER.session() as session:
        result = session.run(query)
        trajectories = []
        for record in result:
            traj = {
                "method": "concept_trajectory",
                "pair": [record['concept1'], record['concept2']],
                "strength": record['co_occurrence'],
                "prediction": f"Research combining {record['concept1']} + {record['concept2']} is active trajectory"
            }
            trajectories.append(traj)
        return trajectories

def generate_all_hypotheses():
    """Run all hypothesis generation methods."""
    print("="*60)
    print("POLYMAX POLYMATHIC DISCOVERY ENGINE")
    print("="*60)

    all_hypotheses = {}

    print("\n1. UNEXPLORED METHOD-CONCEPT PAIRS")
    print("-"*40)
    hyps = method_1_unexplored_pairs(10)
    all_hypotheses["unexplored_pairs"] = hyps
    for h in hyps[:5]:
        print(f"  → {h['hypothesis']}")

    print("\n2. ANALOGY PATTERNS (X:Y :: A:B)")
    print("-"*40)
    hyps = method_2_analogy_patterns()
    all_hypotheses["analogy"] = hyps
    for h in hyps[:5]:
        print(f"  → {h['hypothesis']}")

    print("\n3. BRIDGE PAPERS (High Cross-Domain)")
    print("-"*40)
    bridges = method_3_bridge_papers()
    all_hypotheses["bridges"] = bridges
    for b in bridges[:5]:
        print(f"  → {b['paper'][:50]}... (breadth: {b['breadth']})")

    print("\n4. HUB METHOD EXTENSIONS")
    print("-"*40)
    hyps = method_4_hub_extension()
    all_hypotheses["hub_extension"] = hyps
    for h in hyps[:5]:
        print(f"  → {h['hub_method']} → {h['unexplored']}")

    print("\n5. CONCEPT TRAJECTORIES (Active Pairings)")
    print("-"*40)
    trajs = method_5_concept_trajectory()
    all_hypotheses["trajectories"] = trajs
    for t in trajs[:5]:
        print(f"  → {t['pair']} (strength: {t['strength']})")

    # Save all hypotheses
    output = {
        "generated": datetime.now().isoformat(),
        "total_hypotheses": sum(len(v) for v in all_hypotheses.values()),
        "hypotheses": all_hypotheses
    }

    with open("/home/user/work/polymax/data/polymathic_hypotheses.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"TOTAL: {output['total_hypotheses']} hypotheses generated")
    print(f"Saved to: polymathic_hypotheses.json")

    return all_hypotheses

def rank_hypotheses(hypotheses):
    """Rank hypotheses by expected value."""
    # Simple scoring: unexplored pairs score highest if they involve hub methods
    ranked = []

    for h in hypotheses.get("unexplored_pairs", []):
        # Score based on method popularity (more papers = more validated)
        score = 7.0  # Base score for unexplored
        if "UNI" in h.get("method_name", ""):
            score += 2.0  # UNI is proven
        if "spatial" in h.get("concept_name", "").lower():
            score += 1.5  # Spatial is hot
        if "foundation" in h.get("concept_name", "").lower():
            score += 1.0

        ranked.append({
            "hypothesis": h["hypothesis"],
            "score": score,
            "type": "unexplored_pair",
            "rationale": h.get("rationale", ""),
            "test": h.get("test", "")
        })

    # Sort by score
    ranked.sort(key=lambda x: x["score"], reverse=True)

    print("\n" + "="*60)
    print("TOP 10 RANKED HYPOTHESES BY EXPECTED VALUE")
    print("="*60)
    for i, h in enumerate(ranked[:10], 1):
        print(f"\n{i}. [EV: {h['score']:.1f}] {h['hypothesis']}")
        print(f"   Type: {h['type']}")
        print(f"   Test: {h['test'][:80]}..." if len(h.get('test', '')) > 80 else f"   Test: {h.get('test', 'N/A')}")

    return ranked

if __name__ == "__main__":
    from pathlib import Path
    Path("/home/user/work/polymax/data").mkdir(exist_ok=True)

    hyps = generate_all_hypotheses()
    ranked = rank_hypotheses(hyps)

    print("\n\nDone! Check /home/user/work/polymax/data/ for outputs.")
