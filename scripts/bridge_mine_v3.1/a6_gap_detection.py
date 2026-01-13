#!/usr/bin/env python3
"""
A6 Bucket Gap Detection: 3D Morphology / Spatial Reconstruction

Focused gap detection for spatial omics domain:
1. Identify spatial-relevant PROBLEM nodes
2. Find METHOD→PROBLEM edges from other domains
3. Detect untried transfers via :SIMILAR_TO
4. Filter out synonym/plural pairs
"""

import os
import sys
import json
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass

sys.path.insert(0, '/home/user/polymath-repo')

import psycopg2
from neo4j import GraphDatabase
import numpy as np

# Database connections
PG_CONN = "dbname=polymath user=polymath"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "polymathic2026")

# A6 Bucket: 3D morphology / spatial reconstruction keywords
SPATIAL_KEYWORDS = {
    # Core spatial
    'spatial', 'reconstruction', 'registration', 'alignment',
    '3d', 'three_dimensional', 'volumetric', 'tomography', 'tomographic',
    # Tissue/section specific
    'tissue', 'section', 'serial_section', 'histology', 'morphology',
    'deformable', 'warping', 'deformation',
    # Spatial transcriptomics
    'transcriptomics', 'spatial_transcriptomics', 'gene_expression',
    'spot', 'visium', 'xenium', 'merfish', 'starmap',
    # Image analysis
    'segmentation', 'detection', 'localization', 'resolution',
    'super_resolution', 'upsampling', 'imputation',
}

# Methods from inverse problems / other domains that might transfer
TRANSFER_CANDIDATE_METHODS = {
    # Inverse problems
    'optimal_transport', 'wasserstein', 'sinkhorn',
    'compressed_sensing', 'sparse_reconstruction', 'inpainting',
    'deconvolution', 'super_resolution', 'upsampling',
    # Registration/alignment
    'deformable_registration', 'diffeomorphic', 'demons',
    'optical_flow', 'flow_estimation', 'motion_estimation',
    # Tomography
    'tomographic_reconstruction', 'filtered_back_projection', 'radon',
    'ct_reconstruction', 'iterative_reconstruction',
    # Neural approaches
    'neural_radiance_field', 'nerf', 'implicit_neural',
    'diffusion_model', 'score_matching', 'denoising',
    'variational_autoencoder', 'vae', 'generative',
    # Physics-informed
    'physics_informed', 'pinn', 'constraint_learning',
}


def is_synonym_pair(name1: str, name2: str) -> bool:
    """Check if two names are likely synonyms (plural/spelling variants)."""
    # Normalize
    n1 = name1.lower().replace('_', '').replace('-', '').replace(' ', '')
    n2 = name2.lower().replace('_', '').replace('-', '').replace(' ', '')

    # Check if one is substring of other
    if n1 in n2 or n2 in n1:
        return True

    # Check edit distance (simple)
    if len(n1) > 5 and len(n2) > 5:
        # Strip common suffixes
        for suffix in ['s', 'es', 'ing', 'ion', 'tion', 'ation', 'isation', 'ization']:
            if n1.endswith(suffix):
                n1_stem = n1[:-len(suffix)]
            else:
                n1_stem = n1
            if n2.endswith(suffix):
                n2_stem = n2[:-len(suffix)]
            else:
                n2_stem = n2

        # If stems are very similar
        if n1_stem == n2_stem:
            return True

        # Levenshtein-like check (simple version)
        if abs(len(n1) - len(n2)) <= 2:
            matches = sum(c1 == c2 for c1, c2 in zip(n1, n2))
            if matches / max(len(n1), len(n2)) > 0.85:
                return True

    return False


def get_spatial_problems(driver) -> List[dict]:
    """Get PROBLEM nodes related to spatial/3D domains."""
    print("Finding spatial-relevant PROBLEM nodes...")

    with driver.session() as session:
        # Get all problems
        result = session.run("""
            MATCH (p:PROBLEM)
            RETURN p.name as name, p.mention_count as mentions
            ORDER BY mentions DESC
        """)

        spatial_problems = []
        for r in result:
            name = r['name'].lower()
            # Check if name contains spatial keywords
            is_spatial = any(kw in name for kw in SPATIAL_KEYWORDS)
            if is_spatial:
                spatial_problems.append({
                    'name': r['name'],
                    'mentions': r['mentions']
                })

        print(f"  Found {len(spatial_problems)} spatial-relevant problems")
        for p in spatial_problems[:10]:
            print(f"    {p['name']} ({p['mentions']})")

        return spatial_problems


def get_transfer_methods(driver) -> List[dict]:
    """Get METHOD nodes that are transfer candidates."""
    print("\nFinding transfer candidate METHODs...")

    with driver.session() as session:
        result = session.run("""
            MATCH (m:METHOD)
            RETURN m.name as name, m.mention_count as mentions
            ORDER BY mentions DESC
        """)

        transfer_methods = []
        for r in result:
            name = r['name'].lower().replace(' ', '_')
            # Check if name matches transfer candidate patterns
            is_candidate = any(kw in name for kw in TRANSFER_CANDIDATE_METHODS)
            if is_candidate:
                transfer_methods.append({
                    'name': r['name'],
                    'mentions': r['mentions']
                })

        print(f"  Found {len(transfer_methods)} transfer candidate methods")
        for m in transfer_methods[:10]:
            print(f"    {m['name']} ({m['mentions']})")

        return transfer_methods


def build_spatial_similarity(driver, spatial_problems: List[dict]) -> List[Tuple[str, str, float]]:
    """Build similarity edges between spatial problems."""
    print("\nBuilding similarity edges for spatial problems...")

    if len(spatial_problems) < 2:
        print("  Not enough spatial problems")
        return []

    try:
        from FlagEmbedding import BGEM3FlagModel

        print("  Loading BGE-M3 model...")
        model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

        # Embed problem names
        problem_names = [p['name'] for p in spatial_problems]
        embeddings_result = model.encode(problem_names, batch_size=32)
        problem_vecs = embeddings_result['dense_vecs']

        # Compute pairwise similarities
        edges = []
        for i in range(len(problem_names)):
            for j in range(i + 1, len(problem_names)):
                p1, p2 = problem_names[i], problem_names[j]

                # Skip synonym pairs
                if is_synonym_pair(p1, p2):
                    continue

                # Cosine similarity
                e1, e2 = problem_vecs[i], problem_vecs[j]
                sim = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))

                # Higher threshold for meaningful similarity
                if sim > 0.75:
                    edges.append((p1, p2, sim))

        edges.sort(key=lambda x: -x[2])
        print(f"  Found {len(edges)} non-synonym similarity edges (threshold > 0.75)")
        for p1, p2, sim in edges[:10]:
            print(f"    {p1} ~ {p2} ({sim:.3f})")

        return edges

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return []


def run_focused_gap_detection(driver, spatial_problems: List[dict], transfer_methods: List[dict]) -> List[dict]:
    """Run gap detection focused on spatial problems and transfer methods."""
    print("\n" + "="*70)
    print("A6 BUCKET GAP DETECTION: 3D Morphology / Spatial Reconstruction")
    print("="*70)

    spatial_names = {p['name'] for p in spatial_problems}
    transfer_names = {m['name'] for m in transfer_methods}

    gaps = []

    with driver.session() as session:
        # Query 1: Find methods that solve non-spatial problems similar to spatial problems
        print("\nQuery: Methods solving similar problems not yet applied to spatial...")

        # Convert transfer_names to list for Neo4j parameter
        transfer_names_list = list(transfer_names)

        result = session.run("""
            MATCH (m:METHOD)-[:SOLVES]->(p1:PROBLEM)-[:SIMILAR_TO]-(p2:PROBLEM)
            WHERE NOT (m)-[:SOLVES]->(p2)
              AND m.name IN $transfer_methods
            WITH m, p1, p2,
                 m.mention_count as method_mentions,
                 m.doc_count as method_docs
            RETURN m.name as method,
                   p1.name as source_problem,
                   p2.name as target_problem,
                   method_mentions,
                   method_docs
            ORDER BY method_mentions DESC
            LIMIT 100
        """, transfer_methods=transfer_names_list)

        for r in result:
            method = r['method']
            source = r['source_problem']
            target = r['target_problem']

            # Skip synonym pairs
            if is_synonym_pair(source, target):
                continue

            # Score based on method mentions and domain relevance
            score = r['method_mentions'] or 0

            # Boost if method is a transfer candidate
            if method.lower().replace(' ', '_') in TRANSFER_CANDIDATE_METHODS or \
               any(kw in method.lower() for kw in TRANSFER_CANDIDATE_METHODS):
                score *= 1.5

            # Boost if target is spatial-relevant
            if target in spatial_names or any(kw in target.lower() for kw in SPATIAL_KEYWORDS):
                score *= 2.0

            gaps.append({
                'method': method,
                'source_problem': source,
                'target_problem': target,
                'method_mentions': r['method_mentions'],
                'method_docs': r['method_docs'],
                'score': score,
                'is_spatial_target': target in spatial_names,
                'is_transfer_method': method in transfer_names
            })

        # Deduplicate by method-target pair
        seen = set()
        unique_gaps = []
        for g in gaps:
            key = (g['method'], g['target_problem'])
            if key not in seen:
                seen.add(key)
                unique_gaps.append(g)

        # Sort by score
        unique_gaps.sort(key=lambda x: -x['score'])

        print(f"\nFound {len(unique_gaps)} unique gap candidates")
        print("\nTop 20 gaps:")
        for i, g in enumerate(unique_gaps[:20]):
            spatial_flag = "🎯 SPATIAL" if g['is_spatial_target'] else ""
            transfer_flag = "⚡ TRANSFER" if g['is_transfer_method'] else ""
            print(f"\n{i+1}. {g['method']} (score: {g['score']:.0f}) {transfer_flag}")
            print(f"   SOLVES: {g['source_problem']}")
            print(f"   GAP →: {g['target_problem']} {spatial_flag}")

        return unique_gaps


def main():
    """Run A6 bucket gap detection."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

    try:
        # Step 1: Get spatial-relevant problems
        spatial_problems = get_spatial_problems(driver)

        # Step 2: Get transfer candidate methods
        transfer_methods = get_transfer_methods(driver)

        # Step 3: Build spatial similarity edges
        similarity_edges = build_spatial_similarity(driver, spatial_problems)

        # Add these edges to Neo4j
        if similarity_edges:
            print("\nAdding spatial similarity edges to Neo4j...")
            with driver.session() as session:
                batch = [{'p1': p1, 'p2': p2, 'sim': s} for p1, p2, s in similarity_edges]
                session.run("""
                    UNWIND $edges AS edge
                    MATCH (p1:PROBLEM {name: edge.p1})
                    MATCH (p2:PROBLEM {name: edge.p2})
                    MERGE (p1)-[r:SIMILAR_TO]->(p2)
                    SET r.similarity = edge.sim,
                        r.spatial_bucket = true,
                        r.updated_at = datetime()
                """, edges=batch)
            print(f"  Added {len(batch)} spatial similarity edges")

        # Step 4: Run focused gap detection
        gaps = run_focused_gap_detection(driver, spatial_problems, transfer_methods)

        # Save results
        output_dir = '/home/user/work/polymax/reports/bridge_mine_v4_a6_bucket'
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, 'A6_GAP_CANDIDATES.json')
        with open(output_file, 'w') as f:
            json.dump(gaps, f, indent=2)
        print(f"\nSaved {len(gaps)} gap candidates to {output_file}")

        # Generate summary report
        summary_file = os.path.join(output_dir, 'A6_SUMMARY.md')
        with open(summary_file, 'w') as f:
            f.write("# A6 Bucket Gap Detection: 3D Morphology / Spatial Reconstruction\n\n")
            f.write(f"**Generated**: {os.popen('date').read().strip()}\n\n")
            f.write("## Statistics\n\n")
            f.write(f"- Spatial problems identified: {len(spatial_problems)}\n")
            f.write(f"- Transfer candidate methods: {len(transfer_methods)}\n")
            f.write(f"- Spatial similarity edges: {len(similarity_edges)}\n")
            f.write(f"- Gap candidates found: {len(gaps)}\n\n")

            f.write("## Top 20 Gap Candidates\n\n")
            for i, g in enumerate(gaps[:20]):
                f.write(f"### {i+1}. {g['method']} → {g['target_problem']}\n\n")
                f.write(f"- **Score**: {g['score']:.0f}\n")
                f.write(f"- **Source problem**: {g['source_problem']}\n")
                f.write(f"- **Method mentions**: {g['method_mentions']}\n")
                f.write(f"- **Spatial target**: {'Yes' if g['is_spatial_target'] else 'No'}\n")
                f.write(f"- **Transfer method**: {'Yes' if g['is_transfer_method'] else 'No'}\n\n")

        print(f"Saved summary to {summary_file}")

    finally:
        driver.close()


if __name__ == "__main__":
    main()
