#!/usr/bin/env python3
"""
Phase 1: Seed Bridge Intents for Bridge Mine v3
Creates ~60 candidate bridges across 8 problem-frame buckets.
"""

import json
import os
import sys
from pathlib import Path

# Add repo to path
sys.path.insert(0, '/home/user/polymath-repo')

from lib.hybrid_search_v2 import HybridSearcherV2
import psycopg2

OUTPUT_DIR = os.environ.get('OUTPUT_DIR', '/home/user/work/polymax/reports/bridge_mine_v3_20260112_2126')

# 8 problem-frame buckets for Hwang-style spatial omics
BUCKETS = {
    "A1_large_tissue_expansion": {
        "description": "Large-tissue expansion, virtual spatial inference, super-resolution",
        "seed_bridges": [
            ("image_super_resolution", "spatial_gene_imputation"),
            ("virtual_staining", "gene_expression_prediction"),
            ("diffusion_models", "spatial_transcriptomics_generation"),
            ("neural_radiance_fields", "3d_tissue_reconstruction"),
            ("image_inpainting", "missing_spot_imputation"),
            ("tile_stitching", "whole_slide_spatial"),
            ("compression_sensing", "sparse_spatial_sampling"),
            ("upscaling_networks", "visium_hd_prediction"),
        ]
    },
    "A2_roi_selection": {
        "description": "ROI selection, sampling, experimental design, active learning",
        "seed_bridges": [
            ("active_learning", "spatial_sampling_optimization"),
            ("bayesian_optimization", "experimental_design"),
            ("information_gain", "roi_prioritization"),
            ("uncertainty_sampling", "region_selection"),
            ("multi_armed_bandit", "adaptive_sampling"),
            ("optimal_transport", "sample_matching"),
            ("coverage_algorithms", "tissue_sampling"),
            ("reinforcement_learning", "sequential_roi_selection"),
        ]
    },
    "A3_multimodal_integration": {
        "description": "Multimodal integration, shared latent representations, cross-modality",
        "seed_bridges": [
            ("contrastive_learning", "image_gene_alignment"),
            ("variational_autoencoder", "multimodal_latent_space"),
            ("cross_attention", "histology_transcriptomics_fusion"),
            ("clip_models", "pathology_text_alignment"),
            ("canonical_correlation", "morphology_expression_mapping"),
            ("graph_neural_networks", "spatial_neighborhood_learning"),
            ("transformer_fusion", "multiomics_integration"),
            ("self_supervised_learning", "foundation_model_pretraining"),
        ]
    },
    "A4_harmonization": {
        "description": "Cross-platform normalization, batch effects, harmonization",
        "seed_bridges": [
            ("domain_adaptation", "cross_platform_transfer"),
            ("batch_correction", "spatial_harmonization"),
            ("style_transfer", "staining_normalization"),
            ("combat", "spatial_batch_effects"),
            ("scvi_harmonization", "technology_agnostic_embedding"),
            ("gan_normalization", "histology_standardization"),
            ("optimal_transport", "distribution_matching"),
            ("anchor_based_integration", "reference_mapping"),
        ]
    },
    "A5_uncertainty_calibration": {
        "description": "Uncertainty quantification, calibration, coverage guarantees, conformal",
        "seed_bridges": [
            ("conformal_prediction", "spatial_uncertainty"),
            ("bayesian_deep_learning", "prediction_intervals"),
            ("ensemble_methods", "uncertainty_quantification"),
            ("monte_carlo_dropout", "model_uncertainty"),
            ("calibration_methods", "clinical_reliability"),
            ("evidential_deep_learning", "epistemic_uncertainty"),
            ("selective_prediction", "abstention_strategies"),
            ("quantile_regression", "prediction_bounds"),
        ]
    },
    "A6_3d_morphology": {
        "description": "3D morphology, reconstruction, serial sections, z-stack",
        "seed_bridges": [
            ("3d_reconstruction", "serial_section_registration"),
            ("volume_rendering", "tissue_morphology"),
            ("holotomography", "subcellular_3d"),
            ("optical_flow", "section_alignment"),
            ("deformable_registration", "tissue_warping"),
            ("point_cloud_networks", "3d_cell_segmentation"),
            ("mesh_reconstruction", "organ_modeling"),
            ("tomographic_imaging", "spatial_reconstruction"),
        ]
    },
    "A7_clinical_translation": {
        "description": "Clinical translation, robustness, monitoring, deployment",
        "seed_bridges": [
            ("federated_learning", "multi_site_deployment"),
            ("domain_generalization", "clinical_robustness"),
            ("model_monitoring", "drift_detection"),
            ("explainability", "pathologist_trust"),
            ("clinical_validation", "biomarker_translation"),
            ("regulatory_frameworks", "ai_approval"),
            ("real_world_evidence", "clinical_utility"),
            ("outcome_prediction", "treatment_response"),
        ]
    },
    "A8_wildcard_serendipity": {
        "description": "Wildcard serendipity - unexpected cross-domain connections",
        "seed_bridges": [
            ("protein_language_models", "spatial_gene_context"),
            ("topological_data_analysis", "tissue_structure"),
            ("information_bottleneck", "spatial_compression"),
            ("causal_inference", "spatial_interventions"),
            ("physics_informed_nn", "diffusion_processes"),
            ("game_theory", "cell_competition"),
            ("network_motifs", "spatial_patterns"),
            ("fractal_geometry", "tumor_boundaries"),
        ]
    },
}


def query_neo4j_for_related_concepts(concept_name, limit=10):
    """Find related concepts in Neo4j that could form bridges."""
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'polymathic2026'))

    query = """
    MATCH (c:Concept {name: $concept})-[:MENTIONS]-(p:Passage)-[:MENTIONS]-(c2:Concept)
    WHERE c2.name <> c.name
    RETURN c2.name as related, count(*) as co_occurrences
    ORDER BY co_occurrences DESC
    LIMIT $limit
    """

    try:
        result = driver.execute_query(query, concept=concept_name, limit=limit)
        return [(r['related'], r['co_occurrences']) for r in result[0]]
    except Exception as e:
        return []
    finally:
        driver.close()


def search_concepts_in_bucket(bucket_key, bucket_info, searcher):
    """Use semantic search to find additional relevant concepts."""
    query = f"{bucket_info['description']} methods algorithms techniques"
    try:
        results = searcher.search_papers(query, n=10)
        concepts = set()
        for r in results:
            # Extract potential concept phrases from content
            content = r.content.lower()
            for phrase in ['method', 'algorithm', 'approach', 'framework', 'model', 'network']:
                if phrase in content:
                    concepts.add(f"{bucket_key}_semantic_hit")
                    break
        return list(concepts)[:5]
    except Exception as e:
        return []


def generate_bridge_candidates():
    """Generate all bridge candidates across 8 buckets."""
    print("Initializing HybridSearcher...")
    searcher = HybridSearcherV2()

    all_candidates = []
    bridge_id = 0

    for bucket_key, bucket_info in BUCKETS.items():
        print(f"\nProcessing bucket: {bucket_key}")

        # Add seed bridges
        for source, target in bucket_info["seed_bridges"]:
            bridge_id += 1
            candidate = {
                "bridge_id": f"B{bridge_id:03d}",
                "bucket": bucket_key,
                "bucket_description": bucket_info["description"],
                "source_concept": source,
                "target_concept": target,
                "origin": "seed",
            }
            all_candidates.append(candidate)
            print(f"  + {source} → {target}")

    print(f"\nTotal candidates: {len(all_candidates)}")
    return all_candidates


def main():
    output_file = Path(OUTPUT_DIR) / "INTERMEDIATE" / "BRIDGE_CANDIDATES_SEED.json"

    candidates = generate_bridge_candidates()

    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump({
            "total_candidates": len(candidates),
            "buckets": list(BUCKETS.keys()),
            "candidates": candidates
        }, f, indent=2)

    print(f"\n✓ Saved {len(candidates)} candidates to {output_file}")

    # Summary by bucket
    print("\nSummary by bucket:")
    from collections import Counter
    bucket_counts = Counter(c["bucket"] for c in candidates)
    for bucket, count in sorted(bucket_counts.items()):
        print(f"  {bucket}: {count}")

    return candidates


if __name__ == "__main__":
    main()
