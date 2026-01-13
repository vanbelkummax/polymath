#!/usr/bin/env python3
"""
A6 Bucket Gap Detection: 3D Morphology / Spatial Reconstruction
WITH LLM-AS-JUDGE for PROBLEM node validation

Improvements over v3.1:
1. Gemini Flash validates each PROBLEM node for spatial biology relevance
2. Structured JSON output ensures deterministic filtering
3. Hard constraints: only spatial_transcriptomics/digital_pathology domains pass
4. Rejects generic detection tasks from non-biological domains
"""

import os
import sys
import json
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
import time

sys.path.insert(0, '/home/user/polymath-repo')

import psycopg2
from neo4j import GraphDatabase
import numpy as np

# LLM judge integration
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    print("WARNING: google-generativeai not installed, falling back to keyword filtering")

# Database connections
PG_CONN = "dbname=polymath user=polymath"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "polymathic2026")

# A6 Bucket: 3D morphology / spatial reconstruction keywords (FALLBACK ONLY)
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
    # Image analysis - REMOVED 'detection' (too generic)
    'segmentation', 'localization', 'resolution',
    'super_resolution', 'upsampling', 'imputation',
}

# Methods from inverse problems / other domains that might transfer
TRANSFER_CANDIDATE_METHODS = {
    # Inverse problems
    'optimal_transport', 'wasserstein', 'sinkhorn',
    'compressed_sensing', 'sparse_coding', 'dictionary_learning',
    # Generative models
    'diffusion_models', 'ddpm', 'score_matching',
    'variational_autoencoder', 'vae', 'generative_adversarial_networks', 'gan',
    'normalizing_flows', 'flow_matching',
    # Computer vision
    'super_resolution', 'image_inpainting', 'denoising',
    'deformable_registration', 'optical_flow', 'depth_estimation',
    # Geometric methods
    'graph_neural_networks', 'point_cloud', 'mesh_processing',
}

@dataclass
class ProblemValidation:
    """LLM judgment on whether a PROBLEM is spatial biology relevant."""
    name: str
    domain: str  # spatial_transcriptomics, digital_pathology, general_bio, non_bio
    problem_family: str  # e.g., "cell_segmentation", "registration", "detection"
    is_biological: bool
    is_spatial_relevant: bool
    confidence: float  # 0-1
    reasoning: str


class GeminiProblemJudge:
    """Use Gemini Flash to validate PROBLEM nodes for spatial biology relevance."""

    def __init__(self, api_key: str):
        if not HAS_GEMINI:
            raise ImportError("google-generativeai not installed")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name='gemini-2.0-flash-exp',
            generation_config={
                'temperature': 0.0,  # Deterministic
                'response_mime_type': 'application/json',
                'response_schema': {
                    'type': 'object',
                    'properties': {
                        'domain': {'type': 'string'},
                        'problem_family': {'type': 'string'},
                        'is_biological': {'type': 'boolean'},
                        'is_spatial_relevant': {'type': 'boolean'},
                        'confidence': {'type': 'number'},
                        'reasoning': {'type': 'string'}
                    },
                    'required': ['domain', 'problem_family', 'is_biological', 'is_spatial_relevant', 'confidence', 'reasoning']
                }
            }
        )

        self.system_prompt = """You are an expert biomedical AI analyzing whether a research PROBLEM is relevant to spatial transcriptomics or digital pathology.

SPATIAL TRANSCRIPTOMICS includes:
- Technologies: Visium, Visium HD, Slide-seq, MERFISH, seqFISH, Xenium, CosMx, STARmap
- Problems: cell segmentation, spatial gene expression prediction, deconvolution, batch correction, tissue registration, H&E-to-gene mapping, spot/bin-to-cell assignment, spatial domain identification

DIGITAL PATHOLOGY includes:
- WSI analysis, H&E staining, tumor detection, tissue classification, morphology quantification, cell phenotyping

REJECT as non_bio:
- Cybersecurity (cyber_attack_detection, intrusion_detection)
- Infrastructure (crack_detection, defect_detection, fault_detection on machines/buildings)
- Environmental (deforestation_detection, accident_detection)
- AI alignment (safety_alignment, preference_alignment, value_alignment)
- Generic computer vision WITHOUT biological context

For each PROBLEM name, classify:
1. domain: spatial_transcriptomics | digital_pathology | general_bio | non_bio
2. problem_family: brief descriptor (e.g., "segmentation", "registration", "quality_control")
3. is_biological: true if related to biology/medicine
4. is_spatial_relevant: true if directly relevant to spatial transcriptomics or digital pathology
5. confidence: 0-1 score for your judgment
6. reasoning: 1-2 sentence explanation

Be strict: only mark is_spatial_relevant=true if the problem is DIRECTLY applicable to tissue imaging, spatial omics, or pathology."""

    def validate_problem(self, problem_name: str) -> ProblemValidation:
        """Validate a single PROBLEM node."""
        prompt = f"""Analyze this research PROBLEM for spatial biology relevance:

PROBLEM: {problem_name}

Return JSON with domain classification and relevance judgment."""

        try:
            response = self.model.generate_content([self.system_prompt, prompt])
            result = json.loads(response.text)

            return ProblemValidation(
                name=problem_name,
                domain=result['domain'],
                problem_family=result['problem_family'],
                is_biological=result['is_biological'],
                is_spatial_relevant=result['is_spatial_relevant'],
                confidence=result['confidence'],
                reasoning=result['reasoning']
            )
        except Exception as e:
            print(f"  ERROR validating {problem_name}: {e}")
            # Conservative fallback
            return ProblemValidation(
                name=problem_name,
                domain='non_bio',
                problem_family='unknown',
                is_biological=False,
                is_spatial_relevant=False,
                confidence=0.0,
                reasoning=f"Error during validation: {str(e)}"
            )

    def validate_batch(self, problem_names: List[str], delay: float = 0.1) -> List[ProblemValidation]:
        """Validate a batch of PROBLEM nodes with rate limiting."""
        results = []
        for i, name in enumerate(problem_names):
            if i > 0 and i % 10 == 0:
                print(f"  Validated {i}/{len(problem_names)} problems...")

            result = self.validate_problem(name)
            results.append(result)
            time.sleep(delay)  # Rate limiting

        return results


def get_spatial_problems_llm_judge(driver, api_key: Optional[str] = None) -> List[dict]:
    """Get PROBLEM nodes related to spatial/3D domains using LLM judge."""
    print("Finding spatial-relevant PROBLEM nodes (LLM judge mode)...")

    # Get API key
    if api_key is None:
        api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        print("  WARNING: No GEMINI_API_KEY found, falling back to keyword filtering")
        return get_spatial_problems_fallback(driver)

    # Initialize judge
    try:
        judge = GeminiProblemJudge(api_key)
    except ImportError:
        print("  WARNING: Gemini not available, falling back to keyword filtering")
        return get_spatial_problems_fallback(driver)

    with driver.session() as session:
        # Get all problems
        result = session.run("""
            MATCH (p:PROBLEM)
            WHERE p.mention_count >= 10
            RETURN p.name as name, p.mention_count as mentions
            ORDER BY mentions DESC
            LIMIT 500
        """)

        all_problems = [{'name': r['name'], 'mentions': r['mentions']} for r in result]
        print(f"  Loaded {len(all_problems)} PROBLEM nodes (>=10 mentions)")

        # Validate with LLM
        print(f"  Validating with Gemini Flash...")
        validations = judge.validate_batch([p['name'] for p in all_problems])

        # Filter by strict criteria
        spatial_problems = []
        rejected = []
        for prob, val in zip(all_problems, validations):
            # Hard constraint: must be spatial_relevant with confidence >= 0.8
            if val.is_spatial_relevant and val.confidence >= 0.8:
                spatial_problems.append({
                    'name': prob['name'],
                    'mentions': prob['mentions'],
                    'domain': val.domain,
                    'family': val.problem_family,
                    'confidence': val.confidence,
                    'reasoning': val.reasoning
                })
            else:
                rejected.append({
                    'name': prob['name'],
                    'domain': val.domain,
                    'is_bio': val.is_biological,
                    'confidence': val.confidence,
                    'reasoning': val.reasoning
                })

        print(f"  ✓ Accepted {len(spatial_problems)} spatial-relevant problems")
        print(f"  ✗ Rejected {len(rejected)} non-spatial problems")

        # Show top accepted
        print("\n  Top spatial problems:")
        for p in spatial_problems[:15]:
            print(f"    {p['name']} ({p['mentions']}) - {p['domain']} [{p['confidence']:.2f}]")

        # Show sample rejected
        print("\n  Sample rejected problems:")
        for r in rejected[:10]:
            print(f"    {r['name']} - {r['domain']} [{r['confidence']:.2f}] - {r['reasoning']}")

        # Save validation log
        output_dir = '/home/user/work/polymax/reports/bridge_mine_v4_a6_bucket/'
        os.makedirs(output_dir, exist_ok=True)

        with open(f'{output_dir}/problem_validation_log.json', 'w') as f:
            json.dump({
                'accepted': spatial_problems,
                'rejected': rejected,
                'total_evaluated': len(all_problems),
                'acceptance_rate': len(spatial_problems) / len(all_problems) if all_problems else 0
            }, f, indent=2)

        return spatial_problems


def get_spatial_problems_fallback(driver) -> List[dict]:
    """Fallback keyword-based filtering (original logic with 'detection' removed)."""
    print("Finding spatial-relevant PROBLEM nodes (keyword fallback)...")

    with driver.session() as session:
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


def is_synonym_pair(name1: str, name2: str) -> bool:
    """Check if two problem names are likely synonyms/plurals."""
    n1 = name1.lower().replace('_', ' ')
    n2 = name2.lower().replace('_', ' ')

    # Exact match
    if n1 == n2:
        return True

    # Plural/singular
    if n1.rstrip('s') == n2.rstrip('s'):
        return True

    # One is substring of other (e.g., "detection" vs "cancer_detection")
    if len(n1) > 5 and len(n2) > 5:
        if n1 in n2 or n2 in n1:
            return True

    return False


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


def detect_gaps(driver, spatial_problems: List[dict], transfer_methods: List[dict]) -> List[dict]:
    """Detect transfer gaps using Neo4j graph query."""
    print("\nDetecting transfer gaps...")

    # Extract names
    spatial_names = [p['name'] for p in spatial_problems]
    method_names = [m['name'] for m in transfer_methods]

    with driver.session() as session:
        result = session.run("""
            // Find methods solving similar problems across domains
            MATCH (m:METHOD)-[:SOLVES]->(p1:PROBLEM)-[:SIMILAR_TO]-(p2:PROBLEM)
            WHERE m.name IN $transfer_methods
              AND p2.name IN $spatial_problems
              AND NOT (m)-[:SOLVES]->(p2)

            WITH m, p1, p2,
                 m.mention_count as method_mentions,
                 p1.mention_count as source_mentions,
                 p2.mention_count as target_mentions

            RETURN m.name as method,
                   p1.name as source_problem,
                   p2.name as target_problem,
                   method_mentions,
                   source_mentions,
                   target_mentions
            ORDER BY method_mentions DESC, target_mentions DESC
            LIMIT 100
        """, transfer_methods=method_names, spatial_problems=spatial_names)

        gaps = []
        for r in result:
            # Skip synonym pairs
            if is_synonym_pair(r['source_problem'], r['target_problem']):
                continue

            gaps.append({
                'method': r['method'],
                'source_problem': r['source_problem'],
                'target_problem': r['target_problem'],
                'method_mentions': r['method_mentions'],
                'source_mentions': r['source_mentions'],
                'target_mentions': r['target_mentions'],
                'score': r['method_mentions'] + r['target_mentions']
            })

        print(f"  Found {len(gaps)} transfer gap candidates")
        return gaps


def main():
    """Run A6 bucket gap detection with LLM judge."""
    print("=" * 70)
    print("BridgeMine v4.1: A6 Bucket Gap Detection (LLM Judge)")
    print("=" * 70)

    # Connect to Neo4j
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

    try:
        # 1. Find spatial-relevant problems (LLM judge)
        api_key = os.environ.get('GEMINI_API_KEY')
        spatial_problems = get_spatial_problems_llm_judge(driver, api_key)

        if not spatial_problems:
            print("\nERROR: No spatial problems found")
            return

        # 2. Find transfer candidate methods
        transfer_methods = get_transfer_methods(driver)

        if not transfer_methods:
            print("\nERROR: No transfer methods found")
            return

        # 3. Detect gaps
        gaps = detect_gaps(driver, spatial_problems, transfer_methods)

        # 4. Save results
        output_dir = '/home/user/work/polymax/reports/bridge_mine_v4_a6_bucket/'
        os.makedirs(output_dir, exist_ok=True)

        output_file = f'{output_dir}/A6_GAP_CANDIDATES_LLM_JUDGE.json'
        with open(output_file, 'w') as f:
            json.dump({
                'spatial_problems': spatial_problems,
                'transfer_methods': transfer_methods,
                'gap_candidates': gaps,
                'metadata': {
                    'num_spatial_problems': len(spatial_problems),
                    'num_transfer_methods': len(transfer_methods),
                    'num_gaps': len(gaps),
                    'llm_judge': 'gemini-2.0-flash-exp',
                    'confidence_threshold': 0.8
                }
            }, f, indent=2)

        print(f"\n✓ Saved {len(gaps)} gap candidates to {output_file}")
        print(f"✓ Validation log saved to {output_dir}/problem_validation_log.json")

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Spatial problems: {len(spatial_problems)}")
        print(f"Transfer methods: {len(transfer_methods)}")
        print(f"Gap candidates: {len(gaps)}")

        if gaps:
            print("\nTop 5 gaps:")
            for i, gap in enumerate(gaps[:5], 1):
                print(f"{i}. {gap['method']} ({gap['method_mentions']}) "
                      f"for {gap['target_problem']} ({gap['target_mentions']})")

    finally:
        driver.close()


if __name__ == '__main__':
    main()
