#!/usr/bin/env python3
"""
Concept-Type Gating for Bridge Mine v3.1

Classifies concepts as METHOD, PROBLEM, OUTCOME, DATASET, EVAL, CLINICAL_PROCESS
and rejects category-error bridges (e.g., OUTCOME → OUTCOME).

Valid bridges:
- METHOD → PROBLEM (primary)
- METHOD → OUTCOME (only if outcome has operational metric)
"""

import json
import os
import sys
import time
from typing import Dict, List, Tuple, Optional
from enum import Enum

sys.path.insert(0, '/home/user/polymath-repo')

import google.generativeai as genai

# Load API key
API_KEY = os.environ.get('GEMINI_API_KEY')
if not API_KEY:
    try:
        with open('/home/user/polymath-repo/.env', 'r') as f:
            for line in f:
                if line.startswith('GEMINI_API_KEY='):
                    API_KEY = line.strip().split('=', 1)[1]
                    break
    except:
        pass

if API_KEY:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash')
else:
    model = None


class ConceptType(str, Enum):
    METHOD = "METHOD"           # Technique, algorithm, approach (e.g., "tomographic imaging")
    PROBLEM = "PROBLEM"         # Challenge to solve (e.g., "sparse view reconstruction")
    OUTCOME = "OUTCOME"         # Desired result (e.g., "clinical utility")
    DATASET = "DATASET"         # Data resource (e.g., "TCGA", "Visium HD")
    EVAL = "EVAL"               # Evaluation metric/protocol (e.g., "SSIM", "F1 score")
    CLINICAL_PROCESS = "CLINICAL_PROCESS"  # Clinical workflow (e.g., "validation", "translation")
    UNKNOWN = "UNKNOWN"


TYPE_CLASSIFICATION_PROMPT = """Classify this scientific concept into ONE of these categories:

CONCEPT: {concept}
CONTEXT (optional): {context}

CATEGORIES:
- METHOD: A technique, algorithm, model, or approach that can be applied to solve problems
  Examples: "deep learning", "tomographic imaging", "contrastive learning", "deconvolution"

- PROBLEM: A scientific/engineering challenge or task to be solved
  Examples: "sparse view reconstruction", "cell segmentation", "gene expression prediction"

- OUTCOME: A high-level goal or desired result (often abstract, non-operational)
  Examples: "clinical utility", "biomarker translation", "diagnostic accuracy"

- DATASET: A named data resource or benchmark
  Examples: "TCGA", "Visium HD", "ImageNet", "AAPM Low-Dose CT Challenge"

- EVAL: An evaluation metric or assessment protocol
  Examples: "SSIM", "PSNR", "F1 score", "AUC-ROC", "Pearson correlation"

- CLINICAL_PROCESS: A step in clinical research workflow
  Examples: "clinical validation", "regulatory approval", "trial design"

Output ONLY the category name (METHOD, PROBLEM, OUTCOME, DATASET, EVAL, or CLINICAL_PROCESS).
If unclear, output UNKNOWN."""


# Known type mappings for common concepts (fast path)
KNOWN_TYPES: Dict[str, ConceptType] = {
    # Methods
    "tomographic_imaging": ConceptType.METHOD,
    "learned_primal_dual": ConceptType.METHOD,
    "contrastive_learning": ConceptType.METHOD,
    "deconvolution": ConceptType.METHOD,
    "super_resolution": ConceptType.METHOD,
    "attention_mechanism": ConceptType.METHOD,
    "graph_neural_network": ConceptType.METHOD,
    "transformer": ConceptType.METHOD,
    "diffusion_model": ConceptType.METHOD,
    "variational_autoencoder": ConceptType.METHOD,
    "self_supervised_learning": ConceptType.METHOD,
    "foundation_model": ConceptType.METHOD,
    "multimodal_learning": ConceptType.METHOD,

    # Problems
    "spatial_reconstruction": ConceptType.PROBLEM,
    "cell_segmentation": ConceptType.PROBLEM,
    "gene_expression_prediction": ConceptType.PROBLEM,
    "sparse_view_reconstruction": ConceptType.PROBLEM,
    "registration": ConceptType.PROBLEM,
    "denoising": ConceptType.PROBLEM,
    "imputation": ConceptType.PROBLEM,
    "cell_type_annotation": ConceptType.PROBLEM,
    "spatial_deconvolution": ConceptType.PROBLEM,
    "tissue_classification": ConceptType.PROBLEM,

    # Outcomes (often category-error sources)
    "clinical_utility": ConceptType.OUTCOME,
    "biomarker_translation": ConceptType.OUTCOME,
    "diagnostic_accuracy": ConceptType.OUTCOME,
    "therapeutic_response": ConceptType.OUTCOME,
    "prognostic_value": ConceptType.OUTCOME,

    # Clinical processes
    "clinical_validation": ConceptType.CLINICAL_PROCESS,
    "regulatory_approval": ConceptType.CLINICAL_PROCESS,
    "real_world_evidence": ConceptType.CLINICAL_PROCESS,
    "biomarker_discovery": ConceptType.CLINICAL_PROCESS,

    # Datasets
    "visium_hd": ConceptType.DATASET,
    "tcga": ConceptType.DATASET,
    "spatial_transcriptomics_atlas": ConceptType.DATASET,

    # Eval
    "ssim": ConceptType.EVAL,
    "psnr": ConceptType.EVAL,
    "pearson_correlation": ConceptType.EVAL,
}


def normalize_concept(concept: str) -> str:
    """Normalize concept string for lookup."""
    return concept.lower().replace(' ', '_').replace('-', '_')


def classify_concept(concept: str, context: str = "") -> ConceptType:
    """
    Classify a concept into one of the defined types.
    Uses known mappings first, falls back to Gemini for unknowns.
    """
    normalized = normalize_concept(concept)

    # Fast path: known types
    if normalized in KNOWN_TYPES:
        return KNOWN_TYPES[normalized]

    # LLM classification
    if model is None:
        return ConceptType.UNKNOWN

    prompt = TYPE_CLASSIFICATION_PROMPT.format(
        concept=concept,
        context=context if context else "spatial transcriptomics research"
    )

    try:
        response = model.generate_content(
            prompt,
            generation_config={
                'temperature': 0.0,
                'max_output_tokens': 20,
            }
        )
        result = response.text.strip().upper()

        for ctype in ConceptType:
            if ctype.value in result:
                # Cache for future use
                KNOWN_TYPES[normalized] = ctype
                return ctype

        # Default to UNKNOWN - forces manual review/lower ranking
        # (Changed from METHOD to prevent garbage sneaking through as valid bridges)
        return ConceptType.UNKNOWN

    except Exception as e:
        print(f"  Type classification error for '{concept}': {e}")
        return ConceptType.UNKNOWN


def is_valid_bridge(source_type: ConceptType, target_type: ConceptType) -> Tuple[bool, str]:
    """
    Check if a source→target bridge is valid based on concept types.

    Returns:
        (is_valid, reason)
    """
    # METHOD → PROBLEM is the gold standard
    if source_type == ConceptType.METHOD and target_type == ConceptType.PROBLEM:
        return True, "Valid METHOD→PROBLEM bridge"

    # METHOD → OUTCOME is OK if outcome is measurable
    if source_type == ConceptType.METHOD and target_type == ConceptType.OUTCOME:
        return True, "METHOD→OUTCOME bridge (requires operational metric)"

    # METHOD → EVAL is valid (applying method for evaluation)
    if source_type == ConceptType.METHOD and target_type == ConceptType.EVAL:
        return True, "Valid METHOD→EVAL bridge"

    # PROBLEM → METHOD is reverse (literature review style)
    if source_type == ConceptType.PROBLEM and target_type == ConceptType.METHOD:
        return True, "PROBLEM→METHOD bridge (technique discovery)"

    # Category errors
    if source_type == ConceptType.OUTCOME and target_type == ConceptType.OUTCOME:
        return False, "REJECTED: OUTCOME→OUTCOME is tautological"

    if source_type == ConceptType.CLINICAL_PROCESS and target_type == ConceptType.OUTCOME:
        return False, "REJECTED: CLINICAL_PROCESS→OUTCOME is tautological (e.g., 'validation → clinical utility')"

    if source_type == ConceptType.OUTCOME and target_type == ConceptType.CLINICAL_PROCESS:
        return False, "REJECTED: OUTCOME→CLINICAL_PROCESS is tautological"

    # DATASET sources are strange
    if source_type == ConceptType.DATASET:
        return False, "REJECTED: DATASET should not be source of bridge (can be target for benchmarking)"

    # Unknown types: allow with warning
    if source_type == ConceptType.UNKNOWN or target_type == ConceptType.UNKNOWN:
        return True, "WARNING: Contains UNKNOWN type, manual review recommended"

    # Default: allow other combinations but flag
    return True, f"ALLOWED: {source_type.value}→{target_type.value} (non-standard)"


def gate_bridges(bridges: List[Dict]) -> List[Dict]:
    """
    Filter bridges through concept-type gating.
    Adds 'source_type', 'target_type', 'gate_status', 'gate_reason' fields.
    """
    gated = []

    for bridge in bridges:
        source = bridge.get('source_concept', '')
        target = bridge.get('target_concept', '')

        source_type = classify_concept(source, bridge.get('bucket_description', ''))
        time.sleep(0.1)  # Rate limiting
        target_type = classify_concept(target, bridge.get('bucket_description', ''))
        time.sleep(0.1)

        is_valid, reason = is_valid_bridge(source_type, target_type)

        bridge['source_type'] = source_type.value
        bridge['target_type'] = target_type.value
        bridge['gate_status'] = 'PASS' if is_valid else 'REJECT'
        bridge['gate_reason'] = reason

        gated.append(bridge)

    return gated


def main():
    """Test concept type classification."""
    test_concepts = [
        "tomographic imaging",
        "spatial reconstruction",
        "clinical utility",
        "real world evidence",
        "deep learning",
        "gene expression prediction",
        "SSIM",
        "Visium HD",
    ]

    print("Concept Type Classification Tests:\n")
    for concept in test_concepts:
        ctype = classify_concept(concept)
        print(f"  {concept} → {ctype.value}")

    print("\n\nBridge Validity Tests:\n")
    test_bridges = [
        ("tomographic_imaging", "spatial_reconstruction"),  # METHOD → PROBLEM ✓
        ("real_world_evidence", "clinical_utility"),        # CLINICAL_PROCESS → OUTCOME ✗
        ("deep_learning", "cell_segmentation"),             # METHOD → PROBLEM ✓
        ("clinical_utility", "biomarker_translation"),      # OUTCOME → OUTCOME ✗
    ]

    for source, target in test_bridges:
        s_type = classify_concept(source)
        t_type = classify_concept(target)
        is_valid, reason = is_valid_bridge(s_type, t_type)
        status = "✓" if is_valid else "✗"
        print(f"  {source} → {target}")
        print(f"    Types: {s_type.value} → {t_type.value}")
        print(f"    {status} {reason}\n")


if __name__ == "__main__":
    main()
