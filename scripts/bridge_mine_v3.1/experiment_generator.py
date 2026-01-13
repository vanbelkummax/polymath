#!/usr/bin/env python3
"""
Experiment Generator for Bridge Mine v3.1

Enforces named specifications:
- Named baseline model (e.g., HisToGene, Map3D)
- Named adaptation target (what operator changes)
- Named primary metric + dataset (e.g., SSIM on Visium HD)

If generator cannot provide these → output NEEDS_HUMAN_SPEC, not a templated card.
"""

import json
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
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


class ExperimentStatus(str, Enum):
    FULLY_SPECIFIED = "FULLY_SPECIFIED"
    NEEDS_HUMAN_SPEC = "NEEDS_HUMAN_SPEC"
    GENERATION_FAILED = "GENERATION_FAILED"


@dataclass
class ExperimentCard:
    """A runnable experiment specification."""
    bridge_id: str
    title: str
    status: ExperimentStatus

    # Required named specifications
    baseline_model: str           # e.g., "HisToGene", "Map3D", "STAGATE"
    baseline_paper: Optional[str] # DOI or title if known
    proposed_method: str          # Named method being transferred
    adaptation_description: str   # What operator/component changes

    # Dataset
    dataset_name: str            # e.g., "Visium HD", "10x Xenium", "Slide-seq V2"
    dataset_source: str          # Where to get it (GEO accession, URL, etc.)
    subset_description: str      # Which subset to use

    # Metrics
    primary_metric: str          # e.g., "SSIM", "Pearson r", "F1 score"
    secondary_metrics: List[str]
    success_threshold: str       # e.g., "5% improvement over baseline"

    # Implementation
    estimated_compute: str       # e.g., "1x RTX 3090, 4 hours"
    implementation_steps: List[str]
    code_starting_point: str     # GitHub repo or file to modify

    # Falsifiability
    falsification_test: str      # How to know if the bridge hypothesis is wrong
    expected_failure_modes: List[str]

    # Confidence
    confidence_score: float      # 0-1, based on evidence strength
    confidence_rationale: str

    # Optional
    notes: str


EXPERIMENT_GENERATION_PROMPT = """Generate a detailed experiment card for testing this cross-domain bridge.

BRIDGE: {source_concept} → {target_concept}
EVIDENCE SUMMARY:
- {source_support_count} SOURCE_SUPPORT spans (method works on source domain)
- {target_support_count} TARGET_SUPPORT spans (target problem characterized)
- {analogy_support_count} ANALOGY_SUPPORT spans (transfer rationale)

ANALOGY HYPOTHESIS (if available):
{analogy_hypothesis}

KEY EVIDENCE SNIPPETS:
{evidence_snippets}

REQUIREMENTS - You must provide SPECIFIC NAMES, not templates:

1. BASELINE MODEL: Name a specific existing model/method to compare against.
   GOOD: "HisToGene (Pang et al., 2021)" or "STAGATE (Dong et al., 2022)"
   BAD: "existing spatial transcriptomics methods" or "baseline approach"

2. DATASET: Name a specific publicly available dataset.
   GOOD: "10x Genomics Visium HD Human Brain dataset (GEO: GSE123456)"
   BAD: "spatial transcriptomics dataset" or "publicly available data"

3. METRIC: Name a specific computable metric with success threshold.
   GOOD: "Pearson correlation of predicted vs actual expression, >0.7 baseline is 0.65"
   BAD: "prediction accuracy" or "performance metrics"

4. IMPLEMENTATION: Reference specific code/repos to modify.
   GOOD: "Modify encoder in github.com/author/repo/model.py line 120"
   BAD: "implement the method" or "add the technique"

If you CANNOT provide specific names for any of these, return:
{{"status": "NEEDS_HUMAN_SPEC", "missing": ["list of what needs human input"]}}

Otherwise, return a complete JSON experiment card:
{{
    "status": "FULLY_SPECIFIED",
    "title": "descriptive title",
    "baseline_model": "specific model name with citation",
    "baseline_paper": "DOI or paper title",
    "proposed_method": "name of method being transferred",
    "adaptation_description": "what changes in the architecture/pipeline",
    "dataset_name": "specific dataset name",
    "dataset_source": "GEO accession, URL, or access method",
    "subset_description": "which samples/subset to use",
    "primary_metric": "metric name",
    "secondary_metrics": ["list", "of", "metrics"],
    "success_threshold": "quantitative improvement needed",
    "estimated_compute": "hardware and time estimate",
    "implementation_steps": ["step 1", "step 2", "..."],
    "code_starting_point": "github URL or file path",
    "falsification_test": "how to know if hypothesis is wrong",
    "expected_failure_modes": ["mode 1", "mode 2"],
    "confidence_score": 0.0-1.0,
    "confidence_rationale": "why this confidence level",
    "notes": "additional context"
}}"""


# Known baselines and datasets for common spatial transcriptomics tasks
KNOWN_BASELINES = {
    'spatial_reconstruction': [
        {'name': 'HisToGene', 'paper': 'Pang et al., 2021', 'repo': 'https://github.com/OliverMR/HisToGene'},
        {'name': 'ST-Net', 'paper': 'He et al., 2020', 'repo': 'https://github.com/bryanhe/ST-Net'},
        {'name': 'Map3D', 'paper': 'Various', 'repo': 'N/A'},
    ],
    'cell_segmentation': [
        {'name': 'Cellpose', 'paper': 'Stringer et al., 2021', 'repo': 'https://github.com/MouseLand/cellpose'},
        {'name': 'StarDist', 'paper': 'Schmidt et al., 2018', 'repo': 'https://github.com/stardist/stardist'},
    ],
    'gene_expression_prediction': [
        {'name': 'HisToGene', 'paper': 'Pang et al., 2021', 'repo': 'https://github.com/OliverMR/HisToGene'},
        {'name': 'BLEEP', 'paper': 'Xie et al., 2023', 'repo': 'https://github.com/research/bleep'},
    ],
    'spatial_deconvolution': [
        {'name': 'RCTD', 'paper': 'Cable et al., 2022', 'repo': 'https://github.com/dmcable/spacexr'},
        {'name': 'Cell2location', 'paper': 'Kleshchevnikov et al., 2022', 'repo': 'https://github.com/BayraktarLab/cell2location'},
    ],
}

KNOWN_DATASETS = {
    'visium': [
        {'name': '10x Genomics Visium Human Brain', 'accession': 'Available at 10xgenomics.com', 'modality': 'Visium'},
        {'name': '10x Genomics Visium FFPE Human Breast Cancer', 'accession': 'Available at 10xgenomics.com', 'modality': 'Visium'},
    ],
    'visium_hd': [
        {'name': 'Visium HD Human CRC', 'accession': 'GEO: GSE280318', 'modality': 'Visium HD 2μm'},
        {'name': 'Visium HD Mouse Brain', 'accession': 'Available at 10xgenomics.com', 'modality': 'Visium HD'},
    ],
    'xenium': [
        {'name': '10x Xenium Human Breast Cancer', 'accession': 'Available at 10xgenomics.com', 'modality': 'Xenium'},
    ],
    'slide_seq': [
        {'name': 'Slide-seq V2 Mouse Cerebellum', 'accession': 'GEO: GSE127207', 'modality': 'Slide-seq V2'},
    ],
}


def get_relevant_baselines(target_concept: str) -> List[dict]:
    """Find known baselines relevant to target concept."""
    target_lower = target_concept.lower().replace('_', ' ')
    matches = []

    for key, baselines in KNOWN_BASELINES.items():
        if any(word in target_lower for word in key.split('_')):
            matches.extend(baselines)

    return matches


def get_relevant_datasets(context: str = '') -> List[dict]:
    """Find relevant datasets for spatial transcriptomics experiments."""
    all_datasets = []
    for datasets in KNOWN_DATASETS.values():
        all_datasets.extend(datasets)
    return all_datasets[:5]  # Return top 5


def generate_experiment_card(
    bridge_info: dict,
    evidence_summary: dict,
    evidence_snippets: List[str] = None
) -> ExperimentCard:
    """
    Generate a fully-specified experiment card for a bridge.

    Returns NEEDS_HUMAN_SPEC if LLM cannot provide named specifications.
    """
    bridge_id = bridge_info.get('bridge_id', '')
    source = bridge_info.get('source_concept', '')
    target = bridge_info.get('target_concept', '')

    # Get known baselines/datasets for context
    baselines = get_relevant_baselines(target)
    datasets = get_relevant_datasets()

    analogy = evidence_summary.get('analogy_hypothesis', 'No specific analogy retrieved')

    snippets_text = '\n'.join(evidence_snippets[:3]) if evidence_snippets else 'No evidence snippets available'

    prompt = EXPERIMENT_GENERATION_PROMPT.format(
        source_concept=source,
        target_concept=target,
        source_support_count=evidence_summary.get('source_support_count', 0),
        target_support_count=evidence_summary.get('target_support_count', 0),
        analogy_support_count=evidence_summary.get('analogy_support_count', 0),
        analogy_hypothesis=analogy,
        evidence_snippets=snippets_text
    )

    # Add known baselines/datasets as hints
    if baselines:
        prompt += f"\n\nKNOWN RELEVANT BASELINES:\n"
        for b in baselines[:3]:
            prompt += f"- {b['name']} ({b['paper']}): {b['repo']}\n"

    if datasets:
        prompt += f"\nKNOWN RELEVANT DATASETS:\n"
        for d in datasets[:3]:
            prompt += f"- {d['name']} ({d['accession']})\n"

    if model is None:
        return ExperimentCard(
            bridge_id=bridge_id,
            title=f"Experiment: {source} → {target}",
            status=ExperimentStatus.GENERATION_FAILED,
            baseline_model="",
            baseline_paper=None,
            proposed_method=source,
            adaptation_description="",
            dataset_name="",
            dataset_source="",
            subset_description="",
            primary_metric="",
            secondary_metrics=[],
            success_threshold="",
            estimated_compute="",
            implementation_steps=[],
            code_starting_point="",
            falsification_test="",
            expected_failure_modes=[],
            confidence_score=0.0,
            confidence_rationale="No API configured",
            notes="Generation failed - no API"
        )

    try:
        response = model.generate_content(
            prompt,
            generation_config={
                'temperature': 0.3,
                'max_output_tokens': 1500,
            }
        )
        text = response.text.strip()

        # Parse JSON
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0].strip()
        elif '```' in text:
            text = text.split('```')[1].split('```')[0].strip()

        result = json.loads(text)

        status = ExperimentStatus(result.get('status', 'NEEDS_HUMAN_SPEC'))

        if status == ExperimentStatus.NEEDS_HUMAN_SPEC:
            return ExperimentCard(
                bridge_id=bridge_id,
                title=f"INCOMPLETE: {source} → {target}",
                status=ExperimentStatus.NEEDS_HUMAN_SPEC,
                baseline_model="NEEDS_HUMAN_SPEC",
                baseline_paper=None,
                proposed_method=source,
                adaptation_description="",
                dataset_name="NEEDS_HUMAN_SPEC",
                dataset_source="",
                subset_description="",
                primary_metric="NEEDS_HUMAN_SPEC",
                secondary_metrics=[],
                success_threshold="",
                estimated_compute="",
                implementation_steps=[],
                code_starting_point="",
                falsification_test="",
                expected_failure_modes=[],
                confidence_score=0.0,
                confidence_rationale=f"Missing: {result.get('missing', ['unknown'])}",
                notes=f"LLM could not provide named specifications: {result.get('missing', [])}"
            )

        # Validate required fields have actual names (not templates)
        baseline = result.get('baseline_model', '')
        dataset = result.get('dataset_name', '')
        metric = result.get('primary_metric', '')

        template_phrases = ['existing', 'baseline', 'method', 'approach', 'dataset', 'available', 'metrics']

        def is_template(value: str) -> bool:
            value_lower = value.lower()
            return any(phrase in value_lower for phrase in template_phrases) and len(value.split()) < 4

        missing = []
        if not baseline or is_template(baseline):
            missing.append('baseline_model')
        if not dataset or is_template(dataset):
            missing.append('dataset_name')
        if not metric or is_template(metric):
            missing.append('primary_metric')

        if missing:
            return ExperimentCard(
                bridge_id=bridge_id,
                title=f"INCOMPLETE: {source} → {target}",
                status=ExperimentStatus.NEEDS_HUMAN_SPEC,
                baseline_model=baseline or "NEEDS_HUMAN_SPEC",
                baseline_paper=result.get('baseline_paper'),
                proposed_method=result.get('proposed_method', source),
                adaptation_description=result.get('adaptation_description', ''),
                dataset_name=dataset or "NEEDS_HUMAN_SPEC",
                dataset_source=result.get('dataset_source', ''),
                subset_description=result.get('subset_description', ''),
                primary_metric=metric or "NEEDS_HUMAN_SPEC",
                secondary_metrics=result.get('secondary_metrics', []),
                success_threshold=result.get('success_threshold', ''),
                estimated_compute=result.get('estimated_compute', ''),
                implementation_steps=result.get('implementation_steps', []),
                code_starting_point=result.get('code_starting_point', ''),
                falsification_test=result.get('falsification_test', ''),
                expected_failure_modes=result.get('expected_failure_modes', []),
                confidence_score=0.3,
                confidence_rationale=f"Template placeholders detected: {missing}",
                notes=f"LLM provided template placeholders instead of names: {missing}"
            )

        # Fully specified!
        return ExperimentCard(
            bridge_id=bridge_id,
            title=result.get('title', f"{source} → {target}"),
            status=ExperimentStatus.FULLY_SPECIFIED,
            baseline_model=baseline,
            baseline_paper=result.get('baseline_paper'),
            proposed_method=result.get('proposed_method', source),
            adaptation_description=result.get('adaptation_description', ''),
            dataset_name=dataset,
            dataset_source=result.get('dataset_source', ''),
            subset_description=result.get('subset_description', ''),
            primary_metric=metric,
            secondary_metrics=result.get('secondary_metrics', []),
            success_threshold=result.get('success_threshold', ''),
            estimated_compute=result.get('estimated_compute', ''),
            implementation_steps=result.get('implementation_steps', []),
            code_starting_point=result.get('code_starting_point', ''),
            falsification_test=result.get('falsification_test', ''),
            expected_failure_modes=result.get('expected_failure_modes', []),
            confidence_score=result.get('confidence_score', 0.7),
            confidence_rationale=result.get('confidence_rationale', ''),
            notes=result.get('notes', '')
        )

    except json.JSONDecodeError as e:
        return ExperimentCard(
            bridge_id=bridge_id,
            title=f"GENERATION FAILED: {source} → {target}",
            status=ExperimentStatus.GENERATION_FAILED,
            baseline_model="",
            baseline_paper=None,
            proposed_method=source,
            adaptation_description="",
            dataset_name="",
            dataset_source="",
            subset_description="",
            primary_metric="",
            secondary_metrics=[],
            success_threshold="",
            estimated_compute="",
            implementation_steps=[],
            code_starting_point="",
            falsification_test="",
            expected_failure_modes=[],
            confidence_score=0.0,
            confidence_rationale=f"JSON parse error: {str(e)[:100]}",
            notes=f"Could not parse LLM response"
        )

    except Exception as e:
        return ExperimentCard(
            bridge_id=bridge_id,
            title=f"GENERATION FAILED: {source} → {target}",
            status=ExperimentStatus.GENERATION_FAILED,
            baseline_model="",
            baseline_paper=None,
            proposed_method=source,
            adaptation_description="",
            dataset_name="",
            dataset_source="",
            subset_description="",
            primary_metric="",
            secondary_metrics=[],
            success_threshold="",
            estimated_compute="",
            implementation_steps=[],
            code_starting_point="",
            falsification_test="",
            expected_failure_modes=[],
            confidence_score=0.0,
            confidence_rationale=f"Error: {str(e)[:100]}",
            notes=f"Generation failed: {str(e)[:200]}"
        )


def save_experiment_card_yaml(card: ExperimentCard, output_dir: Path) -> Path:
    """Save experiment card as YAML."""
    import yaml

    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to dict
    card_dict = asdict(card)
    card_dict['status'] = card.status.value

    # Clean filename
    filename = f"{card.bridge_id[:50].replace('/', '_')}.yaml"
    filepath = output_dir / filename

    with open(filepath, 'w') as f:
        yaml.dump(card_dict, f, default_flow_style=False, sort_keys=False)

    return filepath


def main():
    """Test experiment generation."""

    test_bridge = {
        'bridge_id': 'B048_tomographic_spatial',
        'source_concept': 'tomographic imaging',
        'target_concept': 'spatial reconstruction'
    }

    test_evidence = {
        'source_support_count': 5,
        'target_support_count': 3,
        'analogy_support_count': 1,
        'analogy_hypothesis': 'Sparse-view CT reconstruction is mathematically analogous to missing z-slice imputation in serial section imaging.'
    }

    test_snippets = [
        "Learned primal-dual networks achieve state-of-the-art sparse-view CT reconstruction.",
        "3D reconstruction from serial sections remains challenging due to missing z-slices.",
        "The inverse problem formulation is similar in both domains."
    ]

    print("Generating experiment card...")
    card = generate_experiment_card(test_bridge, test_evidence, test_snippets)

    print(f"\nStatus: {card.status.value}")
    print(f"Title: {card.title}")
    print(f"Baseline: {card.baseline_model}")
    print(f"Dataset: {card.dataset_name}")
    print(f"Metric: {card.primary_metric}")
    print(f"Confidence: {card.confidence_score}")
    print(f"Rationale: {card.confidence_rationale}")

    if card.implementation_steps:
        print(f"\nImplementation steps:")
        for i, step in enumerate(card.implementation_steps[:5]):
            print(f"  {i+1}. {step}")

    if card.falsification_test:
        print(f"\nFalsification: {card.falsification_test}")

    # Save as YAML
    output_path = save_experiment_card_yaml(card, Path('/tmp/experiment_cards'))
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
