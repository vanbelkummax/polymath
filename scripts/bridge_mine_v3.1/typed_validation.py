#!/usr/bin/env python3
"""
Typed Validation for Bridge Mine v3.1

Replaces single SUPPORTS label with typed evidence:
- SOURCE_SUPPORT: Method works on source problem
- TARGET_SUPPORT: Target problem exists and is characterized
- ANALOGY_SUPPORT: Structural similarity justifying transfer
- GARBAGE: Irrelevant

A bridge "passes" if:
- ≥2 SOURCE_SUPPORT spans from ≥2 docs, AND
- ≥1 TARGET_SUPPORT span from ≥1 doc (or curated statement), AND
- ≥1 ANALOGY_SUPPORT rationale (can be LLM-generated but must be falsifiable)
"""

import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
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


class SupportType(str, Enum):
    SOURCE_SUPPORT = "SOURCE_SUPPORT"
    TARGET_SUPPORT = "TARGET_SUPPORT"
    ANALOGY_SUPPORT = "ANALOGY_SUPPORT"
    TANGENTIAL = "TANGENTIAL"
    GARBAGE = "GARBAGE"


@dataclass
class ValidationResult:
    """Result of validating a single span."""
    support_type: SupportType
    confidence: float  # 0.0-1.0
    rationale: str
    is_falsifiable: bool  # For ANALOGY_SUPPORT


@dataclass
class BridgeEvidence:
    """Aggregated evidence for a bridge."""
    bridge_id: str
    source_support: List[dict]  # Spans
    target_support: List[dict]
    analogy_support: List[dict]
    source_docs: set
    target_docs: set
    is_passing: bool
    failure_reason: Optional[str]


TYPED_VALIDATION_PROMPT = """You are an expert reviewer classifying evidence for a cross-domain research bridge.

BRIDGE: {source_concept} (from {source_domain}) → {target_concept} (spatial transcriptomics)

PASSAGE TO CLASSIFY:
"{snippet}"

CLASSIFICATION (choose ONE):

1. SOURCE_SUPPORT - Passage demonstrates that {source_concept} works on its native problem domain.
   The passage shows the SOURCE METHOD/TECHNIQUE is effective, but does NOT mention spatial transcriptomics.
   Example: "Learned primal-dual achieves state-of-the-art sparse-view CT reconstruction"

2. TARGET_SUPPORT - Passage characterizes the TARGET PROBLEM in spatial transcriptomics.
   Shows what {target_concept} means in spatial context, what challenges exist, why it matters.
   Does NOT need to mention the source method.
   Example: "3D reconstruction from serial sections remains challenging due to missing z-slices"

3. ANALOGY_SUPPORT - Passage explicitly connects the SOURCE and TARGET domains.
   Shows structural similarity, explains WHY the transfer should work, or cites prior cross-domain success.
   This is the rarest and most valuable type.
   Example: "The sparse-view reconstruction problem is mathematically analogous to missing z-slice imputation"

4. TANGENTIAL - Mentions one or both concepts but provides no clear support for the bridge.
   May be useful context but not evidence.

5. GARBAGE - Irrelevant, too generic, or doesn't relate to either concept.

Output format (JSON):
{{"type": "SOURCE_SUPPORT|TARGET_SUPPORT|ANALOGY_SUPPORT|TANGENTIAL|GARBAGE", "confidence": 0.0-1.0, "rationale": "one sentence explanation"}}"""


ANALOGY_GENERATION_PROMPT = """Generate a falsifiable analogy hypothesis for this cross-domain bridge:

SOURCE: {source_concept} - {source_description}
TARGET: {target_concept} - {target_description}
DOMAIN TARGET: Spatial transcriptomics

Based on the source and target descriptions, generate:
1. A structural analogy: What mathematical/computational structure is shared?
2. A falsifiable hypothesis: If X is true about source, then Y should be true about target
3. A quick falsification test: How could we check if this analogy fails?

Output JSON:
{{
  "structural_analogy": "The [X] in source maps to [Y] in target because...",
  "hypothesis": "If [source property], then [target prediction]",
  "falsification_test": "Run [experiment] - if [result], analogy fails",
  "confidence": 0.0-1.0
}}"""


def validate_span_typed(
    bridge_info: dict,
    snippet: str,
    max_retries: int = 3
) -> ValidationResult:
    """
    Classify a span with typed support labels.

    Args:
        bridge_info: Dict with source_concept, target_concept, bucket_description
        snippet: The text passage to classify

    Returns:
        ValidationResult with support type, confidence, and rationale
    """
    if model is None:
        return ValidationResult(
            support_type=SupportType.GARBAGE,
            confidence=0.0,
            rationale="No API configured",
            is_falsifiable=False
        )

    prompt = TYPED_VALIDATION_PROMPT.format(
        source_concept=bridge_info.get('source_concept', 'source'),
        source_domain=bridge_info.get('bucket_description', 'adjacent field'),
        target_concept=bridge_info.get('target_concept', 'target'),
        snippet=snippet[:1000]  # Truncate long snippets
    )

    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.0,
                    'max_output_tokens': 200,
                }
            )
            text = response.text.strip()

            # Parse JSON response
            # Handle markdown code blocks
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0].strip()
            elif '```' in text:
                text = text.split('```')[1].split('```')[0].strip()

            try:
                result = json.loads(text)
                support_type = SupportType(result.get('type', 'GARBAGE'))
                confidence = float(result.get('confidence', 0.5))
                rationale = result.get('rationale', '')

                return ValidationResult(
                    support_type=support_type,
                    confidence=confidence,
                    rationale=rationale,
                    is_falsifiable=(support_type == SupportType.ANALOGY_SUPPORT)
                )
            except (json.JSONDecodeError, ValueError):
                # Fallback: parse type from text
                text_upper = text.upper()
                for stype in SupportType:
                    if stype.value in text_upper:
                        return ValidationResult(
                            support_type=stype,
                            confidence=0.5,
                            rationale="Parsed from non-JSON response",
                            is_falsifiable=False
                        )

                return ValidationResult(
                    support_type=SupportType.GARBAGE,
                    confidence=0.3,
                    rationale="Could not parse response",
                    is_falsifiable=False
                )

        except Exception as e:
            error_str = str(e)
            if '429' in error_str or 'quota' in error_str.lower():
                wait_time = 10 * (2 ** attempt)
                print(f"    Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            elif attempt < max_retries - 1:
                time.sleep(2)
            else:
                print(f"    Validation error: {e}")
                return ValidationResult(
                    support_type=SupportType.GARBAGE,
                    confidence=0.0,
                    rationale=f"Error: {str(e)[:100]}",
                    is_falsifiable=False
                )

    return ValidationResult(
        support_type=SupportType.GARBAGE,
        confidence=0.0,
        rationale="Max retries exceeded",
        is_falsifiable=False
    )


def generate_analogy(bridge_info: dict) -> Optional[dict]:
    """
    Generate a falsifiable analogy hypothesis for a bridge.

    Used when no ANALOGY_SUPPORT spans are found in retrieval.
    """
    if model is None:
        return None

    prompt = ANALOGY_GENERATION_PROMPT.format(
        source_concept=bridge_info.get('source_concept', ''),
        source_description=bridge_info.get('source_aliases', [''])[0] if bridge_info.get('source_aliases') else '',
        target_concept=bridge_info.get('target_concept', ''),
        target_description=bridge_info.get('target_aliases', [''])[0] if bridge_info.get('target_aliases') else '',
    )

    try:
        response = model.generate_content(
            prompt,
            generation_config={
                'temperature': 0.3,
                'max_output_tokens': 300,
            }
        )
        text = response.text.strip()

        # Parse JSON
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0].strip()
        elif '```' in text:
            text = text.split('```')[1].split('```')[0].strip()

        return json.loads(text)

    except Exception as e:
        print(f"    Analogy generation error: {e}")
        return None


def evaluate_bridge(
    bridge_id: str,
    bridge_info: dict,
    evidence_records: List[dict]
) -> BridgeEvidence:
    """
    Evaluate all evidence for a bridge with typed validation.

    Returns BridgeEvidence with pass/fail status.
    """
    source_support = []
    target_support = []
    analogy_support = []
    source_docs = set()
    target_docs = set()

    for record in evidence_records:
        snippet = record.get('snippet', '')
        if not snippet or len(snippet) < 20:
            continue

        result = validate_span_typed(bridge_info, snippet)
        time.sleep(0.5)  # Rate limiting

        record['support_type'] = result.support_type.value
        record['confidence'] = result.confidence
        record['rationale'] = result.rationale

        doc_id = record.get('doc_id', '')

        if result.support_type == SupportType.SOURCE_SUPPORT:
            source_support.append(record)
            if doc_id:
                source_docs.add(doc_id)

        elif result.support_type == SupportType.TARGET_SUPPORT:
            target_support.append(record)
            if doc_id:
                target_docs.add(doc_id)

        elif result.support_type == SupportType.ANALOGY_SUPPORT:
            analogy_support.append(record)

    # Check passing criteria
    source_ok = len(source_support) >= 2 and len(source_docs) >= 2
    target_ok = len(target_support) >= 1
    analogy_ok = len(analogy_support) >= 1

    # If no analogy support found, try to generate one
    if not analogy_ok and source_ok and target_ok:
        generated = generate_analogy(bridge_info)
        if generated and generated.get('confidence', 0) >= 0.5:
            analogy_support.append({
                'type': 'generated',
                'structural_analogy': generated.get('structural_analogy', ''),
                'hypothesis': generated.get('hypothesis', ''),
                'falsification_test': generated.get('falsification_test', ''),
                'confidence': generated.get('confidence', 0.5)
            })
            analogy_ok = True

    # Determine pass/fail
    is_passing = source_ok and target_ok and analogy_ok

    failure_reasons = []
    if not source_ok:
        failure_reasons.append(f"SOURCE: {len(source_support)} spans from {len(source_docs)} docs (need ≥2 from ≥2)")
    if not target_ok:
        failure_reasons.append(f"TARGET: {len(target_support)} spans (need ≥1)")
    if not analogy_ok:
        failure_reasons.append("ANALOGY: No structural connection found or generated")

    return BridgeEvidence(
        bridge_id=bridge_id,
        source_support=source_support,
        target_support=target_support,
        analogy_support=analogy_support,
        source_docs=source_docs,
        target_docs=target_docs,
        is_passing=is_passing,
        failure_reason=" | ".join(failure_reasons) if failure_reasons else None
    )


def main():
    """Test typed validation on sample snippets."""

    test_bridge = {
        'source_concept': 'tomographic imaging',
        'target_concept': 'spatial reconstruction',
        'bucket_description': '3D morphology'
    }

    test_snippets = [
        # Should be SOURCE_SUPPORT
        "Learned primal-dual networks achieve state-of-the-art results on sparse-view CT reconstruction, outperforming traditional FBP by 3dB PSNR.",

        # Should be TARGET_SUPPORT
        "3D reconstruction from serial tissue sections remains challenging in spatial transcriptomics due to missing z-slices and section misalignment.",

        # Should be ANALOGY_SUPPORT
        "The mathematical structure of missing z-slice imputation is analogous to sparse-view tomographic reconstruction, both being ill-posed inverse problems with angular undersampling.",

        # Should be TANGENTIAL
        "Deep learning has been applied to many medical imaging tasks with varying success rates.",

        # Should be GARBAGE
        "The weather in Paris is nice this time of year.",
    ]

    print("Typed Validation Tests:\n")
    for snippet in test_snippets:
        result = validate_span_typed(test_bridge, snippet)
        print(f"SNIPPET: {snippet[:80]}...")
        print(f"  TYPE: {result.support_type.value}")
        print(f"  CONFIDENCE: {result.confidence}")
        print(f"  RATIONALE: {result.rationale}")
        print()
        time.sleep(1)

    print("\n\nAnalogy Generation Test:\n")
    analogy = generate_analogy(test_bridge)
    if analogy:
        print(json.dumps(analogy, indent=2))


if __name__ == "__main__":
    main()
