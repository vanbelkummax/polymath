#!/usr/bin/env python3
"""
Local Entity Extractor v2 using Ollama
Typed concept extraction with confidence scores, aliases, and concept types.

Replaces regex-based extraction with LLM-powered entity recognition.
Returns structured JSON objects instead of plain strings.
"""

import os
import re
import json
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict

try:
    import ollama
except ImportError:
    ollama = None

# Import text quality utilities for robust extraction
try:
    from lib.text_quality import text_quality, normalize_text_soft, normalize_for_match
except ImportError:
    # Fallback if running from different directory
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from lib.text_quality import text_quality, normalize_text_soft, normalize_for_match

logger = logging.getLogger(__name__)


@dataclass
class ExtractedConcept:
    """Structured concept with metadata."""
    name: str
    type: str  # method, objective, prior, model, dataset, field, math_object, metric, domain
    aliases: List[str]
    confidence: float  # 0.0-1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LocalEntityExtractor:
    """
    Entity extractor using local Ollama models.
    Version 2: Returns typed concepts with confidence and aliases.

    Features:
    - Fast model (qwen2.5:7b-instruct) for quick extraction
    - Heavy model (qwen2.5:14b-instruct) as fallback for complex texts
    - Dirty JSON parsing to handle LLM chatter
    - Retry logic with model escalation
    - Type hints for concept categorization
    """

    VALID_TYPES = {
        "method", "objective", "prior", "model", "dataset",
        "field", "math_object", "metric", "domain", "algorithm",
        "architecture", "technique"
    }

    def __init__(
        self,
        fast_model: Optional[str] = None,
        heavy_model: Optional[str] = None,
        timeout: int = 60,
        extractor_version: str = "llm_v2"
    ):
        """Initialize the extractor with model configuration.

        Args:
            fast_model: Quick model for simple extraction (default: qwen2.5:7b-instruct)
            heavy_model: Fallback model for complex texts (default: qwen2.5:14b-instruct)
            timeout: Request timeout in seconds
            extractor_version: Version tag for provenance tracking
        """
        self.fast_model = fast_model or os.environ.get("LOCAL_LLM_FAST", "qwen2.5:7b-instruct")
        self.heavy_model = heavy_model or os.environ.get("LOCAL_LLM_HEAVY", "qwen2.5:14b-instruct")
        self.timeout = timeout
        self.extractor_version = extractor_version

        if ollama is None:
            logger.warning("ollama package not installed. Install with: pip install ollama")

    def _build_prompt(self, text: str) -> str:
        """Build the extraction prompt for typed concepts."""
        # Truncate very long texts (keep first 6k + last 2k chars)
        max_chars = 8000
        if len(text) > max_chars:
            text = text[:6000] + "\n...\n" + text[-2000:]

        return f"""Analyze this scientific text and extract 5-12 core concepts as structured JSON.

For each concept, provide:
- name: snake_case identifier (e.g., "spatial_transcriptomics")
- type: category from [method, objective, prior, model, dataset, field, math_object, metric, domain, algorithm, technique]
- aliases: list of alternative names/abbreviations (empty if none)
- confidence: your confidence score 0.0-1.0

Return ONLY a JSON object in this exact format (no markdown, no explanations):
{{"concepts":[{{"name":"optimal_transport","type":"method","aliases":["OT","Sinkhorn"],"confidence":0.85}},{{"name":"gene_expression","type":"domain","aliases":[],"confidence":0.92}}]}}

Text to analyze:
{text}

JSON object with concepts:"""

    def _dirty_json_parse(self, response: str) -> Optional[List[Dict[str, Any]]]:
        """Extract typed concept JSON from LLM response with chatter.

        Handles multiple common failure modes:
        1. Response is already clean JSON
        2. Response has markdown code fences (```json ... ```)
        3. Response has explanatory text before/after JSON
        4. Response has incomplete/malformed JSON

        Args:
            response: Raw LLM response text

        Returns:
            Parsed list of concept dicts, or None if parsing fails
        """
        if not response:
            return None

        # Try to clean the response
        cleaned = response.strip()

        # Remove markdown code fences
        cleaned = re.sub(r'```(?:json)?\s*', '', cleaned)
        cleaned = re.sub(r'```\s*$', '', cleaned)

        # Try multiple extraction strategies
        strategies = [
            # Strategy 1: Extract {"concepts": [...]}
            lambda s: re.search(r'\{"concepts"\s*:\s*\[.*?\]\}', s, re.DOTALL),
            # Strategy 2: Extract just the array [...]
            lambda s: re.search(r'\[.*?\]', s, re.DOTALL),
            # Strategy 3: Find first { to last }
            lambda s: re.search(r'\{.*\}', s, re.DOTALL),
        ]

        for strategy in strategies:
            match = strategy(cleaned)
            if not match:
                continue

            json_str = match.group(0)

            try:
                parsed = json.loads(json_str)

                # If we got {"concepts": [...]}, extract the list
                if isinstance(parsed, dict) and "concepts" in parsed:
                    parsed = parsed["concepts"]

                # If we got a bare list, wrap it
                if isinstance(parsed, list):
                    return self._validate_concepts(parsed)

                # If we got a single object, wrap it
                if isinstance(parsed, dict):
                    return self._validate_concepts([parsed])

            except json.JSONDecodeError:
                continue

        logger.debug(f"Failed all JSON parsing strategies on: {response[:200]}...")
        return None

    def _validate_concepts(self, concepts: List[Dict]) -> List[Dict[str, Any]]:
        """Validate and normalize concept dicts.

        Args:
            concepts: Raw list of dicts from LLM

        Returns:
            Validated and normalized concept dicts
        """
        validated = []

        for item in concepts:
            if not isinstance(item, dict):
                continue

            # Extract fields with defaults
            name = item.get("name", "")
            concept_type = item.get("type", "domain")
            aliases = item.get("aliases", [])
            confidence = item.get("confidence", 0.5)

            # Normalize name
            if isinstance(name, str) and name:
                name = name.lower().strip().replace(" ", "_").replace("-", "_")
                name = re.sub(r'[^a-z0-9_]', '', name)

                if len(name) <= 2:  # Skip very short
                    continue

                # Normalize type
                if concept_type not in self.VALID_TYPES:
                    concept_type = "domain"  # Default fallback

                # Normalize aliases
                if not isinstance(aliases, list):
                    aliases = []
                aliases = [str(a).strip() for a in aliases if a]

                # Normalize confidence
                try:
                    confidence = float(confidence)
                    confidence = max(0.0, min(1.0, confidence))
                except (ValueError, TypeError):
                    confidence = 0.5

                validated.append({
                    "name": name,
                    "type": concept_type,
                    "aliases": aliases,
                    "confidence": confidence
                })

        return validated

    def _call_ollama(self, text: str, model: str) -> Optional[str]:
        """Call Ollama API with the given model.

        Args:
            text: Text to analyze
            model: Model name to use

        Returns:
            Model response text, or None on error
        """
        if ollama is None:
            logger.error("ollama package not available")
            return None

        prompt = self._build_prompt(text)

        try:
            response = ollama.generate(
                model=model,
                prompt=prompt,
                options={
                    "temperature": 0.2,  # Low temp for consistent output
                    "num_predict": 512,  # Limit response length (was 500, increased for JSON)
                }
            )

            return response.get("response", "")

        except Exception as e:
            logger.warning(f"Ollama call failed with {model}: {e}")
            return None

    def extract_concepts(self, text: str) -> List[Dict[str, Any]]:
        """Extract typed concepts from text using local LLMs.

        Strategy:
        1. Try fast model first (2 attempts)
        2. Parse with dirty JSON parser
        3. On failure, escalate to heavy model (2 attempts)
        4. Return empty list on total failure (never crash)

        Args:
            text: Scientific text to analyze

        Returns:
            List of concept dicts with {name, type, aliases, confidence}
        """
        if not text or len(text.strip()) < 50:
            return []

        # Attempt 1-2: Fast model
        for attempt in range(2):
            logger.debug(f"Attempt {attempt+1}/2 with fast model: {self.fast_model}")
            response = self._call_ollama(text, self.fast_model)

            if response:
                concepts = self._dirty_json_parse(response)
                if concepts:
                    logger.debug(f"Fast model extracted {len(concepts)} concepts")
                    return concepts

        # Attempt 3-4: Heavy model fallback
        for attempt in range(2):
            logger.debug(f"Attempt {attempt+1}/2 with heavy model: {self.heavy_model}")
            response = self._call_ollama(text, self.heavy_model)

            if response:
                concepts = self._dirty_json_parse(response)
                if concepts:
                    logger.debug(f"Heavy model extracted {len(concepts)} concepts")
                    return concepts

        # Total failure - return empty list, don't crash
        logger.warning("All extraction attempts failed, returning empty list")
        return []

    def extract_concepts_batch(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> List[List[Dict[str, Any]]]:
        """Extract concepts from multiple texts.

        Args:
            texts: List of texts to analyze
            show_progress: Whether to show progress bar

        Returns:
            List of concept lists, one per input text
        """
        results = []

        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(texts, desc="Extracting concepts")
            except ImportError:
                iterator = texts
        else:
            iterator = texts

        for text in iterator:
            concepts = self.extract_concepts(text)
            results.append(concepts)

        return results

    def extract_concepts_with_evidence(self, text: str) -> List[Dict[str, Any]]:
        """Extract concepts WITH evidence binding and quality gating.

        Quality-gated extraction strategy:
        1. Assess text quality (malformed PDF detection)
        2. If low quality: Use canonical-only extraction with support="none"
        3. If clean: Use evidence-mode extraction with substring validation

        Returns concepts with embedded evidence dict:
        - canonical: normalized concept name
        - type, aliases, confidence
        - evidence: {surface, context, support, quality, source_text}

        Support types:
        - "literal": surface+context are exact substrings of raw text (AUDIT-GRADE)
        - "normalized": matches only after normalize_for_match()
        - "inferred": high-confidence LLM extraction without text evidence
        - "none": concept extracted but no verifiable evidence

        CRITICAL: Only "literal" support is citable in grants/papers.
        """
        # Truncate long text
        max_chars = 8000
        original_text = text
        if len(text) > max_chars:
            text = text[:6000] + "\n...\n" + text[-2000:]

        # Step 1: Assess text quality
        quality = text_quality(text)
        logger.debug(f"Text quality: {quality['label']} (score={quality['score']})")

        # Step 2: Quality gating - decide extraction strategy
        if quality['score'] < 0.5 or quality['label'] in ('no_space', 'glued'):
            # Low quality - use canonical-only extraction on soft-normalized text
            logger.info(f"Low quality text (score={quality['score']}), using canonical-only extraction")
            return self._extract_canonical_only(text, quality)
        else:
            # Clean text - use evidence-mode extraction on RAW text
            logger.debug("Clean text, using evidence-mode extraction")
            return self._extract_with_evidence_validation(text, quality)

    def _extract_canonical_only(
        self,
        text: str,
        quality: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Fallback extraction for malformed text.

        Uses normalize_text_soft() for discovery but marks all concepts as support="none"
        since we cannot verify exact substrings in malformed text.

        Args:
            text: Raw passage text (malformed)
            quality: Quality metrics from text_quality()

        Returns:
            List of concepts with evidence.support="none"
        """
        # Soft normalize for better LLM comprehension
        normalized = normalize_text_soft(text)

        prompt = f"""Extract scientific concepts from this text. Return ONLY canonical names.

Text:
{normalized}

Output JSON (no markdown):
{{"concepts":[{{"canonical":"optimal_transport","type":"method","aliases":["OT","Wasserstein"],"confidence":0.85}}]}}

JSON:"""

        # Try fast model
        try:
            response = ollama.chat(
                model=self.fast_model,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.1, 'num_predict': 2048}
            )
            response_text = response.message.content if hasattr(response, 'message') else ''
            concepts = self._parse_canonical_response(response_text, quality, source_text="soft_normalized")
            if concepts:
                return concepts
        except Exception as e:
            logger.warning(f"Canonical extraction failed (fast): {e}")

        # Fallback to heavy
        if self.heavy_model:
            try:
                response = ollama.chat(
                    model=self.heavy_model,
                    messages=[{'role': 'user', 'content': prompt}],
                    options={'temperature': 0.1, 'num_predict': 2048}
                )
                response_text = response.message.content if hasattr(response, 'message') else ''
                concepts = self._parse_canonical_response(response_text, quality, source_text="soft_normalized")
                if concepts:
                    return concepts
            except Exception as e:
                logger.warning(f"Canonical extraction failed (heavy): {e}")

        return []

    def _extract_with_evidence_validation(
        self,
        text: str,
        quality: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Evidence-mode extraction for clean text with strict substring validation.

        Requires LLM to return exact surface+context substrings from RAW text.
        Validates each concept and categorizes support type.

        Args:
            text: Raw passage text (clean)
            quality: Quality metrics from text_quality()

        Returns:
            List of concepts with validated evidence
        """
        prompt = f"""Extract scientific concepts from this text WITH literal evidence.

CRITICAL RULES:
1. Copy surface form EXACTLY as it appears in the text (preserve case/spacing/punctuation)
2. Copy context EXACTLY as a substring (10-30 words) containing the surface form
3. Both surface and context must be EXACT substrings - do not paraphrase
4. Normalize to canonical snake_case name
5. If surface differs from canonical, list it in aliases

Output JSON (no markdown, no explanations):
{{"concepts":[{{"canonical":"optimal_transport","surface":"Wasserstein distance","context":"computed Wasserstein distance between distributions","type":"method","aliases":["OT"],"confidence":0.85}}]}}

Text:
{text}

JSON:"""

        # Try fast model
        try:
            response = ollama.chat(
                model=self.fast_model,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.1, 'num_predict': 2048}
            )
            response_text = response.message.content if hasattr(response, 'message') else ''
            concepts = self._parse_evidence_response(response_text, text, quality)
            if concepts:
                return concepts
        except Exception as e:
            logger.warning(f"Evidence extraction failed (fast): {e}")

        # Fallback to heavy
        if self.heavy_model:
            try:
                response = ollama.chat(
                    model=self.heavy_model,
                    messages=[{'role': 'user', 'content': prompt}],
                    options={'temperature': 0.1, 'num_predict': 2048}
                )
                response_text = response.message.content if hasattr(response, 'message') else ''
                concepts = self._parse_evidence_response(response_text, text, quality)
                if concepts:
                    return concepts
            except Exception as e:
                logger.warning(f"Evidence extraction failed (heavy): {e}")

        logger.warning("All evidence extraction attempts failed")
        return []

    def _parse_canonical_response(
        self,
        response_text: str,
        quality: Dict[str, Any],
        source_text: str
    ) -> List[Dict[str, Any]]:
        """
        Parse canonical-only response (no evidence binding).

        All concepts get support="none" since we cannot verify substrings
        in malformed text.

        Args:
            response_text: LLM response
            quality: Quality metrics from text_quality()
            source_text: "raw" or "soft_normalized"

        Returns:
            List of concepts with evidence.support="none"
        """
        # Try to find JSON in response
        json_match = re.search(r'\{.*"concepts".*\}', response_text, re.DOTALL)
        if not json_match:
            return []

        try:
            data = json.loads(json_match.group(0))
            concepts = data.get('concepts', [])

            valid_concepts = []
            for c in concepts:
                if not isinstance(c, dict):
                    continue

                canonical = c.get('canonical', '').lower().replace(' ', '_')
                if not canonical:
                    continue

                concept_type = c.get('type', 'domain')
                if concept_type not in self.VALID_TYPES:
                    concept_type = 'domain'

                # Build concept with evidence.support="none"
                valid_concepts.append({
                    'canonical': canonical,
                    'type': concept_type,
                    'aliases': c.get('aliases', []),
                    'confidence': float(c.get('confidence', 0.7)),
                    'evidence': {
                        'surface': None,
                        'context': None,
                        'support': 'none',
                        'quality': quality,
                        'source_text': source_text
                    }
                })

            return valid_concepts
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Canonical parse failed: {e}")
            return []

    def _parse_evidence_response(
        self,
        response_text: str,
        raw_text: str,
        quality: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Parse evidence-bound response with strict substring validation.

        Categorizes support type:
        - "literal": surface+context are exact substrings of raw_text
        - "normalized": matches only after normalize_for_match()
        - "inferred": high-confidence LLM extraction without text match
        - Drop concept if confidence < 0.9 and no text match

        Args:
            response_text: LLM response
            raw_text: Original raw passage text
            quality: Quality metrics from text_quality()

        Returns:
            List of concepts with validated evidence
        """
        # Try to find JSON in response
        json_match = re.search(r'\{.*"concepts".*\}', response_text, re.DOTALL)
        if not json_match:
            return []

        try:
            data = json.loads(json_match.group(0))
            concepts = data.get('concepts', [])

            valid_concepts = []
            for c in concepts:
                if not isinstance(c, dict):
                    continue

                canonical = c.get('canonical', '').lower().replace(' ', '_')
                surface = c.get('surface', '')
                context = c.get('context', '')  # Changed from 'snippet' to match new contract

                if not canonical:
                    continue

                concept_type = c.get('type', 'domain')
                if concept_type not in self.VALID_TYPES:
                    concept_type = 'domain'

                confidence = float(c.get('confidence', 0.7))

                # Validate evidence and categorize support type
                support_type = self._validate_evidence(surface, context, raw_text, confidence)

                # Drop concepts without evidence unless high confidence
                if support_type is None:
                    logger.debug(f"Dropping concept '{canonical}' - no evidence and confidence < 0.9")
                    continue

                # Build concept with validated evidence
                valid_concepts.append({
                    'canonical': canonical,
                    'type': concept_type,
                    'aliases': c.get('aliases', []),
                    'confidence': confidence,
                    'evidence': {
                        'surface': surface if surface else None,
                        'context': context if context else None,
                        'support': support_type,
                        'quality': quality,
                        'source_text': 'raw'
                    }
                })

            return valid_concepts
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Evidence parse failed: {e}")
            return []

    def _validate_evidence(
        self,
        surface: str,
        context: str,
        raw_text: str,
        confidence: float
    ) -> Optional[str]:
        """
        Validate evidence and categorize support type.

        Returns:
            "literal" | "normalized" | "inferred" | None

        Rules:
        - "literal": surface in raw_text AND context in raw_text AND surface in context
        - "normalized": matches after normalize_for_match() but not literal
        - "inferred": confidence >= 0.9 but no text match
        - None: drop this concept (low confidence + no match)
        """
        # Check literal match (exact substring)
        if surface and context:
            if (surface in raw_text and
                context in raw_text and
                surface in context):
                return "literal"

            # Check normalized match (fuzzy)
            norm_surface = normalize_for_match(surface)
            norm_context = normalize_for_match(context)
            norm_raw = normalize_for_match(raw_text)

            if (norm_surface in norm_raw and
                norm_context in norm_raw and
                norm_surface in norm_context):
                return "normalized"

        # No text match - only keep if high confidence
        if confidence >= 0.9:
            return "inferred"

        # Drop concept - no evidence and low confidence
        return None


# Convenience function for one-off extraction
def extract_concepts(text: str) -> List[Dict[str, Any]]:
    """Extract typed concepts from text using default extractor.

    Args:
        text: Scientific text to analyze

    Returns:
        List of concept dicts with {name, type, aliases, confidence}
    """
    extractor = LocalEntityExtractor()
    return extractor.extract_concepts(text)


if __name__ == "__main__":
    # Test the extractor
    test_text = """
    This paper presents a novel approach to spatial transcriptomics analysis
    using graph neural networks. We leverage attention mechanisms to capture
    spatial patterns in gene expression data. Our method incorporates
    diffusion-based imputation to handle sparse measurements and uses
    contrastive learning for robust feature extraction. The variational
    inference framework with Gaussian process priors enables uncertainty
    quantification.
    """

    extractor = LocalEntityExtractor()
    concepts = extractor.extract_concepts(test_text)

    print("Extracted concepts (typed):")
    for c in concepts:
        print(f"  {c['name']} ({c['type']}, conf={c['confidence']:.2f}, aliases={c['aliases']})")
