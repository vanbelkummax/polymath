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
    - Fast model (qwen2.5:3b) for quick extraction
    - Heavy model (deepseek-r1:8b) as fallback for complex texts
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
            fast_model: Quick model for simple extraction (default: qwen2.5:3b)
            heavy_model: Fallback model for complex texts (default: deepseek-r1:8b)
            timeout: Request timeout in seconds
            extractor_version: Version tag for provenance tracking
        """
        self.fast_model = fast_model or os.environ.get("LOCAL_LLM_FAST", "qwen2.5:3b")
        self.heavy_model = heavy_model or os.environ.get("LOCAL_LLM_HEAVY", "deepseek-r1:8b")
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
