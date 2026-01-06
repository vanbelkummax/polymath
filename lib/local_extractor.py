#!/usr/bin/env python3
"""
Local Entity Extractor using Ollama

Uses local LLMs (qwen3:4b, deepseek-r1:8b) for robust concept extraction.
Replaces regex-based extraction with LLM-powered entity recognition.
"""

import os
import re
import json
import logging
from typing import List, Optional

try:
    import ollama
except ImportError:
    ollama = None

logger = logging.getLogger(__name__)


class LocalEntityExtractor:
    """
    Entity extractor using local Ollama models.

    Features:
    - Fast model (qwen3:4b) for quick extraction
    - Heavy model (deepseek-r1:8b) as fallback for complex texts
    - Dirty JSON parsing to handle LLM chatter
    - Retry logic with model escalation
    """

    def __init__(
        self,
        fast_model: Optional[str] = None,
        heavy_model: Optional[str] = None,
        timeout: int = 60
    ):
        """Initialize the extractor with model configuration.

        Args:
            fast_model: Quick model for simple extraction (default: qwen3:4b)
            heavy_model: Fallback model for complex texts (default: deepseek-r1:8b)
            timeout: Request timeout in seconds
        """
        self.fast_model = fast_model or os.environ.get("LOCAL_LLM_FAST", "qwen3:4b")
        self.heavy_model = heavy_model or os.environ.get("LOCAL_LLM_HEAVY", "deepseek-r1:8b")
        self.timeout = timeout

        if ollama is None:
            logger.warning("ollama package not installed. Install with: pip install ollama")

    def _build_prompt(self, text: str) -> str:
        """Build the extraction prompt."""
        # Truncate very long texts to avoid context overflow
        max_chars = 4000
        if len(text) > max_chars:
            text = text[:max_chars] + "..."

        return f"""Analyze this scientific text. Identify 5-10 core concepts that represent the main ideas, methods, or domains discussed.

Rules:
- Normalize concepts to snake_case (e.g., "gene expression" → "gene_expression")
- Focus on scientific/technical concepts, not generic words
- Return ONLY a JSON list of strings, no explanation

Example output: ["entropy", "diffusion_coefficient", "neural_network", "spatial_transcriptomics"]

Text to analyze:
{text}

JSON list of concepts:"""

    def _dirty_json_parse(self, response: str) -> Optional[List[str]]:
        """Extract JSON array from LLM response with chatter.

        LLMs often add explanations before/after the JSON. This extracts
        just the array using regex.

        Args:
            response: Raw LLM response text

        Returns:
            Parsed list of strings, or None if parsing fails
        """
        if not response:
            return None

        # Try to find a JSON array in the response
        # Pattern matches [...] even with newlines inside
        match = re.search(r'\[.*?\]', response, re.DOTALL)

        if not match:
            logger.debug(f"No JSON array found in response: {response[:200]}...")
            return None

        json_str = match.group(0)

        try:
            parsed = json.loads(json_str)

            # Validate it's a list of strings
            if isinstance(parsed, list):
                # Filter and normalize to strings
                concepts = []
                for item in parsed:
                    if isinstance(item, str):
                        # Normalize: lowercase, strip, replace spaces with underscores
                        normalized = item.lower().strip().replace(" ", "_").replace("-", "_")
                        # Remove any non-alphanumeric except underscores
                        normalized = re.sub(r'[^a-z0-9_]', '', normalized)
                        if normalized and len(normalized) > 2:  # Skip very short concepts
                            concepts.append(normalized)
                return concepts

            return None

        except json.JSONDecodeError as e:
            logger.debug(f"JSON parse error: {e}")
            return None

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
                    "temperature": 0.1,  # Low temp for consistent output
                    "num_predict": 500,  # Limit response length
                }
            )

            return response.get("response", "")

        except Exception as e:
            logger.warning(f"Ollama call failed with {model}: {e}")
            return None

    def extract_concepts(self, text: str) -> List[str]:
        """Extract concepts from text using local LLMs.

        Strategy:
        1. Try fast model first
        2. Parse with dirty JSON parser
        3. On failure, retry fast model once
        4. On second failure, escalate to heavy model
        5. Return empty list on total failure (never crash)

        Args:
            text: Scientific text to analyze

        Returns:
            List of normalized concept strings (snake_case)
        """
        if not text or len(text.strip()) < 50:
            return []

        # Attempt 1: Fast model
        logger.debug(f"Attempting extraction with fast model: {self.fast_model}")
        response = self._call_ollama(text, self.fast_model)

        if response:
            concepts = self._dirty_json_parse(response)
            if concepts:
                logger.debug(f"Fast model extracted {len(concepts)} concepts")
                return concepts

        # Attempt 2: Retry fast model
        logger.debug("Retrying with fast model...")
        response = self._call_ollama(text, self.fast_model)

        if response:
            concepts = self._dirty_json_parse(response)
            if concepts:
                logger.debug(f"Fast model retry extracted {len(concepts)} concepts")
                return concepts

        # Attempt 3: Heavy model fallback
        logger.debug(f"Falling back to heavy model: {self.heavy_model}")
        response = self._call_ollama(text, self.heavy_model)

        if response:
            concepts = self._dirty_json_parse(response)
            if concepts:
                logger.debug(f"Heavy model extracted {len(concepts)} concepts")
                return concepts

        # Total failure - return empty list, don't crash
        logger.warning("All extraction attempts failed, returning empty list")
        return []

    def extract_concepts_batch(self, texts: List[str], show_progress: bool = True) -> List[List[str]]:
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
def extract_concepts(text: str) -> List[str]:
    """Extract concepts from text using default extractor.

    Args:
        text: Scientific text to analyze

    Returns:
        List of normalized concept strings
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
    contrastive learning for robust feature extraction.
    """

    extractor = LocalEntityExtractor()
    concepts = extractor.extract_concepts(test_text)
    print(f"Extracted concepts: {concepts}")
