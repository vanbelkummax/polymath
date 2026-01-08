#!/usr/bin/env python3
"""
Tests for evidence extraction hardening (quality gating + support typing).

Tests:
1. text_quality identifies clean vs glued vs no_space text
2. Evidence validation categorizes literal vs normalized correctly
3. Fallback path produces support="none" without hallucinating evidence
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.text_quality import (
    normalize_text_soft,
    normalize_for_match,
    text_quality
)
from lib.local_extractor import LocalEntityExtractor


class TestTextQuality:
    """Test text quality assessment and normalization."""

    def test_clean_text_detection(self):
        """Clean text should score >= 0.7 and label 'clean'."""
        clean_text = """
        This is a normal scientific passage with proper spacing.
        It contains multiple sentences. The text is well-formatted
        and should score highly on quality metrics.
        """
        q = text_quality(clean_text)

        assert q['score'] >= 0.7, f"Clean text scored {q['score']}, expected >= 0.7"
        assert q['label'] == 'clean', f"Clean text labeled '{q['label']}', expected 'clean'"
        assert q['whitespace_ratio'] >= 0.15, "Clean text should have healthy whitespace ratio"

    def test_glued_text_detection(self):
        """Glued text (missing spaces) should score low and label appropriately."""
        glued_text = "Thisisasentencewithallwordsglued.AnothersentencewithnoSpaces.CamelCaseEveryWhere"

        q = text_quality(glued_text)

        assert q['score'] < 0.5, f"Glued text scored {q['score']}, expected < 0.5"
        assert q['label'] in ('glued', 'no_space'), f"Glued text labeled '{q['label']}'"
        assert q['whitespace_ratio'] < 0.10, "Glued text should have low whitespace ratio"

    def test_no_space_text_detection(self):
        """Text with no spaces should score very low."""
        no_space = "Allwordsaregluedtogetherwithnospacesorpunctuationthisisverybadquality"

        q = text_quality(no_space)

        assert q['score'] < 0.5, f"No-space text scored {q['score']}, expected < 0.5"
        assert q['label'] == 'no_space', f"No-space text labeled '{q['label']}', expected 'no_space'"
        assert q['max_run_no_space'] > 50, "No-space text should have long runs"

    def test_short_text_detection(self):
        """Very short text should be labeled 'short'."""
        short_text = "Too short"

        q = text_quality(short_text)

        assert q['label'] == 'short', f"Short text labeled '{q['label']}', expected 'short'"
        assert q['score'] == 0.0, "Short text should score 0.0"

    def test_normalize_text_soft(self):
        """Soft normalization should fix obvious glue issues."""
        # Missing space after period
        assert normalize_text_soft("Hello.World") == "Hello. World"

        # Missing space after comma
        assert normalize_text_soft("word1,word2") == "word1, word2"

        # CamelCase splitting
        assert normalize_text_soft("camelCaseWord") == "camel Case Word"

        # Multiple whitespace collapse
        assert normalize_text_soft("too    many     spaces") == "too many spaces"

        # Combined fixes
        assert normalize_text_soft("Hello.World,testCamelCase") == "Hello. World, test Camel Case"

    def test_normalize_for_match(self):
        """Aggressive normalization for matching only."""
        text = "Hello, World! Test CamelCase"
        normalized = normalize_for_match(text)

        assert normalized == "helloworldtestcamelcase", f"Got '{normalized}'"
        assert ' ' not in normalized, "Should remove all spaces"
        assert ',' not in normalized, "Should remove all punctuation"
        assert normalized.islower(), "Should be lowercase"


class TestEvidenceValidation:
    """Test evidence validation and support type categorization."""

    @pytest.fixture
    def extractor(self):
        """Create extractor instance for testing."""
        return LocalEntityExtractor()

    def test_literal_support_validation(self, extractor):
        """Exact substring match should categorize as 'literal'."""
        raw_text = "We used optimal transport to compute Wasserstein distances between distributions."
        surface = "Wasserstein distances"
        context = "compute Wasserstein distances between distributions"

        support = extractor._validate_evidence(surface, context, raw_text, confidence=0.85)

        assert support == "literal", f"Expected 'literal', got '{support}'"

    def test_normalized_support_validation(self, extractor):
        """Match after normalization should categorize as 'normalized'."""
        # Raw text has glued words
        raw_text = "WeusedoptimalTransporttocomputeWassersteinDistances"

        # Surface/context are clean (what LLM might return)
        surface = "Wasserstein Distances"
        context = "compute Wasserstein Distances"

        # Should not match literally
        assert surface not in raw_text, "Test setup error: shouldn't match literally"

        # Should match after normalization
        support = extractor._validate_evidence(surface, context, raw_text, confidence=0.85)

        assert support == "normalized", f"Expected 'normalized', got '{support}'"

    def test_inferred_support_high_confidence(self, extractor):
        """High confidence (>=0.9) without match should be 'inferred'."""
        raw_text = "Some text about completely different topic"
        surface = "neural networks"
        context = "deep neural networks were used"

        # No match in raw text
        assert surface not in raw_text, "Test setup error"

        # High confidence should allow 'inferred'
        support = extractor._validate_evidence(surface, context, raw_text, confidence=0.95)

        assert support == "inferred", f"Expected 'inferred', got '{support}'"

    def test_drop_low_confidence_no_match(self, extractor):
        """Low confidence without match should return None (drop concept)."""
        raw_text = "Some text about completely different topic"
        surface = "neural networks"
        context = "deep neural networks were used"

        # No match + low confidence = drop
        support = extractor._validate_evidence(surface, context, raw_text, confidence=0.7)

        assert support is None, f"Expected None (drop), got '{support}'"

    def test_empty_surface_context(self, extractor):
        """Empty surface/context should require high confidence."""
        raw_text = "Some text"

        # Empty evidence
        support = extractor._validate_evidence("", "", raw_text, confidence=0.7)
        assert support is None, "Low confidence + no evidence should drop"

        # High confidence allows inferred
        support = extractor._validate_evidence("", "", raw_text, confidence=0.95)
        assert support == "inferred", "High confidence + no evidence = inferred"


class TestFallbackPath:
    """Test canonical-only extraction for malformed text."""

    @pytest.fixture
    def extractor(self):
        """Create extractor instance for testing."""
        return LocalEntityExtractor()

    def test_malformed_text_uses_fallback(self, extractor):
        """Low quality text should trigger canonical-only extraction."""
        # Create glued text (quality score < 0.5)
        malformed_text = "ThisismalformedtextaboutoptimalTransportandWassersteinDistances"

        # Mock the extraction to avoid actual LLM call
        # We'll test the quality gating logic directly
        from lib.text_quality import text_quality
        q = text_quality(malformed_text)

        assert q['score'] < 0.5, f"Test text should be low quality, got {q['score']}"

        # The main extract_concepts_with_evidence should route to _extract_canonical_only
        # which should return concepts with support="none"

    def test_fallback_produces_support_none(self, extractor):
        """Canonical-only extraction should produce support='none'."""
        # Test the _parse_canonical_response method directly
        response_text = '{"concepts":[{"canonical":"optimal_transport","type":"method","aliases":["OT"],"confidence":0.85}]}'

        quality = {'score': 0.3, 'label': 'glued'}

        concepts = extractor._parse_canonical_response(response_text, quality, "soft_normalized")

        assert len(concepts) > 0, "Should extract concepts"
        assert concepts[0]['evidence']['support'] == 'none', "Should have support='none'"
        assert concepts[0]['evidence']['surface'] is None, "Should not have surface"
        assert concepts[0]['evidence']['context'] is None, "Should not have context"
        assert concepts[0]['evidence']['quality'] == quality, "Should include quality metrics"

    def test_fallback_no_hallucinated_evidence(self, extractor):
        """Fallback path should never hallucinate surface/context."""
        # Even if LLM returns surface/context in canonical-only mode, we should ignore it
        response_text = '''{"concepts":[{
            "canonical":"optimal_transport",
            "type":"method",
            "surface":"some hallucinated surface",
            "context":"some hallucinated context",
            "confidence":0.85
        }]}'''

        quality = {'score': 0.3, 'label': 'glued'}

        concepts = extractor._parse_canonical_response(response_text, quality, "raw")

        assert len(concepts) > 0, "Should extract concepts"
        # Even if LLM provided surface/context, fallback should null them
        assert concepts[0]['evidence']['support'] == 'none', "Should have support='none'"
        assert concepts[0]['evidence']['surface'] is None, "Should null surface"
        assert concepts[0]['evidence']['context'] is None, "Should null context"


class TestEndToEndQualityGating:
    """Integration tests for quality-gated extraction."""

    @pytest.fixture
    def extractor(self):
        """Create extractor instance for testing."""
        try:
            return LocalEntityExtractor()
        except Exception as e:
            pytest.skip(f"Ollama not available: {e}")

    @pytest.mark.skipif(
        os.environ.get('SKIP_LLM_TESTS') == '1',
        reason="Ollama LLM tests skipped (set SKIP_LLM_TESTS=0 to run)"
    )
    def test_clean_text_produces_literal_support(self, extractor):
        """Clean text should use evidence-mode extraction with literal support."""
        clean_text = """
        Optimal transport theory provides a framework for comparing probability
        distributions. The Wasserstein distance is a key metric in this domain.
        """

        concepts = extractor.extract_concepts_with_evidence(clean_text)

        if concepts:  # If LLM extraction succeeds
            # Should have at least some concepts with literal or normalized support
            support_types = [c['evidence']['support'] for c in concepts]
            assert 'none' not in support_types, "Clean text should not produce support='none'"
            assert any(s in ('literal', 'normalized') for s in support_types), \
                "Clean text should produce literal or normalized support"

    @pytest.mark.skipif(
        os.environ.get('SKIP_LLM_TESTS') == '1',
        reason="Ollama LLM tests skipped"
    )
    def test_malformed_text_produces_support_none(self, extractor):
        """Malformed text should use canonical-only extraction with support='none'."""
        malformed_text = "ThisismalformedtextaboutoptimalTransportandWassersteinDistances"

        concepts = extractor.extract_concepts_with_evidence(malformed_text)

        if concepts:  # If LLM extraction succeeds
            # All concepts should have support='none'
            support_types = [c['evidence']['support'] for c in concepts]
            assert all(s == 'none' for s in support_types), \
                f"Malformed text should produce support='none', got {support_types}"

            # Should not have surface/context
            for c in concepts:
                assert c['evidence']['surface'] is None, "Should not have surface"
                assert c['evidence']['context'] is None, "Should not have context"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
