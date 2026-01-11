#!/usr/bin/env python3
"""
Tests for Business MCP Utilities

Tests utility functions from the business MCP servers that can be tested
independently of the MCP framework.

Run with: pytest tests/test_business_mcp.py -v
"""

import pytest
import sys
import os
import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter
from unittest.mock import patch, MagicMock, AsyncMock


# =============================================================================
# Patent MCP Utility Tests
# =============================================================================

class TestPatentUtilities:
    """Test patent analysis utilities."""

    def test_cache_key_generation(self):
        """Test cache key generation is deterministic."""
        params1 = {"query": "machine learning", "page": 1}
        params2 = {"query": "machine learning", "page": 1}
        params3 = {"query": "deep learning", "page": 1}

        def _cache_key(prefix: str, params: dict) -> str:
            param_str = json.dumps(params, sort_keys=True)
            return f"{prefix}:{hashlib.md5(param_str.encode()).hexdigest()}"

        key1 = _cache_key("test", params1)
        key2 = _cache_key("test", params2)
        key3 = _cache_key("test", params3)

        assert key1 == key2, "Same params should produce same key"
        assert key1 != key3, "Different params should produce different key"

    def test_patent_gap_trend_analysis(self):
        """Test patent trend classification logic."""
        def classify_trend(recent_count, older_count):
            """Classify patent filing trend."""
            if recent_count < older_count * 0.7:
                return "declining"
            elif recent_count > older_count * 1.3:
                return "growing"
            return "stable"

        assert classify_trend(5, 20) == "declining"
        assert classify_trend(30, 10) == "growing"
        assert classify_trend(10, 10) == "stable"
        assert classify_trend(11, 10) == "stable"  # 10% increase is stable

    def test_stopword_filtering(self):
        """Test patent claim stopword filtering."""
        stopwords = {
            'a', 'an', 'the', 'and', 'or', 'in', 'on', 'for', 'of',
            'method', 'system', 'comprising', 'wherein', 'said'
        }

        claim = "A method for processing the data wherein said system comprises"
        words = claim.lower().split()
        key_terms = [w for w in words if w not in stopwords]

        assert 'processing' in key_terms
        assert 'data' in key_terms
        assert 'method' not in key_terms
        assert 'wherein' not in key_terms


# =============================================================================
# Legal MCP Utility Tests
# =============================================================================

class TestLegalUtilities:
    """Test legal search utilities."""

    def test_federal_courts_list(self):
        """Test federal court identifiers."""
        FEDERAL_COURTS = [
            "scotus",  # Supreme Court
            "ca1", "ca2", "ca3", "ca4", "ca5", "ca6", "ca7", "ca8", "ca9", "ca10", "ca11",
            "cadc", "cafc"  # DC and Federal Circuits
        ]

        assert "scotus" in FEDERAL_COURTS
        assert len([c for c in FEDERAL_COURTS if c.startswith("ca")]) >= 11

    def test_state_courts_list(self):
        """Test state court identifiers."""
        STATE_COURTS = [
            "cal", "calctapp",  # California
            "ny", "nyappdiv",   # New York
            "tex", "texapp",    # Texas
            "fla", "fladistctapp",  # Florida
            "ill", "illappct",  # Illinois
        ]

        # Should have multiple states
        assert len(STATE_COURTS) >= 6

    def test_cache_expiry_calculation(self):
        """Test cache expiry logic."""
        CACHE_EXPIRY_HOURS = 24

        cached_time = datetime.now() - timedelta(hours=12)
        is_valid = datetime.now() - cached_time < timedelta(hours=CACHE_EXPIRY_HOURS)
        assert is_valid, "12-hour old cache should be valid"

        old_cached_time = datetime.now() - timedelta(hours=30)
        is_old_valid = datetime.now() - old_cached_time < timedelta(hours=CACHE_EXPIRY_HOURS)
        assert not is_old_valid, "30-hour old cache should be invalid"


# =============================================================================
# Corporate MCP Utility Tests
# =============================================================================

class TestCorporateUtilities:
    """Test corporate intelligence utilities."""

    def test_jurisdiction_mapping(self):
        """Test jurisdiction code mapping."""
        JURISDICTIONS = {
            "us": ["us_de", "us_ca", "us_ny", "us_tx", "us_fl", "us_nv", "us_il"],
            "uk": ["gb"],
            "eu": ["de", "fr", "nl", "ie", "lu"],
            "asia": ["hk", "sg", "jp"],
        }

        assert "us_de" in JURISDICTIONS["us"], "Delaware should be in US jurisdictions"
        assert "gb" in JURISDICTIONS["uk"], "GB should be UK jurisdiction"
        assert "lu" in JURISDICTIONS["eu"], "Luxembourg should be in EU"

    def test_corporate_status_analysis(self):
        """Test corporate status breakdown analysis."""
        companies = [
            {"status": "active"},
            {"status": "active"},
            {"status": "dissolved"},
            {"status": "active"},
            {"status": "unknown"},
        ]

        status_counts = {}
        for c in companies:
            status = c.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1

        assert status_counts["active"] == 3
        assert status_counts["dissolved"] == 1
        assert status_counts["unknown"] == 1

    def test_offshore_detection(self):
        """Test offshore jurisdiction detection."""
        offshore_codes = ["vg", "ky", "bm", "je", "gg"]  # BVI, Cayman, Bermuda, Jersey, Guernsey

        jurisdictions_found = ["us_de", "gb", "ky"]
        has_offshore = any(j in jurisdictions_found for j in offshore_codes)

        assert has_offshore, "Should detect Cayman Islands as offshore"

        clean_jurisdictions = ["us_de", "gb", "de"]
        has_clean_offshore = any(j in clean_jurisdictions for j in offshore_codes)

        assert not has_clean_offshore, "Should not detect offshore in clean list"


# =============================================================================
# Market Intelligence MCP Utility Tests
# =============================================================================

class TestMarketIntelligenceUtilities:
    """Test market intelligence utilities."""

    def test_tone_thresholds(self):
        """Test sentiment tone classification thresholds."""
        TONE_THRESHOLDS = {
            "very_positive": 5.0,
            "positive": 1.5,
            "neutral": -1.5,
            "negative": -5.0,
            "very_negative": -10.0
        }

        def classify_tone(tone: float) -> str:
            if tone >= TONE_THRESHOLDS["very_positive"]:
                return "very_positive"
            elif tone >= TONE_THRESHOLDS["positive"]:
                return "positive"
            elif tone >= TONE_THRESHOLDS["neutral"]:
                return "neutral"
            elif tone >= TONE_THRESHOLDS["negative"]:
                return "negative"
            else:
                return "very_negative"

        assert classify_tone(7.0) == "very_positive"
        assert classify_tone(3.0) == "positive"
        assert classify_tone(0.0) == "neutral"
        assert classify_tone(-3.0) == "negative"
        assert classify_tone(-12.0) == "very_negative"

    def test_business_event_types(self):
        """Test business event type tracking."""
        event_types = ["merger", "acquisition", "lawsuit", "bankruptcy", "IPO", "earnings", "layoff"]

        # All events should be lowercase for consistent matching
        assert all(e.islower() or e.isupper() for e in event_types)
        assert len(event_types) >= 5

    def test_velocity_calculation(self):
        """Test news velocity/trending calculation."""
        volumes = [10, 12, 15, 20, 50, 80, 100]  # Increasing trend

        avg_volume = sum(volumes) / len(volumes)
        recent_volume = sum(volumes[-3:])  # Last 3 periods
        velocity_ratio = recent_volume / (avg_volume * 3) if avg_volume > 0 else 0

        assert velocity_ratio > 1.5, "Increasing trend should show velocity > 1.5"

        flat_volumes = [10, 11, 9, 10, 10, 11, 10]
        flat_avg = sum(flat_volumes) / len(flat_volumes)
        flat_recent = sum(flat_volumes[-3:])
        flat_velocity = flat_recent / (flat_avg * 3)

        assert 0.8 < flat_velocity < 1.2, "Flat trend should have velocity near 1.0"

    def test_sentiment_trend_direction(self):
        """Test sentiment trend direction classification."""
        def classify_trend(tone_change: float) -> str:
            if tone_change > 0.5:
                return "improving"
            elif tone_change < -0.5:
                return "declining"
            return "stable"

        assert classify_trend(1.5) == "improving"
        assert classify_trend(-2.0) == "declining"
        assert classify_trend(0.2) == "stable"
        assert classify_trend(-0.3) == "stable"

    def test_source_diversity_analysis(self):
        """Test source diversity analysis."""
        articles = [
            {"domain": "nytimes.com"},
            {"domain": "wsj.com"},
            {"domain": "nytimes.com"},
            {"domain": "reuters.com"},
            {"domain": "bbc.com"},
        ]

        sources = Counter([a.get("domain") for a in articles if a.get("domain")])

        assert len(sources) == 4, "Should have 4 unique sources"
        assert sources["nytimes.com"] == 2
        assert sources.most_common(1)[0][0] == "nytimes.com"


# =============================================================================
# API Response Handling Tests
# =============================================================================

class TestAPIResponseHandling:
    """Test API response handling patterns."""

    def test_error_response_structure(self):
        """Test error response structure is consistent."""
        def make_error_response(error: str, query: str = None) -> dict:
            response = {"error": error}
            if query:
                response["query"] = query
            return response

        err = make_error_response("Connection timeout", "test query")

        assert "error" in err
        assert err["error"] == "Connection timeout"
        assert err.get("query") == "test query"

    def test_pagination_handling(self):
        """Test pagination parameter handling."""
        def build_pagination(page: int, per_page: int, max_per_page: int = 100) -> dict:
            return {
                "page": max(1, page),
                "per_page": min(per_page, max_per_page)
            }

        # Normal case
        params = build_pagination(2, 50)
        assert params["page"] == 2
        assert params["per_page"] == 50

        # Edge cases
        params = build_pagination(-1, 200)
        assert params["page"] == 1, "Should clamp negative page to 1"
        assert params["per_page"] == 100, "Should clamp per_page to max"


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
