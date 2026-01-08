"""
Text quality assessment and normalization utilities for Polymath.

Handles malformed PDF extractions (glued words, missing spaces) while preserving
audit-grade citation integrity. Two modes:
1. Soft normalization: fixes obvious issues deterministically for discovery
2. Match normalization: aggressive normalization for fuzzy matching only

NEVER use normalized text for citations - only for concept extraction.
"""

import re
import string
from typing import Dict, Any


def normalize_text_soft(text: str) -> str:
    """
    Apply conservative, deterministic fixes to malformed text.

    Safe for concept extraction but NEVER for citations.

    Fixes:
    - Insert space after sentence-ending punctuation when followed by letter/digit
    - Insert space after commas when followed by letter/digit
    - Insert space at camelCase boundaries (cautiously)
    - Collapse repeated whitespace

    Args:
        text: Raw passage text (potentially malformed)

    Returns:
        Softly normalized text

    Examples:
        >>> normalize_text_soft("Hello.World,test")
        'Hello. World, test'
        >>> normalize_text_soft("camelCaseWord")
        'camel Case Word'
    """
    if not text:
        return text

    result = text

    # Insert space after sentence-ending punctuation when followed by alphanumeric
    # Pattern: [.!?;:] followed directly by [a-zA-Z0-9]
    result = re.sub(r'([.!?;:])([a-zA-Z0-9])', r'\1 \2', result)

    # Insert space after comma when followed by alphanumeric
    result = re.sub(r',([a-zA-Z0-9])', r', \1', result)

    # Cautiously split camelCase: lowercase followed by uppercase
    # Only do this for likely word boundaries (not acronyms like "XMLParser")
    # Pattern: lowercase letter followed by uppercase letter
    result = re.sub(r'([a-z])([A-Z])', r'\1 \2', result)

    # Collapse repeated whitespace (spaces, tabs, newlines) into single space
    result = re.sub(r'\s+', ' ', result)

    # Strip leading/trailing whitespace
    result = result.strip()

    return result


def normalize_for_match(text: str) -> str:
    """
    Aggressive normalization for fuzzy substring matching ONLY.

    NEVER use this for citations or display - matching purposes only.

    Transforms:
    - Lowercase
    - Remove ALL whitespace
    - Remove ALL punctuation

    Args:
        text: Any text string

    Returns:
        Normalized text for matching

    Examples:
        >>> normalize_for_match("Hello, World!")
        'helloworld'
        >>> normalize_for_match("camelCase Word")
        'camelcaseword'
    """
    if not text:
        return ""

    # Lowercase
    result = text.lower()

    # Remove all whitespace
    result = re.sub(r'\s+', '', result)

    # Remove all punctuation
    result = result.translate(str.maketrans('', '', string.punctuation))

    return result


def text_quality(text: str) -> Dict[str, Any]:
    """
    Assess text quality to determine if it's suitable for literal citation.

    Metrics:
    - whitespace_ratio: fraction of chars that are whitespace (healthy: 0.15-0.25)
    - max_run_no_space: longest sequence of non-whitespace chars (healthy: <50)
    - camel_rate: fraction of lowercase->uppercase transitions (healthy: <0.05)
    - avg_word_len: average word length (healthy: 4-8)

    Score calculation:
    - Start at 1.0
    - Penalize low whitespace ratio
    - Penalize long runs without spaces
    - Penalize high camelCase rate
    - Penalize very short/long avg word length

    Labels:
    - "clean": score >= 0.7
    - "glued": high camel_rate or long no_space runs
    - "no_space": very low whitespace ratio
    - "short": text < 20 chars

    Args:
        text: Raw passage text

    Returns:
        Dict with keys: score (float 0-1), label (str), metrics (dict)

    Examples:
        >>> q = text_quality("This is a normal sentence with proper spacing.")
        >>> q['score'] > 0.7
        True
        >>> q['label']
        'clean'

        >>> q = text_quality("Thisisallgluedtogetherwithnospaces")
        >>> q['label']
        'no_space'
    """
    if not text or len(text) < 20:
        return {
            "score": 0.0,
            "label": "short",
            "whitespace_ratio": 0.0,
            "max_run_no_space": len(text) if text else 0,
            "camel_rate": 0.0,
            "avg_word_len": 0.0
        }

    # Calculate metrics
    total_chars = len(text)
    whitespace_chars = sum(1 for c in text if c.isspace())
    whitespace_ratio = whitespace_chars / total_chars if total_chars > 0 else 0.0

    # Find longest run of non-whitespace characters
    runs = re.split(r'\s+', text)
    max_run_no_space = max(len(run) for run in runs) if runs else 0

    # Count camelCase transitions (lowercase -> uppercase)
    camel_transitions = len(re.findall(r'[a-z][A-Z]', text))
    camel_rate = camel_transitions / total_chars if total_chars > 0 else 0.0

    # Average word length
    words = text.split()
    avg_word_len = sum(len(w) for w in words) / len(words) if words else 0.0

    # Calculate score (start at 1.0, apply penalties)
    score = 1.0

    # Penalty for low whitespace ratio (healthy range: 0.15-0.25)
    if whitespace_ratio < 0.10:
        score -= 0.4  # Severe penalty
    elif whitespace_ratio < 0.15:
        score -= 0.2

    # Penalty for long runs without spaces (healthy: <50 chars)
    if max_run_no_space > 100:
        score -= 0.3
    elif max_run_no_space > 50:
        score -= 0.15

    # Penalty for high camelCase rate (healthy: <0.05)
    if camel_rate > 0.10:
        score -= 0.3
    elif camel_rate > 0.05:
        score -= 0.15

    # Penalty for abnormal average word length (healthy: 4-8)
    if avg_word_len < 3 or avg_word_len > 12:
        score -= 0.1

    # Clamp score to [0, 1]
    score = max(0.0, min(1.0, score))

    # Determine label
    if score >= 0.7:
        label = "clean"
    elif whitespace_ratio < 0.10:
        label = "no_space"
    elif camel_rate > 0.05 or max_run_no_space > 50:
        label = "glued"
    else:
        label = "marginal"

    return {
        "score": round(score, 3),
        "label": label,
        "whitespace_ratio": round(whitespace_ratio, 3),
        "max_run_no_space": max_run_no_space,
        "camel_rate": round(camel_rate, 3),
        "avg_word_len": round(avg_word_len, 1)
    }
