#!/usr/bin/env python3
"""
Fix common OCR artifacts in text files.
"""

import re
import sys
from pathlib import Path

# Common OCR error patterns -> corrections
OCR_FIXES = [
    # Bullet points and special chars
    (r'•', '-'),
    (r'»', ''),
    (r'«', ''),
    (r'ü', 'u'),
    (r'ö', 'o'),
    (r'ä', 'a'),

    # Common word breaks
    (r'(\w+)-\s*\n\s*(\w+)', r'\1\2'),  # rejoin hyphenated words

    # Strogatz-specific fixes
    (r'differentiati\(M', 'differentiation'),
    (r'ind%rndent', 'independent'),
    (r'deve101Ed', 'developed'),
    (r'grriodic', 'periodic'),
    (r'resiECt', 'respect'),
    (r'surmising', 'surprising'),
    (r'discOvery', 'discovery'),
    (r'tir\s+over', 'the over'),
    (r'tir', 'the'),
    (r'equati«ms', 'equations'),
    (r'trajMies', 'trajectories'),
    (r'ixluding', 'including'),
    (r'curve-sketch', 'curve-sketching'),
    (r'm•pulation', 'population'),
    (r'diK', 'disc'),

    # ESL-specific fixes
    (r'explod«i', 'exploded'),
    (r'ß', 'β'),
    (r'\bOf\b', 'of'),  # common capitalization error

    # General cleanup
    (r'\s+', ' '),  # normalize whitespace
    (r'(\w)\s*\.\s*(\w)', r'\1. \2'),  # fix sentence spacing
]


def fix_ocr(text: str) -> str:
    """Apply OCR fixes to text."""
    for pattern, replacement in OCR_FIXES:
        text = re.sub(pattern, replacement, text)
    return text


def main():
    if len(sys.argv) < 2:
        print("Usage: fix_ocr_artifacts.py <input_file> [output_file]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    original_len = len(text)
    fixed = fix_ocr(text)

    # Count fixes
    changes = sum(1 for a, b in zip(text, fixed) if a != b)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(fixed)
        print(f"Fixed {changes} characters, saved to {output_path}")
    else:
        # Overwrite in place
        with open(input_path, 'w', encoding='utf-8') as f:
            f.write(fixed)
        print(f"Fixed {changes} characters in place")

    return fixed


if __name__ == "__main__":
    main()
