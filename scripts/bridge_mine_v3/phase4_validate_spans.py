#!/usr/bin/env python3
"""
Phase 4: Validate Evidence Spans with Gemini 2.0 Flash
Classify each span as SUPPORTS / TANGENTIAL / GARBAGE
"""

import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, '/home/user/polymath-repo')

import google.generativeai as genai

OUTPUT_DIR = os.environ.get('OUTPUT_DIR', '/home/user/work/polymax/reports/bridge_mine_v3_20260112_2126')

# Load API key
API_KEY = os.environ.get('GEMINI_API_KEY')
if not API_KEY:
    with open('/home/user/polymath-repo/.env', 'r') as f:
        for line in f:
            if line.startswith('GEMINI_API_KEY='):
                API_KEY = line.strip().split('=', 1)[1]
                break

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

VALIDATION_PROMPT = """You are an expert reviewer validating whether a text passage provides evidence for a proposed cross-domain bridge.

BRIDGE: {source_concept} → {target_concept}
BUCKET: {bucket_description}

PASSAGE TO EVALUATE:
"{snippet}"

CLASSIFICATION RULES:
- SUPPORTS: The passage directly discusses or demonstrates how concepts related to {source_concept} can be applied to or inform {target_concept}, or vice versa. There must be explicit connection between the two domains.
- TANGENTIAL: The passage mentions one or both concepts but doesn't clearly connect them. May be useful context but not direct evidence.
- GARBAGE: The passage is irrelevant, too generic, or doesn't relate meaningfully to either concept.

Output ONLY one of: SUPPORTS, TANGENTIAL, GARBAGE"""


def validate_span(bridge_info: dict, snippet: str, max_retries: int = 3) -> str:
    """Classify a single span."""
    prompt = VALIDATION_PROMPT.format(
        source_concept=bridge_info['source_concept'],
        target_concept=bridge_info['target_concept'],
        bucket_description=bridge_info.get('bucket_description', bridge_info.get('bucket', '')),
        snippet=snippet[:800]  # Truncate long snippets
    )

    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.0,
                    'max_output_tokens': 20,
                }
            )
            result = response.text.strip().upper()
            if result in ['SUPPORTS', 'TANGENTIAL', 'GARBAGE']:
                return result
            # Handle partial matches
            if 'SUPPORTS' in result:
                return 'SUPPORTS'
            if 'TANGENTIAL' in result:
                return 'TANGENTIAL'
            return 'GARBAGE'
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
                return 'GARBAGE'
    return 'GARBAGE'


def main():
    # Load raw evidence
    evidence_file = Path(OUTPUT_DIR) / "EVIDENCE" / "RAW_EVIDENCE.jsonl"
    if not evidence_file.exists():
        print(f"ERROR: {evidence_file} not found. Run Phase 3 first.")
        return

    # Load bridge info for context
    aliases_file = Path(OUTPUT_DIR) / "INTERMEDIATE" / "ALIASES.jsonl"
    bridge_info = {}
    with open(aliases_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            bridge_info[data['bridge_id']] = data

    # Load evidence
    evidence_records = []
    with open(evidence_file, 'r') as f:
        for line in f:
            evidence_records.append(json.loads(line))

    print(f"Loaded {len(evidence_records)} evidence records to validate")

    # Check for resume
    validated_file = Path(OUTPUT_DIR) / "EVIDENCE" / "VALIDATED_EVIDENCE.jsonl"
    validated_ids = set()
    if validated_file.exists():
        with open(validated_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # Create unique key
                    key = f"{data['bridge_id']}_{data['passage_id']}"
                    validated_ids.add(key)
                except:
                    pass
        print(f"Resuming from {len(validated_ids)} already validated")

    # Validate
    stats = defaultdict(lambda: {'supports': 0, 'tangential': 0, 'garbage': 0, 'docs': set()})
    total = len(evidence_records)

    with open(validated_file, 'a') as f:
        for i, record in enumerate(evidence_records):
            key = f"{record['bridge_id']}_{record['passage_id']}"
            if key in validated_ids:
                continue

            bridge_id = record['bridge_id']
            info = bridge_info.get(bridge_id, {
                'source_concept': record.get('bridge_id', '').split('_')[0],
                'target_concept': '',
                'bucket_description': record.get('bucket', '')
            })

            if (i + 1) % 50 == 0:
                print(f"[{i+1}/{total}] Validating {bridge_id}...")

            classification = validate_span(info, record.get('snippet', ''))
            time.sleep(0.5)  # Rate limiting

            record['classification'] = classification
            f.write(json.dumps(record) + '\n')
            f.flush()

            # Track stats
            stats[bridge_id][classification.lower()] += 1
            if classification == 'SUPPORTS' and record.get('doc_id'):
                stats[bridge_id]['docs'].add(record['doc_id'])

    # Generate summary
    summary_file = Path(OUTPUT_DIR) / "VALIDATION_SUMMARY.md"
    with open(summary_file, 'w') as f:
        f.write("# Bridge Validation Summary\n\n")
        f.write(f"**Total evidence spans validated**: {len(evidence_records)}\n\n")

        # Count passing bridges
        passing = []
        for bridge_id, s in stats.items():
            supports = s['supports']
            docs = len(s['docs'])
            if supports >= 2 and docs >= 2:
                passing.append({
                    'bridge_id': bridge_id,
                    'supports': supports,
                    'distinct_docs': docs,
                    'tangential': s['tangential'],
                    'garbage': s['garbage']
                })

        passing.sort(key=lambda x: (x['supports'], x['distinct_docs']), reverse=True)

        f.write(f"## Passing Bridges (≥2 SUPPORTS from ≥2 docs): {len(passing)}\n\n")
        f.write("| Bridge ID | SUPPORTS | Distinct Docs | TANGENTIAL | GARBAGE |\n")
        f.write("|-----------|----------|---------------|------------|----------|\n")
        for p in passing:
            f.write(f"| {p['bridge_id']} | {p['supports']} | {p['distinct_docs']} | {p['tangential']} | {p['garbage']} |\n")

        f.write("\n## All Bridges Summary\n\n")
        f.write("| Bridge ID | SUPPORTS | Docs | TANGENTIAL | GARBAGE | Status |\n")
        f.write("|-----------|----------|------|------------|---------|--------|\n")
        for bridge_id, s in sorted(stats.items()):
            status = "✓ PASS" if (s['supports'] >= 2 and len(s['docs']) >= 2) else "✗ FAIL"
            f.write(f"| {bridge_id} | {s['supports']} | {len(s['docs'])} | {s['tangential']} | {s['garbage']} | {status} |\n")

    print(f"\n✓ Validation complete")
    print(f"  Passing bridges: {len(passing)}")
    print(f"  Summary: {summary_file}")


if __name__ == "__main__":
    main()
