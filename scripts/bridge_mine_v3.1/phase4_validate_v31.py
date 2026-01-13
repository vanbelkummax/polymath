#!/usr/bin/env python3
"""
Phase 4: Validate Evidence Spans with v3.1 Improvements

Integrates:
1. Concept-type gating (reject OUTCOME→OUTCOME, CLINICAL_PROCESS→OUTCOME)
2. Typed validation (SOURCE_SUPPORT, TARGET_SUPPORT, ANALOGY_SUPPORT)
3. Novelty check (Mode A EXISTING vs Mode B NOVEL_CANDIDATE)

A bridge passes v3.1 if:
- Passes concept-type gate (valid source/target type combination)
- ≥2 SOURCE_SUPPORT spans from ≥2 docs
- ≥1 TARGET_SUPPORT span (or curated statement)
- ≥1 ANALOGY_SUPPORT (from retrieval or LLM-generated if falsifiable)
- Novelty mode is NOVEL_CANDIDATE or NEEDS_REVIEW (not EXISTING)
"""

import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict
from dataclasses import asdict

sys.path.insert(0, '/home/user/polymath-repo')
sys.path.insert(0, '/home/user/polymath-repo/scripts/bridge_mine_v3.1')

from concept_type_gating import classify_concept, is_valid_bridge, ConceptType
from typed_validation import validate_span_typed, generate_analogy, SupportType, BridgeEvidence
from novelty_check import check_novelty, NoveltyMode

OUTPUT_DIR = os.environ.get('OUTPUT_DIR', '/home/user/work/polymax/reports/bridge_mine_v3.1_latest')


def load_evidence(output_dir: Path) -> tuple:
    """Load raw evidence and bridge info from Phase 3."""
    evidence_file = output_dir / "EVIDENCE" / "RAW_EVIDENCE.jsonl"
    if not evidence_file.exists():
        print(f"ERROR: {evidence_file} not found. Run Phase 3 first.")
        sys.exit(1)

    evidence_records = []
    with open(evidence_file, 'r') as f:
        for line in f:
            evidence_records.append(json.loads(line))

    # Load bridge info
    aliases_file = output_dir / "INTERMEDIATE" / "ALIASES.jsonl"
    bridge_info = {}
    if aliases_file.exists():
        with open(aliases_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                bridge_info[data['bridge_id']] = data

    return evidence_records, bridge_info


def phase4_validate_v31(output_dir: Path):
    """
    Run v3.1 validation pipeline:
    1. Concept-type gating
    2. Typed span validation
    3. Novelty check
    """
    print("=" * 60)
    print("BRIDGE MINE v3.1 - Phase 4: Typed Validation")
    print("=" * 60)

    evidence_records, bridge_info = load_evidence(output_dir)
    print(f"\nLoaded {len(evidence_records)} evidence records for {len(bridge_info)} bridges")

    # Group evidence by bridge
    evidence_by_bridge = defaultdict(list)
    for record in evidence_records:
        bridge_id = record.get('bridge_id', '')
        if bridge_id:
            evidence_by_bridge[bridge_id].append(record)

    # Results storage
    results = []
    gated_out = []
    validated = []
    failed_validation = []
    novelty_filtered = []

    # ========================================
    # STEP 1: Concept-Type Gating
    # ========================================
    print("\n[1/3] Concept-Type Gating...")

    for bridge_id, info in bridge_info.items():
        source = info.get('source_concept', '')
        target = info.get('target_concept', '')

        source_type = classify_concept(source, info.get('bucket_description', ''))
        time.sleep(0.1)
        target_type = classify_concept(target, info.get('bucket_description', ''))
        time.sleep(0.1)

        is_valid, reason = is_valid_bridge(source_type, target_type)

        info['source_type'] = source_type.value
        info['target_type'] = target_type.value
        info['gate_status'] = 'PASS' if is_valid else 'REJECT'
        info['gate_reason'] = reason

        if not is_valid:
            gated_out.append({
                'bridge_id': bridge_id,
                'source': source,
                'target': target,
                'source_type': source_type.value,
                'target_type': target_type.value,
                'reason': reason
            })
            print(f"  ✗ GATED OUT: {bridge_id} ({source_type.value} → {target_type.value})")

    print(f"\n  Gated out: {len(gated_out)} bridges (category errors)")
    remaining_bridges = [bid for bid, info in bridge_info.items() if info.get('gate_status') == 'PASS']
    print(f"  Remaining: {len(remaining_bridges)} bridges")

    # ========================================
    # STEP 2: Typed Validation
    # ========================================
    print("\n[2/3] Typed Span Validation...")

    validated_evidence_file = output_dir / "EVIDENCE" / "VALIDATED_EVIDENCE_V31.jsonl"
    validated_evidence_file.parent.mkdir(parents=True, exist_ok=True)

    with open(validated_evidence_file, 'w') as f:
        for i, bridge_id in enumerate(remaining_bridges):
            info = bridge_info[bridge_id]
            evidence = evidence_by_bridge.get(bridge_id, [])

            if not evidence:
                failed_validation.append({
                    'bridge_id': bridge_id,
                    'reason': 'No evidence retrieved'
                })
                continue

            print(f"  [{i+1}/{len(remaining_bridges)}] {bridge_id} ({len(evidence)} spans)...")

            # Classify each span
            source_support = []
            target_support = []
            analogy_support = []
            source_docs = set()
            target_docs = set()

            for record in evidence:
                snippet = record.get('snippet', '')
                if not snippet or len(snippet) < 20:
                    continue

                result = validate_span_typed(info, snippet)
                time.sleep(0.5)  # Rate limiting

                record['support_type'] = result.support_type.value
                record['confidence'] = result.confidence
                record['rationale'] = result.rationale
                f.write(json.dumps(record) + '\n')

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

            # Check thresholds
            source_ok = len(source_support) >= 2 and len(source_docs) >= 2
            target_ok = len(target_support) >= 1
            analogy_ok = len(analogy_support) >= 1

            # Try to generate analogy if missing
            if not analogy_ok and source_ok and target_ok:
                print(f"    Generating analogy for {bridge_id}...")
                generated = generate_analogy(info)
                time.sleep(0.5)
                if generated and generated.get('confidence', 0) >= 0.5:
                    analogy_support.append({
                        'type': 'generated',
                        'structural_analogy': generated.get('structural_analogy', ''),
                        'hypothesis': generated.get('hypothesis', ''),
                        'falsification_test': generated.get('falsification_test', ''),
                        'confidence': generated.get('confidence', 0.5)
                    })
                    analogy_ok = True
                    f.write(json.dumps({
                        'bridge_id': bridge_id,
                        'support_type': 'ANALOGY_SUPPORT_GENERATED',
                        **generated
                    }) + '\n')

            is_passing = source_ok and target_ok and analogy_ok

            bridge_result = {
                'bridge_id': bridge_id,
                'source_concept': info.get('source_concept'),
                'target_concept': info.get('target_concept'),
                'source_type': info.get('source_type'),
                'target_type': info.get('target_type'),
                'source_support_count': len(source_support),
                'source_doc_count': len(source_docs),
                'target_support_count': len(target_support),
                'analogy_support_count': len(analogy_support),
                'is_passing_typed': is_passing
            }

            if is_passing:
                validated.append(bridge_result)
            else:
                reasons = []
                if not source_ok:
                    reasons.append(f"SOURCE: {len(source_support)} spans/{len(source_docs)} docs")
                if not target_ok:
                    reasons.append(f"TARGET: {len(target_support)} spans")
                if not analogy_ok:
                    reasons.append("ANALOGY: none found or generated")
                bridge_result['failure_reason'] = ' | '.join(reasons)
                failed_validation.append(bridge_result)

            results.append(bridge_result)

    print(f"\n  Validated: {len(validated)} bridges passed typed validation")
    print(f"  Failed: {len(failed_validation)} bridges")

    # ========================================
    # STEP 3: Novelty Check
    # ========================================
    print("\n[3/3] Novelty Check...")

    for i, bridge_result in enumerate(validated):
        bridge_id = bridge_result['bridge_id']
        info = bridge_info.get(bridge_id, {})

        source = info.get('source_concept', '')
        target = info.get('target_concept', '')

        print(f"  [{i+1}/{len(validated)}] Checking novelty: {source} → spatial transcriptomics...")

        novelty = check_novelty(source, target, "spatial transcriptomics")
        time.sleep(1)  # API rate limiting

        bridge_result['novelty_mode'] = novelty.mode.value
        bridge_result['novelty_pubmed_hits'] = novelty.pubmed_hits
        bridge_result['novelty_s2_hits'] = novelty.s2_hits
        bridge_result['novelty_confidence'] = novelty.confidence
        bridge_result['novelty_rationale'] = novelty.rationale
        bridge_result['novelty_evidence_urls'] = novelty.evidence_urls

        if novelty.mode == NoveltyMode.EXISTING:
            novelty_filtered.append(bridge_result)
            print(f"    ✗ EXISTING: {novelty.pubmed_hits + novelty.s2_hits} papers found")
        elif novelty.mode == NoveltyMode.NOVEL_CANDIDATE:
            print(f"    ✓ NOVEL CANDIDATE")
        else:
            print(f"    ⚠ {novelty.mode.value}: needs manual review")

    # Filter to novel candidates
    novel_bridges = [b for b in validated if b.get('novelty_mode') in ['NOVEL_CANDIDATE', 'NEEDS_REVIEW']]
    print(f"\n  Novel candidates: {len(novel_bridges)}")
    print(f"  Filtered (existing): {len(novelty_filtered)}")

    # ========================================
    # Save Results
    # ========================================
    print("\n[4/4] Saving results...")

    # Save gated out bridges
    gated_file = output_dir / "INTERMEDIATE" / "GATED_OUT.json"
    with open(gated_file, 'w') as f:
        json.dump(gated_out, f, indent=2)

    # Save all validation results
    all_results_file = output_dir / "INTERMEDIATE" / "VALIDATION_RESULTS_V31.json"
    with open(all_results_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Save novel candidates (the ones that passed everything)
    novel_file = output_dir / "INTERMEDIATE" / "NOVEL_CANDIDATES.json"
    with open(novel_file, 'w') as f:
        json.dump(novel_bridges, f, indent=2)

    # Generate summary
    summary_file = output_dir / "VALIDATION_SUMMARY_V31.md"
    with open(summary_file, 'w') as f:
        f.write("# Bridge Mine v3.1 Validation Summary\n\n")
        f.write("## Pipeline Overview\n\n")
        f.write(f"- **Input bridges**: {len(bridge_info)}\n")
        f.write(f"- **Gated out (category errors)**: {len(gated_out)}\n")
        f.write(f"- **Passed typed validation**: {len(validated)}\n")
        f.write(f"- **Filtered (existing in lit)**: {len(novelty_filtered)}\n")
        f.write(f"- **Novel candidates (final)**: {len(novel_bridges)}\n\n")

        f.write("## v3.1 Improvements Applied\n\n")
        f.write("1. **Concept-Type Gating**: Rejected OUTCOME→OUTCOME, CLINICAL_PROCESS→OUTCOME bridges\n")
        f.write("2. **Typed Validation**: Required SOURCE_SUPPORT + TARGET_SUPPORT + ANALOGY_SUPPORT\n")
        f.write("3. **Novelty Check**: Filtered bridges with existing PubMed/S2 co-mentions\n\n")

        f.write("## Gated Out Bridges (Category Errors)\n\n")
        if gated_out:
            f.write("| Bridge ID | Source Type | Target Type | Reason |\n")
            f.write("|-----------|-------------|-------------|--------|\n")
            for g in gated_out[:10]:
                f.write(f"| {g['bridge_id'][:30]} | {g['source_type']} | {g['target_type']} | {g['reason'][:40]} |\n")
            if len(gated_out) > 10:
                f.write(f"\n*... and {len(gated_out) - 10} more*\n")
        else:
            f.write("*None*\n")

        f.write("\n## Novel Candidate Bridges (Passed All Checks)\n\n")
        if novel_bridges:
            f.write("| Bridge ID | Source | Target | SOURCE | TARGET | ANALOGY | Novelty |\n")
            f.write("|-----------|--------|--------|--------|--------|---------|----------|\n")
            for b in sorted(novel_bridges, key=lambda x: x['source_support_count'], reverse=True)[:20]:
                f.write(f"| {b['bridge_id'][:25]} | {b['source_concept'][:20]} | {b['target_concept'][:20]} | ")
                f.write(f"{b['source_support_count']}/{b['source_doc_count']} | {b['target_support_count']} | ")
                f.write(f"{b['analogy_support_count']} | {b.get('novelty_mode', '?')} |\n")
        else:
            f.write("*No novel candidates found*\n")

        f.write("\n## Failed Validation\n\n")
        if failed_validation:
            f.write("| Bridge ID | Reason |\n")
            f.write("|-----------|--------|\n")
            for b in failed_validation[:10]:
                reason = b.get('failure_reason', b.get('reason', 'Unknown'))
                f.write(f"| {b['bridge_id'][:30]} | {reason[:60]} |\n")
            if len(failed_validation) > 10:
                f.write(f"\n*... and {len(failed_validation) - 10} more*\n")
        else:
            f.write("*None*\n")

        f.write("\n## Filtered (Already Exists in Literature)\n\n")
        if novelty_filtered:
            f.write("| Bridge ID | PubMed | S2 | Rationale |\n")
            f.write("|-----------|--------|----|-----------|\n")
            for b in novelty_filtered[:10]:
                f.write(f"| {b['bridge_id'][:30]} | {b.get('novelty_pubmed_hits', 0)} | ")
                f.write(f"{b.get('novelty_s2_hits', 0)} | {b.get('novelty_rationale', '')[:50]} |\n")
        else:
            f.write("*None*\n")

    print(f"\n{'=' * 60}")
    print("VALIDATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Novel candidates: {len(novel_bridges)}")
    print(f"Summary: {summary_file}")
    print(f"Candidates: {novel_file}")

    return novel_bridges


def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    novel_bridges = phase4_validate_v31(output_dir)

    if novel_bridges:
        print("\n\nTop 5 Novel Candidates:")
        for i, b in enumerate(sorted(novel_bridges, key=lambda x: x['source_support_count'], reverse=True)[:5]):
            print(f"\n{i+1}. {b['source_concept']} → {b['target_concept']}")
            print(f"   Type: {b['source_type']} → {b['target_type']}")
            print(f"   Evidence: {b['source_support_count']} SOURCE / {b['target_support_count']} TARGET / {b['analogy_support_count']} ANALOGY")
            print(f"   Novelty: {b['novelty_mode']}")


if __name__ == "__main__":
    main()
