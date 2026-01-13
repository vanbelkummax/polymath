"""
Bridge Mine v3.1 - Improved Cross-Domain Discovery Pipeline

Fixes over v3:
1. Concept-type gating (reject category errors like OUTCOME→OUTCOME)
2. Typed validation (SOURCE_SUPPORT, TARGET_SUPPORT, ANALOGY_SUPPORT)
3. Novelty check (filter existing work, surface novel candidates)
4. Named experiment specs (no template placeholders)

Usage:
    python -m scripts.bridge_mine_v3.1.run_pipeline [output_dir]
"""
