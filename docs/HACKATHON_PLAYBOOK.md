# Polymath Hackathon Playbook (Spatial Multimodal)

## Objective
Note: Spatial coverage is already strong; use the audit to confirm and focus on multimodal gaps.

- Use existing spatial coverage to move fast on literature and code retrieval.
- Identify missing multimodal coverage, then ingest or fetch only what is needed.
- Turn evidence into pitch-ready claims with citations.

## Phase 1: Index What We Already Have (Spatial First)
1. Run the audit script to create a baseline index:
   - `python scripts/hackathon_audit.py --out-md docs/hackathon_reports/HACKATHON_AUDIT.md --out-json docs/hackathon_reports/hackathon_audit.json --neo4j`
   - If --neo4j fails, set NEO4J_USER/NEO4J_PASSWORD or remove the flag.
2. Review `docs/hackathon_reports/HACKATHON_AUDIT.md` for keyword coverage and gaps.
3. Create a small "golden query" list for spatial tasks you expect to use.
4. Snapshot code coverage by repo and keyword for quick implementation lookup.

## Phase 2: Identify Gaps (Multimodal and Benchmarks)
- Use the audit output to find missing or thin terms (zero or low coverage).
- Prioritize gaps that affect evaluation and positioning:
  - Missing modalities (for example, Xenium, CosMx, MERFISH).
  - Missing benchmarks or baselines.
  - Missing integration or alignment methods.
- Use the knowledge graph to find orphan concepts and weak bridges.

## Current Gap List (From Latest Audit)
- Missing: bayesprism, co-embedding, graph matching, in situ sequencing, label transfer, multi modal, slideseq, stalign, visium hd
- Thin: cosmx, multiome, multiomics, optimal transport, slide-seq, spatialde, stereo-seq, stereoseq
- Refresh this list after each audit run.

## Seed Targets for Multimodal Coverage
- Modalities: Visium HD, Xenium, CosMx, MERFISH, Slide-seq, Stereo-seq, seqFISH, in situ sequencing
- Methods: graph matching, optimal transport, co-embedding, label transfer
- Tooling: STalign, BayesPrism, SpatialDE
- Benchmarks: paired histology + spatial transcriptomics datasets and cross-modal alignment baselines

## Ingestion Checklist
1. Run the audit (add --neo4j if Postgres has no concept links).
2. Collect missing PDFs into the staging folder.
3. Ingest with `lib/unified_ingest.py` (use enhanced parsing for citability).
4. Add missing repos via `scripts/ingest_github_batch.py`.
5. Re-run the audit to confirm gaps are closed.

## Phase 3: Acquire Missing Data Fast
- Papers:
  - Add PDFs to the staging folder and run `lib/unified_ingest.py`.
  - Use the Literature Sentry for targeted queries if network access is allowed.
- Code:
  - Add missing repos via `scripts/ingest_github_batch.py`.
  - Re-ingest your hackathon repo to keep the code index fresh.
- Always treat Postgres as the system of record. Rebuild ChromaDB from Postgres only.

## Phase 4: Execution Loop During the Hackathon
- Search first, code second: retrieve 3 to 5 baselines before implementing.
- Keep evidence-bound notes to justify claims.
- Re-run the audit after a major ingestion batch to verify gaps are closed.

## Polymathic Advantage Patterns (Cross-Domain Leverage)
- Alignment: optimal transport, graph matching, and manifold alignment.
- Imputation: compressed sensing, matrix completion, and diffusion models.
- Integration: contrastive learning, CCA, and multi-view representation learning.
- Spatial priors: graph neural nets, MRFs, and physics-inspired regularization.

## Integration With Claude Code or Codex (Prompts That Work)
- "Find the top 10 spatial multimodal alignment methods and list associated code repos."
- "Given this baseline repo, locate where alignment is implemented and suggest edits."
- "Find gaps in our coverage for Xenium and CosMx, then propose missing papers."
- "Generate a short, evidence-bound justification for our novelty claim."

## Suggested Keyword Buckets (For Queries and Audits)
- Core: spatial, multimodal, multi-omics, integration, fusion
- Modalities: Visium, Visium HD, Xenium, MERFISH, CosMx, Slide-seq, seqFISH, Stereo-seq
- Imaging: H&E, histology, pathology, immunofluorescence, IHC
- Tasks: alignment, registration, deconvolution, segmentation, imputation, mapping
- Methods: optimal transport, graph matching, contrastive, transformer, GNN, variational
- Tooling: Seurat, Scanpy, Squidpy, cell2location, Tangram, STalign

## Evidence and Pitch Workflow
1. Draft claims for your approach.
2. Run evidence-bound verification for each claim.
3. Use citations in your slides and README to show novelty and rigor.

## Definition of Done
- The audit report shows coverage across spatial and key multimodal terms.
- Missing modalities and benchmarks are ingested or explicitly scoped out.
- The repo code index is updated and searchable.
- The final pitch includes evidence-bound citations.
