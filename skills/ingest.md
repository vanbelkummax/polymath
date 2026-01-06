---
name: ingest
description: Fast unified ingestion of papers/code into Polymath system (ChromaDB + Neo4j + Postgres)
---

# /ingest - Unified Polymath Ingestion

Ingest papers or code into the Polymath knowledge system, updating all stores atomically.

## Quick Usage

```bash
# Ingest single PDF
python3 /home/user/work/polymax/lib/unified_ingest.py "/path/to/paper.pdf"

# Ingest from Windows Downloads (copy first for speed)
cp "/mnt/c/Users/User/Downloads/"*.pdf /home/user/work/polymax/ingest_staging/
python3 /home/user/work/polymax/lib/unified_ingest.py /home/user/work/polymax/ingest_staging/ --move

# Ingest code
python3 /home/user/work/polymax/lib/unified_ingest.py "/path/to/code.py" --type code

# Ingest code directory
python3 /home/user/work/polymax/lib/unified_ingest.py "/path/to/repo/" --type code
```

## What Gets Updated

1. **ChromaDB** (`polymath_v2`): Chunked text with embeddings
2. **Neo4j**: Paper/Code nodes with MENTIONS/USES relationships to concepts
3. **Postgres** (when available): Canonical metadata

## Cross-Domain Concept Linking

The ingestor automatically detects and links these polymathic concepts:

- **Signal**: compressed_sensing, sparse_coding, information_bottleneck, wavelet
- **Physics**: entropy, free_energy, thermodynamics, diffusion, reaction_diffusion
- **Causality**: causal_inference, counterfactual, do_calculus, instrumental_variable
- **Systems**: feedback, control_theory, autopoiesis, emergence, cybernetics
- **Cognitive**: predictive_coding, bayesian_brain, active_inference, affordance
- **ML/AI**: neural_network, transformer, attention, contrastive_learning

## OCR Handling

For scanned PDFs with no extractable text:
1. Tries marker-pdf (GPU-accelerated) first
2. Falls back to pytesseract
3. Use `--no-ocr` to skip OCR attempts

## After Ingestion

Check stats:
```bash
python3 /home/user/work/polymax/polymath_cli.py stats
```

Verify a paper was added:
```bash
python3 /home/user/work/polymax/polymath_cli.py search "title or topic"
```

## MCP Tool

If polymath MCP is loaded, use:
```
mcp__polymath__ingest_paper(path="/path/to/paper.pdf")
mcp__polymath__ingest_batch(directory="/path/to/pdfs/")
```
