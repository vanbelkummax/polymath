# Session Prompt: Full-Stack Knowledge Base Completion

**Paste this at session start.**

---

## Context

You're completing the Polymath knowledge base by filling two gaps identified in the 2026-01-12 audit:

1. **Metadata-only documents**: ~27K citation nodes with 0 passages. Priority: papers by Hwang, Landman, Lau, Sarkar + conceptually connected citations.
2. **Unindexed code repos**: 26 high-value repos on disk but not in DB (squidpy, spatialdata, HIPT, scGPT, etc.)

Current state: 765K passages, 4.87M concepts, 287/313 repos indexed.

## Your Mission

Execute the plan at `/home/user/polymath-repo/docs/NEXT_SESSION_FULLSTACK_INDEXING.md`

**Track A**: Convert 50 priority metadata-only docs → full passage-level documents
**Track B**: Index 26 priority code repositories

Expected yield: +10-20K passages, +50-100K code chunks

## First Steps

```bash
# 1. Read the detailed plan
cat /home/user/polymath-repo/docs/NEXT_SESSION_FULLSTACK_INDEXING.md

# 2. Load environment
cd /home/user/polymath-repo
set -a; source .env; set +a

# 3. Create run folder for Track A
RUN_DIR="/home/user/work/polymax/reports/metadata_to_fullpassage_$(date +%Y_%m_%d_%H%M)"
mkdir -p "$RUN_DIR"
echo "Run folder: $RUN_DIR"

# 4. Check current state
psql -U polymath -d polymath -c "SELECT COUNT(*) as passages FROM passages"
python3 -c "import chromadb; c=chromadb.PersistentClient('chromadb'); print(f'papers: {c.get_collection(\"polymath_bge_m3\").count():,}, code: {c.get_collection(\"polymath_code_bge_m3\").count():,}')"
```

## Hard Constraints

- **NO schema changes**
- **NO hardcoded keys** - use `.env`
- **NO /mnt/ paths** - WSL slow
- **Idempotent** - use existing `title_hash` collision logic
- **Log everything** to run folder

## Key Files

| File | Purpose |
|------|---------|
| `docs/NEXT_SESSION_FULLSTACK_INDEXING.md` | Detailed execution plan |
| `docs/FUTURE_WORK.md` | Gap analysis context |
| `lib/unified_ingest.py` | Main ingestion script |
| `/home/user/CLAUDE.md` | System reference |

## Target Faculty (Track A)

Papers by or connected to:
- **Tae Hyun Hwang** - holotomography, molecular AI, therapy prediction
- **Bennett Landman** - brain MRI, harmonization, body composition
- **Ken Lau** - CRC spatial atlas, single-cell, tumor microenvironment
- **Hirak Sarkar** - spatial transcriptomics methods, deconvolution

## Priority Repos (Track B)

1. **Spatial**: squidpy, spatialdata, hist2st-benchmark, HisToGene, STAGATE, SpaGCN
2. **Foundation**: HIPT, CLAM, UNI, CONCH
3. **Single-cell**: scGPT, Geneformer, scBERT

## Success Criteria

- [ ] No duplicate documents created
- [ ] All ingested targets have `COUNT(passages) > 0`
- [ ] All 26 repos indexed without errors
- [ ] Code chunk count increased by 50-100K
- [ ] Logs and CSVs written to run folder
- [ ] `POSTCHECK.md` created with summary

---

**Start by reading the full plan, then execute phase by phase. Show outputs after each phase.**
