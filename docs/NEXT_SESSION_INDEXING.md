# Next Session: Complete Code Repository Indexing

**Context**: Full database audit completed 2026-01-12. See `/home/user/polymath-repo/docs/FUTURE_WORK.md` for complete gap analysis.

## Session Goal

Index the 26 high-priority code repositories that are on disk but not in the database.

## Current State

| Metric | Value |
|--------|-------|
| Postgres passages | 765,565 |
| Concepts extracted | 4,868,910 (99.73%) |
| ChromaDB papers | 743,337 |
| ChromaDB code | 564,128 |
| Repos on disk | 313 |
| Repos indexed | 287 |
| **Repos to index** | **26** |

## Priority 1: Spatial Transcriptomics (DO FIRST)

These are directly relevant to your Img2ST research:

```bash
cd /home/user/polymath-repo

# Core spatial analysis
python3 lib/unified_ingest.py /home/user/work/polymax/data/github_repos/squidpy --type code
python3 lib/unified_ingest.py /home/user/work/polymax/data/github_repos/spatialdata --type code
python3 lib/unified_ingest.py /home/user/hist2st-benchmark --type code

# H&E to ST methods
python3 lib/unified_ingest.py /home/user/work/polymax/data/github_repos/HisToGene --type code
python3 lib/unified_ingest.py /home/user/work/polymax/data/github_repos/STAGATE --type code
python3 lib/unified_ingest.py /home/user/work/polymax/data/github_repos/SpaGCN --type code
```

## Priority 2: Foundation Models

```bash
# Mahmood lab
python3 lib/unified_ingest.py /home/user/work/comp_path_core/repos/HIPT --type code
python3 lib/unified_ingest.py /home/user/work/comp_path_core/repos/CLAM --type code

# Universal models
python3 lib/unified_ingest.py /home/user/work/polymax/data/github_repos/UNI --type code
python3 lib/unified_ingest.py /home/user/work/polymax/data/github_repos/CONCH --type code
```

## Priority 3: Single-Cell Foundation Models

```bash
python3 lib/unified_ingest.py /home/user/work/singlecell_foundation/repos/scGPT --type code
python3 lib/unified_ingest.py /home/user/work/singlecell_foundation/repos/Geneformer --type code
python3 lib/unified_ingest.py /home/user/work/singlecell_foundation/repos/scBERT --type code
```

## Verification After Indexing

```bash
# Check new totals
python3 -c "
import chromadb
c = chromadb.PersistentClient('/home/user/polymath-repo/chromadb')
print(f'Code chunks: {c.get_collection(\"polymath_code_bge_m3\").count():,}')
"

# Verify specific repo
psql -U polymath -d polymath -c "
SELECT repo_name, COUNT(*) as files
FROM code_files
WHERE repo_name LIKE '%squidpy%'
GROUP BY repo_name
"
```

## Next Steps After This Session

1. **Session 2**: Build gap detection system (`lib/gap_detector.py`)
2. **Session 3**: Build autonomous acquisition pipeline for open access papers
3. **Session 4**: Integrate with Literature Sentry for continuous discovery

## Reference Documents

- **Future Work**: `/home/user/polymath-repo/docs/FUTURE_WORK.md`
- **Architecture**: `/home/user/polymath-repo/docs/POLYMATH_V2_ARCHITECTURE.md`
- **CLAUDE.md**: `/home/user/CLAUDE.md`

## Troubleshooting

If ingest fails with title hash collision:
```python
# The unified_ingest.py already handles this by looking up existing doc_id
# If still failing, check the error message for specific guidance
```

If ChromaDB errors:
```bash
# Check disk space first
df -h /
# If corrupted, remove UUID dirs - they rebuild from SQLite
ls /home/user/polymath-repo/chromadb/
```

---

**Estimated Time**: 2-3 hours for all priority repos
**Expected Result**: +50-100K code chunks indexed
