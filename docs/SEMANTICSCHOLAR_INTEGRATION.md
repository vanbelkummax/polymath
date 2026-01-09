# Semantic Scholar Integration

## Overview

Semantic Scholar API integration provides access to 200M+ academic papers across all fields with:
- Citation and influence metrics
- Author disambiguation
- Open access detection
- Citation graph navigation
- Highly cited paper detection

**Status**: ✅ Production Ready (2026-01-07)

---

## Configuration

### API Key Setup

1. **API key location**: `/home/user/work/polymax/semanticscholar_api_key.txt`
2. **Environment variable**: `SEMANTICSCHOLAR_API_KEY` (already configured in `.env`)
3. **Rate limit**: 1 request per second (enforced automatically)

### Verification

```bash
# Check API key is loaded
export $(cat /home/user/polymath-repo/.env | xargs)
echo $SEMANTICSCHOLAR_API_KEY

# Run tests
cd /home/user/polymath-repo
python3 -m pytest tests/test_semanticscholar.py -v
```

---

## Usage

### Direct API Usage

```python
from lib.sentry.sources.semanticscholar import SemanticScholarSource

# Initialize (reads API key from environment)
ss = SemanticScholarSource()

# Basic search
papers = ss.discover(
    query="spatial transcriptomics",
    max_results=20
)

# Search with filters
papers = ss.discover(
    query="optimal transport",
    year_min=2020,              # Papers from 2020 onwards
    min_citations=50,           # Minimum 50 citations
    open_access_only=True,      # Only OA papers
    max_results=20
)

# Get paper by DOI
paper = ss.get_paper_by_id("10.1038/s41592-022-01409-2", id_type="DOI")

# Get highly cited papers in a field
papers = ss.get_highly_cited(
    field="Computer Science",
    year=2023,
    min_citations=100,
    max_results=20
)

# Search by author
papers = ss.get_author_papers(
    author_id="1234567",  # Semantic Scholar author ID
    max_results=50
)

# Navigate citation graph
references = ss.search_by_citations(
    seed_paper_id="abc123",
    direction="references",  # or "citations"
    max_results=50
)
```

---

## Literature Sentry Integration

Semantic Scholar is automatically included in Literature Sentry searches:

```bash
# Search across all sources (including Semantic Scholar)
python3 -m lib.sentry.cli "spatial transcriptomics visium" --max 20

# Search only Semantic Scholar
python3 -m lib.sentry.cli "optimal transport" --source semanticscholar --max 20
```

### Sources Priority

Semantic Scholar is positioned 2nd in discovery pipeline (after OpenAlex):

1. **OpenAlex** - 250M+ works, comprehensive metadata
2. **Semantic Scholar** - 200M+ papers, citation graphs
3. Europe PMC - full-text availability
4. arXiv - preprints
5. bioRxiv/medRxiv - biology preprints
6. GitHub - code repositories

---

## Features & Capabilities

### 1. Citation Metrics

Every paper includes:
- `citation_count`: Total citations
- `influential_citation_count`: Highly influential citations (semantic meaning)

```python
papers = ss.discover("machine learning", min_citations=100)
for p in papers:
    print(f"{p['title']}: {p['citation_count']} citations ({p['influential_citation_count']} influential)")
```

### 2. Field Filtering

Filter by fields of study:

```python
papers = ss.discover(
    query="neural networks",
    fields_of_study=["Computer Science", "Biology"],
    max_results=20
)
```

Available fields: Computer Science, Medicine, Biology, Chemistry, Physics, Materials Science, Engineering, Mathematics, Psychology, etc.

### 3. Open Access Detection

```python
# Only return papers with open access PDFs
oa_papers = ss.discover(
    query="spatial transcriptomics",
    open_access_only=True,
    max_results=50
)

# Check each paper
for p in oa_papers:
    if p.get("is_open_access") and p.get("pdf_url"):
        print(f"Download: {p['pdf_url']}")
```

### 4. Citation Graph Navigation

Find papers that cite or are cited by a seed paper:

```python
# Get papers cited BY this paper (references)
refs = ss.search_by_citations(
    seed_paper_id="abc123",
    direction="references",
    max_results=100
)

# Get papers that CITE this paper
cites = ss.search_by_citations(
    seed_paper_id="abc123",
    direction="citations",
    max_results=100
)
```

### 5. Multiple ID Types Supported

```python
# By DOI
paper = ss.get_paper_by_id("10.1038/...", id_type="DOI")

# By PubMed ID
paper = ss.get_paper_by_id("12345678", id_type="PubMed")

# By ArXiv ID
paper = ss.get_paper_by_id("2301.12345", id_type="ArXiv")

# By Semantic Scholar ID
paper = ss.get_paper_by_id("abc123", id_type="SemanticScholar")
```

---

## Result Format

Each paper returned as a dict with:

```python
{
    "title": str,
    "abstract": str | None,
    "authors": List[str],
    "year": int | None,
    "doi": str | None,
    "pmid": str | None,
    "arxiv_id": str | None,
    "pdf_url": str | None,
    "is_open_access": bool,
    "citation_count": int,
    "influential_citation_count": int,
    "fields_of_study": List[str],
    "publication_date": str | None,
    "journal": str | None,
    "source": "semanticscholar",
    "source_id": str,  # Semantic Scholar paper ID
    "url": str,  # Link to Semantic Scholar page
    "metadata": {
        "citation_count": int,
        "influential_citations": int,
        "fields": List[str],
        "is_oa": bool
    }
}
```

---

## Rate Limits

**With API key**: 1 request per second

Rate limiting is automatically enforced - no manual throttling needed.

**Without API key**: 100 requests per day (not recommended for production)

---

## Use Cases

### 1. Find Highly Cited Papers in Your Field

```python
papers = ss.get_highly_cited(
    field="Biology",
    year=2023,
    min_citations=50,
    max_results=100
)
```

### 2. Track Citation Impact

```python
# Get paper
paper = ss.get_paper_by_id("10.1038/...", id_type="DOI")

# Check citations
print(f"Total citations: {paper['citation_count']}")
print(f"Influential citations: {paper['influential_citation_count']}")

# Get papers that cite this
citing_papers = ss.search_by_citations(
    seed_paper_id=paper['source_id'],
    direction="citations",
    max_results=100
)
```

### 3. Find Cross-Domain Papers

```python
# Papers that bridge biology and computer science
papers = ss.discover(
    query="machine learning protein structure",
    fields_of_study=["Biology", "Computer Science"],
    min_citations=20,
    max_results=50
)
```

### 4. Build Citation Networks

```python
# Start with seed paper
seed = ss.get_paper_by_id("10.1038/...", id_type="DOI")

# Get its references
refs = ss.search_by_citations(seed['source_id'], "references", 50)

# Get papers that cite the seed
cites = ss.search_by_citations(seed['source_id'], "citations", 50)

# Build graph: seed → refs, seed ← cites
```

---

## Comparison with Other Sources

| Feature | Semantic Scholar | OpenAlex | Europe PMC |
|---------|------------------|----------|------------|
| **Coverage** | 200M+ papers | 250M+ works | 40M+ papers |
| **Citation data** | ✅ Excellent | ✅ Good | ❌ Limited |
| **Influential citations** | ✅ Yes | ❌ No | ❌ No |
| **Open access** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Full text** | ⚠️ Sometimes | ⚠️ Sometimes | ✅ Often |
| **Citation graph** | ✅ Yes | ⚠️ Limited | ❌ No |
| **Rate limit** | 1 req/sec | 10 req/sec | 10 req/sec |
| **Best for** | Citations, impact | Metadata | Full text |

**Recommendation**: Use Semantic Scholar for:
- Citation analysis
- Highly cited paper discovery
- Citation graph navigation
- Influence metrics

Use OpenAlex for:
- Comprehensive metadata
- Large-scale searches
- Author disambiguation

Use Europe PMC for:
- Full-text availability
- Recent biology papers

---

## Testing

All tests in `tests/test_semanticscholar.py`:

```bash
cd /home/user/polymath-repo
export $(cat .env | xargs)

# Run all tests
python3 -m pytest tests/test_semanticscholar.py -v

# Run specific test
python3 -m pytest tests/test_semanticscholar.py::test_search_with_filters -v -s
```

**Test coverage**:
- ✅ API key loading
- ✅ Rate limiting enforcement (1 req/sec)
- ✅ Basic search
- ✅ Search with filters (year, citations)
- ✅ DOI lookup
- ✅ Highly cited papers
- ✅ Open access filtering

---

## Troubleshooting

### Rate Limit Errors

If you see `429 Too Many Requests`:
- Check that API key is loaded: `echo $SEMANTICSCHOLAR_API_KEY`
- Verify rate limiting is working (should wait 1 sec between requests)
- Without API key, limit is 100 req/day total

### No Results

If searches return empty:
- Try broader queries
- Remove strict filters (year_min, min_citations)
- Check field names are correct

### API Key Not Found

```bash
# Check .env file
cat /home/user/polymath-repo/.env | grep SEMANTICSCHOLAR

# Reload environment
export $(cat /home/user/polymath-repo/.env | xargs)

# Verify
echo $SEMANTICSCHOLAR_API_KEY
```

---

## Documentation

- **API Docs**: https://api.semanticscholar.org/api-docs/
- **Fields of Study**: https://api.semanticscholar.org/api-docs/graph#tag/Paper-Data/Fields-of-Study
- **Source code**: `lib/sentry/sources/semanticscholar.py`
- **Tests**: `tests/test_semanticscholar.py`
