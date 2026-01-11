#!/usr/bin/env python3
"""
OpenAlex Citation Network Enrichment

Enriches Polymath documents with:
- OpenAlex IDs
- Citation counts
- Reference lists (who this paper cites)
- Cited-by lists (who cites this paper)
- Author affiliations
- Concepts/topics

This enables:
- "Missing keystone papers" detection
- Citation PageRank
- Bridge paper discovery
- Influence analysis

Usage:
    python3 scripts/enrich_openalex.py --all
    python3 scripts/enrich_openalex.py --missing-only
    python3 scripts/enrich_openalex.py --doc-id UUID
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from pathlib import Path

import httpx
import psycopg2
from psycopg2.extras import execute_values, Json

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.config import get_pg_connection

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# OpenAlex API (polite pool with email)
OPENALEX_API = "https://api.openalex.org"
USER_EMAIL = os.getenv("USER_EMAIL", "user@example.com")

# Rate limiting
REQUESTS_PER_SECOND = 10  # OpenAlex allows 10/sec with email
_last_request = 0


async def rate_limit():
    """Enforce rate limiting"""
    global _last_request
    now = time.time()
    elapsed = now - _last_request
    min_delay = 1.0 / REQUESTS_PER_SECOND
    if elapsed < min_delay:
        await asyncio.sleep(min_delay - elapsed)
    _last_request = time.time()


async def openalex_request(endpoint: str, params: dict = None) -> dict:
    """Make request to OpenAlex API"""
    await rate_limit()

    if params is None:
        params = {}
    params["mailto"] = USER_EMAIL

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            f"{OPENALEX_API}/{endpoint}",
            params=params
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()


async def find_work_by_doi(doi: str) -> Optional[dict]:
    """Find OpenAlex work by DOI"""
    # Normalize DOI
    doi = doi.lower().strip()
    if doi.startswith("https://doi.org/"):
        doi = doi[16:]
    elif doi.startswith("http://doi.org/"):
        doi = doi[15:]
    elif doi.startswith("doi:"):
        doi = doi[4:]

    result = await openalex_request(f"works/https://doi.org/{doi}")
    return result


async def find_work_by_title(title: str, year: Optional[int] = None) -> Optional[dict]:
    """Find OpenAlex work by title search"""
    params = {
        "search": title,
        "per_page": 5
    }

    if year:
        params["filter"] = f"publication_year:{year}"

    result = await openalex_request("works", params)

    if result and result.get("results"):
        # Return best match
        return result["results"][0]
    return None


async def find_work_by_pmid(pmid: str) -> Optional[dict]:
    """Find OpenAlex work by PubMed ID"""
    pmid = pmid.strip()
    if pmid.startswith("PMID:"):
        pmid = pmid[5:]

    result = await openalex_request(f"works/pmid:{pmid}")
    return result


@dataclass
class EnrichmentResult:
    """Result of enrichment"""
    doc_id: str
    openalex_id: Optional[str]
    cited_by_count: Optional[int]
    references_count: Optional[int]
    reference_ids: List[str]
    concepts: List[dict]
    authors_enriched: List[dict]
    success: bool
    error: Optional[str] = None


async def enrich_document(
    doc_id: str,
    doi: Optional[str] = None,
    pmid: Optional[str] = None,
    title: Optional[str] = None,
    year: Optional[int] = None
) -> EnrichmentResult:
    """Enrich a single document with OpenAlex data"""

    work = None
    error = None

    try:
        # Try DOI first (most reliable)
        if doi:
            work = await find_work_by_doi(doi)

        # Try PMID
        if not work and pmid:
            work = await find_work_by_pmid(pmid)

        # Fall back to title search
        if not work and title:
            work = await find_work_by_title(title, year)

        if not work:
            return EnrichmentResult(
                doc_id=doc_id,
                openalex_id=None,
                cited_by_count=None,
                references_count=None,
                reference_ids=[],
                concepts=[],
                authors_enriched=[],
                success=False,
                error="Not found in OpenAlex"
            )

        # Extract data
        openalex_id = work.get("id", "").replace("https://openalex.org/", "")

        # Get references (papers this paper cites)
        reference_ids = []
        for ref in work.get("referenced_works", []):
            ref_id = ref.replace("https://openalex.org/", "")
            reference_ids.append(ref_id)

        # Get concepts
        concepts = []
        for concept in work.get("concepts", []):
            concepts.append({
                "id": concept.get("id", "").replace("https://openalex.org/", ""),
                "name": concept.get("display_name"),
                "score": concept.get("score", 0)
            })

        # Get author info
        authors_enriched = []
        for authorship in work.get("authorships", []):
            author = authorship.get("author", {})
            institutions = []
            for inst in authorship.get("institutions", []):
                institutions.append({
                    "id": inst.get("id", "").replace("https://openalex.org/", ""),
                    "name": inst.get("display_name"),
                    "country": inst.get("country_code")
                })

            authors_enriched.append({
                "id": author.get("id", "").replace("https://openalex.org/", ""),
                "name": author.get("display_name"),
                "orcid": author.get("orcid"),
                "institutions": institutions
            })

        return EnrichmentResult(
            doc_id=doc_id,
            openalex_id=openalex_id,
            cited_by_count=work.get("cited_by_count", 0),
            references_count=len(reference_ids),
            reference_ids=reference_ids,
            concepts=concepts,
            authors_enriched=authors_enriched,
            success=True
        )

    except Exception as e:
        logger.exception(f"Error enriching {doc_id}")
        return EnrichmentResult(
            doc_id=doc_id,
            openalex_id=None,
            cited_by_count=None,
            references_count=None,
            reference_ids=[],
            concepts=[],
            authors_enriched=[],
            success=False,
            error=str(e)
        )


def ensure_schema(conn):
    """Ensure database schema for OpenAlex data"""
    with conn.cursor() as cur:
        # Add OpenAlex columns to documents if not exist
        cur.execute("""
            DO $$
            BEGIN
                -- OpenAlex ID
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'documents' AND column_name = 'openalex_id'
                ) THEN
                    ALTER TABLE documents ADD COLUMN openalex_id TEXT;
                END IF;

                -- Citation count
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'documents' AND column_name = 'cited_by_count'
                ) THEN
                    ALTER TABLE documents ADD COLUMN cited_by_count INTEGER;
                END IF;

                -- References count
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'documents' AND column_name = 'references_count'
                ) THEN
                    ALTER TABLE documents ADD COLUMN references_count INTEGER;
                END IF;

                -- OpenAlex concepts (JSONB)
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'documents' AND column_name = 'openalex_concepts'
                ) THEN
                    ALTER TABLE documents ADD COLUMN openalex_concepts JSONB;
                END IF;

                -- Authors enriched (JSONB)
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'documents' AND column_name = 'authors_enriched'
                ) THEN
                    ALTER TABLE documents ADD COLUMN authors_enriched JSONB;
                END IF;

                -- Enrichment timestamp
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'documents' AND column_name = 'openalex_enriched_at'
                ) THEN
                    ALTER TABLE documents ADD COLUMN openalex_enriched_at TIMESTAMP;
                END IF;
            END $$;
        """)

        # Create citation graph table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS citation_graph (
                source_doc_id UUID REFERENCES documents(id),
                target_openalex_id TEXT NOT NULL,
                target_doc_id UUID REFERENCES documents(id),  -- NULL if we don't have the paper
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (source_doc_id, target_openalex_id)
            );

            CREATE INDEX IF NOT EXISTS idx_citation_graph_target
            ON citation_graph(target_openalex_id);

            CREATE INDEX IF NOT EXISTS idx_citation_graph_target_doc
            ON citation_graph(target_doc_id);
        """)

        # Create view for missing keystone papers
        cur.execute("""
            CREATE OR REPLACE VIEW missing_keystone_papers AS
            SELECT
                cg.target_openalex_id,
                COUNT(DISTINCT cg.source_doc_id) as cited_by_our_papers,
                MAX(cg.created_at) as last_seen
            FROM citation_graph cg
            WHERE cg.target_doc_id IS NULL  -- We don't have this paper
            GROUP BY cg.target_openalex_id
            HAVING COUNT(DISTINCT cg.source_doc_id) >= 3  -- Cited by at least 3 of our papers
            ORDER BY cited_by_our_papers DESC;
        """)

        conn.commit()
        logger.info("Schema ensured")


def save_enrichment(conn, result: EnrichmentResult):
    """Save enrichment result to database"""
    with conn.cursor() as cur:
        if result.success:
            # Update document
            cur.execute("""
                UPDATE documents SET
                    openalex_id = %s,
                    cited_by_count = %s,
                    references_count = %s,
                    openalex_concepts = %s,
                    authors_enriched = %s,
                    openalex_enriched_at = CURRENT_TIMESTAMP
                WHERE id = %s
            """, (
                result.openalex_id,
                result.cited_by_count,
                result.references_count,
                Json(result.concepts),
                Json(result.authors_enriched),
                result.doc_id
            ))

            # Insert citation graph edges
            if result.reference_ids:
                # First, try to match reference IDs to our documents
                values = [(result.doc_id, ref_id) for ref_id in result.reference_ids]
                execute_values(cur, """
                    INSERT INTO citation_graph (source_doc_id, target_openalex_id, target_doc_id)
                    VALUES %s
                    ON CONFLICT (source_doc_id, target_openalex_id) DO NOTHING
                """, values)

                # Link to documents we have
                cur.execute("""
                    UPDATE citation_graph cg
                    SET target_doc_id = d.id
                    FROM documents d
                    WHERE cg.source_doc_id = %s
                      AND d.openalex_id = cg.target_openalex_id
                      AND cg.target_doc_id IS NULL
                """, (result.doc_id,))

        conn.commit()


async def enrich_all(conn, missing_only: bool = False, batch_size: int = 50):
    """Enrich all documents"""

    with conn.cursor() as cur:
        if missing_only:
            cur.execute("""
                SELECT id, doi, pmid, title, year
                FROM documents
                WHERE openalex_id IS NULL
                  AND (doi IS NOT NULL OR pmid IS NOT NULL OR title IS NOT NULL)
                ORDER BY year DESC NULLS LAST
            """)
        else:
            cur.execute("""
                SELECT id, doi, pmid, title, year
                FROM documents
                WHERE doi IS NOT NULL OR pmid IS NOT NULL OR title IS NOT NULL
                ORDER BY year DESC NULLS LAST
            """)

        documents = cur.fetchall()

    total = len(documents)
    logger.info(f"Enriching {total} documents")

    success_count = 0
    fail_count = 0

    for i, (doc_id, doi, pmid, title, year) in enumerate(documents):
        result = await enrich_document(
            doc_id=str(doc_id),
            doi=doi,
            pmid=pmid,
            title=title,
            year=year
        )

        save_enrichment(conn, result)

        if result.success:
            success_count += 1
        else:
            fail_count += 1

        if (i + 1) % 100 == 0:
            logger.info(f"Progress: {i + 1}/{total} ({success_count} success, {fail_count} fail)")

    logger.info(f"Complete: {success_count} success, {fail_count} fail")


async def get_missing_keystones(conn, limit: int = 50) -> List[dict]:
    """Get papers we're missing that are frequently cited"""

    with conn.cursor() as cur:
        cur.execute("""
            SELECT target_openalex_id, cited_by_our_papers
            FROM missing_keystone_papers
            LIMIT %s
        """, (limit,))

        results = []
        for openalex_id, cite_count in cur.fetchall():
            # Fetch metadata from OpenAlex
            work = await openalex_request(f"works/{openalex_id}")
            if work:
                results.append({
                    "openalex_id": openalex_id,
                    "title": work.get("title"),
                    "doi": work.get("doi"),
                    "year": work.get("publication_year"),
                    "cited_by_us": cite_count,
                    "global_citations": work.get("cited_by_count", 0)
                })

        return results


async def compute_citation_metrics(conn) -> dict:
    """Compute citation network metrics"""

    with conn.cursor() as cur:
        # Total citations in our corpus
        cur.execute("SELECT SUM(cited_by_count) FROM documents WHERE cited_by_count IS NOT NULL")
        total_citations = cur.fetchone()[0] or 0

        # Papers with citations
        cur.execute("SELECT COUNT(*) FROM documents WHERE cited_by_count > 0")
        papers_with_citations = cur.fetchone()[0]

        # Average citations
        cur.execute("SELECT AVG(cited_by_count) FROM documents WHERE cited_by_count IS NOT NULL")
        avg_citations = cur.fetchone()[0] or 0

        # Most cited in our corpus
        cur.execute("""
            SELECT title, cited_by_count, openalex_id
            FROM documents
            WHERE cited_by_count IS NOT NULL
            ORDER BY cited_by_count DESC
            LIMIT 10
        """)
        most_cited = [{"title": t, "citations": c, "id": i} for t, c, i in cur.fetchall()]

        # Citation graph stats
        cur.execute("SELECT COUNT(*) FROM citation_graph")
        total_edges = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM citation_graph WHERE target_doc_id IS NOT NULL")
        internal_edges = cur.fetchone()[0]

        # Missing keystones count
        cur.execute("SELECT COUNT(*) FROM missing_keystone_papers")
        missing_keystones = cur.fetchone()[0]

    return {
        "total_citations_global": total_citations,
        "papers_with_citations": papers_with_citations,
        "avg_citations": round(avg_citations, 2),
        "most_cited": most_cited,
        "citation_graph": {
            "total_edges": total_edges,
            "internal_edges": internal_edges,
            "external_edges": total_edges - internal_edges
        },
        "missing_keystone_count": missing_keystones
    }


def main():
    parser = argparse.ArgumentParser(description="OpenAlex enrichment")
    parser.add_argument("--all", action="store_true", help="Enrich all documents")
    parser.add_argument("--missing-only", action="store_true", help="Only enrich documents without OpenAlex data")
    parser.add_argument("--doc-id", type=str, help="Enrich specific document")
    parser.add_argument("--missing-keystones", action="store_true", help="Show missing keystone papers")
    parser.add_argument("--metrics", action="store_true", help="Show citation metrics")
    parser.add_argument("--limit", type=int, default=50, help="Limit for missing keystones")

    args = parser.parse_args()

    conn = get_pg_connection()

    try:
        ensure_schema(conn)

        if args.doc_id:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, doi, pmid, title, year
                    FROM documents WHERE id = %s
                """, (args.doc_id,))
                row = cur.fetchone()
                if not row:
                    logger.error(f"Document not found: {args.doc_id}")
                    return

                result = asyncio.run(enrich_document(
                    doc_id=str(row[0]),
                    doi=row[1],
                    pmid=row[2],
                    title=row[3],
                    year=row[4]
                ))
                save_enrichment(conn, result)
                print(json.dumps({
                    "success": result.success,
                    "openalex_id": result.openalex_id,
                    "cited_by_count": result.cited_by_count,
                    "references_count": result.references_count,
                    "error": result.error
                }, indent=2))

        elif args.missing_keystones:
            keystones = asyncio.run(get_missing_keystones(conn, args.limit))
            print("\n# Missing Keystone Papers\n")
            print("Papers frequently cited by our corpus but not in it:\n")
            for k in keystones:
                print(f"- **{k['title']}** ({k['year']})")
                print(f"  Cited by {k['cited_by_us']} of our papers, {k['global_citations']} global citations")
                if k['doi']:
                    print(f"  DOI: {k['doi']}")
                print()

        elif args.metrics:
            metrics = asyncio.run(compute_citation_metrics(conn))
            print("\n# Citation Network Metrics\n")
            print(f"Total global citations: {metrics['total_citations_global']:,}")
            print(f"Papers with citations: {metrics['papers_with_citations']:,}")
            print(f"Average citations: {metrics['avg_citations']}")
            print(f"\n## Most Cited in Corpus\n")
            for m in metrics["most_cited"]:
                print(f"- {m['title'][:60]}... ({m['citations']} citations)")
            print(f"\n## Citation Graph\n")
            print(f"Total edges: {metrics['citation_graph']['total_edges']:,}")
            print(f"Internal (we have both papers): {metrics['citation_graph']['internal_edges']:,}")
            print(f"External (missing target): {metrics['citation_graph']['external_edges']:,}")
            print(f"\nMissing keystone papers: {metrics['missing_keystone_count']}")

        elif args.all or args.missing_only:
            asyncio.run(enrich_all(conn, missing_only=args.missing_only))

        else:
            parser.print_help()

    finally:
        conn.close()


if __name__ == "__main__":
    main()
