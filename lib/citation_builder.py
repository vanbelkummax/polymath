#!/usr/bin/env python3
"""
Citation Builder - Format Evidence Spans into Verifiable Citations

Converts evidence spans (NLI-scored sentences) into formatted citations
with page references and confidence scores.

Features:
- Multiple citation styles (inline, full, audit trail)
- Batch processing from database
- DOI/year/venue backfill from document registry
- Verifiable page+char references

Version: HARDENING_2026-01-05
"""
import uuid
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VerifiableCitation:
    """A citation with complete evidence provenance."""
    # Document identity
    doc_id: uuid.UUID
    doi: Optional[str]
    pmid: Optional[str]
    arxiv_id: Optional[str]
    title: str
    authors: List[str]
    year: Optional[int]
    venue: Optional[str]

    # Evidence binding
    page_num: int
    span_char_start: int  # Within passage
    span_char_end: int
    evidence_snippet: str  # ≤100 chars
    entailment_score: float
    contradiction_score: float

    # Provenance
    passage_id: uuid.UUID
    span_id: int
    section: str
    quality_score: float

    def format_inline(self) -> str:
        """
        Inline citation format: (Author et al., Year, p.X)

        Examples:
            (Smith et al., 2023, p.7)
            (Doe, 2024, p.12)
        """
        if not self.authors:
            author_short = "Unknown"
        elif len(self.authors) == 1:
            # Single author: (Smith, 2023, p.7)
            author_short = self.authors[0].split()[-1] if self.authors[0] else "Unknown"
        else:
            # Multiple authors: (Smith et al., 2023, p.7)
            first_author = self.authors[0].split()[-1] if self.authors[0] else "Unknown"
            author_short = f"{first_author} et al."

        year_str = str(self.year) if self.year else "n.d."
        return f"({author_short}, {year_str}, p.{self.page_num})"

    def format_full(self) -> str:
        """
        Full citation with DOI.

        Format: Author(s). (Year). Title. Venue. DOI
        """
        # Authors
        if not self.authors:
            author_str = "Unknown"
        elif len(self.authors) <= 3:
            author_str = ", ".join(self.authors)
        else:
            author_str = ", ".join(self.authors[:3]) + ", et al."

        # Year
        year_str = f"({self.year})" if self.year else "(n.d.)"

        # Title
        title_str = self.title if self.title else "Untitled"

        # Venue
        venue_str = f" {self.venue}." if self.venue else ""

        # DOI
        if self.doi:
            doi_str = f" https://doi.org/{self.doi}"
        elif self.pmid:
            doi_str = f" PMID: {self.pmid}"
        elif self.arxiv_id:
            doi_str = f" arXiv:{self.arxiv_id}"
        else:
            doi_str = ""

        return f"{author_str}. {year_str}. {title_str}.{venue_str}{doi_str}"

    def format_audit_trail(self) -> str:
        """
        Full citation + evidence snippet + provenance metadata.

        Used for manual verification and PQE audit trails.
        """
        full_citation = self.format_full()

        audit = f"{full_citation}\n"
        audit += f"  Evidence (p.{self.page_num}, section: {self.section or 'body'}):\n"
        audit += f"    \"{self.evidence_snippet}\"\n"
        audit += f"  Support confidence: {self.entailment_score:.3f} (entailment), {self.contradiction_score:.3f} (contradiction)\n"
        audit += f"  Quality: {self.quality_score:.2f}\n"
        audit += f"  Provenance: doc_id={self.doc_id}, passage_id={self.passage_id}, span_id={self.span_id}"

        return audit


class CitationBuilder:
    """Build verifiable citations from evidence spans."""

    def __init__(self):
        """Initialize citation builder with database connection."""
        from lib.db import db
        self.db = db

    def build_citations_for_claims(self, claim_ids: List[str]) -> Dict[str, List[VerifiableCitation]]:
        """
        Batch fetch citations for multiple claims.

        Args:
            claim_ids: List of claim identifiers (e.g., ["P1", "P2", "mechanism_row_3"])

        Returns:
            Dict mapping claim_id → List[VerifiableCitation] (sorted by entailment score)
        """
        if not claim_ids:
            return {}

        # Build parameterized query for batch fetch
        placeholders = ", ".join(["%s"] * len(claim_ids))
        query = f"""
            SELECT
                cel.claim_id,
                cel.span_id,
                cel.entailment_score,
                es.doc_id,
                es.passage_id,
                es.page_num,
                es.span_char_start,
                es.span_char_end,
                es.span_text,
                es.section,
                es.quality_score,
                es.entailment_score as span_entailment,
                es.contradiction_score,
                d.doi,
                d.pmid,
                d.arxiv_id,
                d.title,
                d.authors,
                d.year,
                d.venue
            FROM claim_evidence_links cel
            JOIN evidence_spans es ON cel.span_id = es.span_id
            JOIN documents d ON es.doc_id = d.doc_id
            WHERE cel.claim_id IN ({placeholders})
            ORDER BY cel.claim_id, cel.entailment_score DESC
        """

        results = self.db.fetch_all(query, tuple(claim_ids))

        # Group by claim_id
        citations_by_claim = {}
        for row in results:
            claim_id = row['claim_id']

            # Extract snippet (≤100 chars)
            snippet = row['span_text'][:100]
            if len(row['span_text']) > 100:
                snippet += "..."

            citation = VerifiableCitation(
                doc_id=uuid.UUID(row['doc_id']),
                doi=row['doi'],
                pmid=row['pmid'],
                arxiv_id=row['arxiv_id'],
                title=row['title'],
                authors=row['authors'] or [],
                year=row['year'],
                venue=row['venue'],
                page_num=row['page_num'],
                span_char_start=row['span_char_start'],
                span_char_end=row['span_char_end'],
                evidence_snippet=snippet,
                entailment_score=row['entailment_score'],
                contradiction_score=row['contradiction_score'],
                passage_id=uuid.UUID(row['passage_id']),
                span_id=row['span_id'],
                section=row['section'],
                quality_score=row['quality_score']
            )

            if claim_id not in citations_by_claim:
                citations_by_claim[claim_id] = []
            citations_by_claim[claim_id].append(citation)

        logger.info(f"Built citations for {len(citations_by_claim)} claims ({sum(len(v) for v in citations_by_claim.values())} total spans)")
        return citations_by_claim

    def build_from_span_id(self, span_id: int) -> Optional[VerifiableCitation]:
        """
        Build a single citation from a span_id.

        Args:
            span_id: Database ID of the evidence span

        Returns:
            VerifiableCitation or None if span not found
        """
        query = """
            SELECT
                es.span_id,
                es.doc_id,
                es.passage_id,
                es.page_num,
                es.span_char_start,
                es.span_char_end,
                es.span_text,
                es.section,
                es.quality_score,
                es.entailment_score,
                es.contradiction_score,
                d.doi,
                d.pmid,
                d.arxiv_id,
                d.title,
                d.authors,
                d.year,
                d.venue
            FROM evidence_spans es
            JOIN documents d ON es.doc_id = d.doc_id
            WHERE es.span_id = %s
        """

        row = self.db.fetch_one(query, (span_id,))
        if not row:
            logger.warning(f"Span {span_id} not found in database")
            return None

        # Extract snippet
        snippet = row['span_text'][:100]
        if len(row['span_text']) > 100:
            snippet += "..."

        return VerifiableCitation(
            doc_id=uuid.UUID(row['doc_id']),
            doi=row['doi'],
            pmid=row['pmid'],
            arxiv_id=row['arxiv_id'],
            title=row['title'],
            authors=row['authors'] or [],
            year=row['year'],
            venue=row['venue'],
            page_num=row['page_num'],
            span_char_start=row['span_char_start'],
            span_char_end=row['span_char_end'],
            evidence_snippet=snippet,
            entailment_score=row['entailment_score'],
            contradiction_score=row['contradiction_score'],
            passage_id=uuid.UUID(row['passage_id']),
            span_id=row['span_id'],
            section=row['section'],
            quality_score=row['quality_score']
        )


if __name__ == "__main__":
    # Test the citation builder
    import sys
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    if len(sys.argv) < 2:
        print("Usage: python3 citation_builder.py <claim_id>")
        print("\nExample:")
        print("  python3 citation_builder.py P1")
        sys.exit(1)

    claim_id = sys.argv[1]

    builder = CitationBuilder()
    citations = builder.build_citations_for_claims([claim_id])

    if claim_id not in citations:
        print(f"No citations found for claim '{claim_id}'")
        sys.exit(0)

    print(f"\n{'='*60}")
    print(f"Citations for claim: {claim_id}")
    print(f"{'='*60}\n")

    for i, cite in enumerate(citations[claim_id], 1):
        print(f"[{i}] {cite.format_inline()}")
        print(f"    {cite.format_full()}")
        print(f"    Evidence: \"{cite.evidence_snippet}\"")
        print(f"    Score: {cite.entailment_score:.3f} (entailment), {cite.contradiction_score:.3f} (contradiction)")
        print()

    print("\nAudit Trail Format:")
    print("="*60)
    print(citations[claim_id][0].format_audit_trail())
