#!/usr/bin/env python3
"""
Citation Verification Protocol

Prevents citation hallucination by enforcing Search → Verify loop.
Never output a DOI unless title matches the specific DOI.
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Citation:
    """Verified citation with metadata."""
    title: str
    authors: Optional[str] = None
    year: Optional[int] = None
    venue: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    verified: bool = False
    source: str = "manual"  # or "search", "polymath"


class CitationVerifier:
    """
    Enforces citation integrity.

    Rules:
    1. Never output DOI without verification
    2. If uncertain, cite title + authors only
    3. Flag mismatches between title and DOI
    """

    def __init__(self):
        self.verified_citations: Dict[str, Citation] = {}

    def verify_doi_matches_title(self, doi: str, expected_title: str) -> Tuple[bool, str]:
        """
        Check if DOI actually points to the expected paper.

        Returns:
            (matches, actual_title or error_message)
        """
        # Pattern: Nature Communications DOI
        nature_pattern = r's\d+-\d+-\d+-\d+'

        if re.search(nature_pattern, doi):
            # This is a Nature family DOI - needs web verification
            # For now, flag as unverified
            return False, f"DOI {doi} needs web verification - do not use without checking"

        # Add more verification logic as needed
        return False, "Unverified DOI - cite by title only"

    def create_safe_citation(
        self,
        title: str,
        authors: Optional[str] = None,
        year: Optional[int] = None,
        venue: Optional[str] = None,
        doi: Optional[str] = None,
        url: Optional[str] = None,
        source: str = "manual"
    ) -> Citation:
        """
        Create citation with verification.

        If DOI provided, requires verification before accepting.
        """
        citation = Citation(
            title=title,
            authors=authors,
            year=year,
            venue=venue,
            doi=None,  # Start with None
            url=url,
            verified=False,
            source=source
        )

        # Only add DOI if verified
        if doi:
            matches, message = self.verify_doi_matches_title(doi, title)
            if matches:
                citation.doi = doi
                citation.verified = True
            else:
                print(f"WARNING: {message}")
                print(f"Citation will use title only: {title}")

        return citation

    def format_citation(self, citation: Citation, style: str = "inline") -> str:
        """
        Format citation for output.

        Styles:
        - inline: [Title](url) - Author et al. (Year)
        - full: Author et al. (Year). Title. Venue. DOI.
        - minimal: Title (Year)
        """
        if style == "inline":
            # Use URL if available, otherwise just title
            if citation.url:
                base = f"[{citation.title}]({citation.url})"
            else:
                base = citation.title

            # Add authors and year if available
            if citation.authors and citation.year:
                return f"{base} - {citation.authors} et al. ({citation.year})"
            elif citation.year:
                return f"{base} ({citation.year})"
            else:
                return base

        elif style == "full":
            parts = []
            if citation.authors:
                parts.append(f"{citation.authors} et al.")
            if citation.year:
                parts.append(f"({citation.year}).")
            parts.append(f"{citation.title}.")
            if citation.venue:
                parts.append(f"{citation.venue}.")
            if citation.doi and citation.verified:
                parts.append(f"DOI: {citation.doi}")
            elif citation.url:
                parts.append(f"URL: {citation.url}")

            return " ".join(parts)

        else:  # minimal
            if citation.year:
                return f"{citation.title} ({citation.year})"
            return citation.title

    def audit_citation_list(self, citations: List[Citation]) -> Dict[str, List[str]]:
        """
        Audit a list of citations for quality issues.

        Returns:
            Dict with issue categories and problematic citations
        """
        issues = {
            "unverified_dois": [],
            "missing_years": [],
            "missing_authors": [],
            "no_links": []
        }

        for cit in citations:
            if cit.doi and not cit.verified:
                issues["unverified_dois"].append(cit.title)
            if not cit.year:
                issues["missing_years"].append(cit.title)
            if not cit.authors:
                issues["missing_authors"].append(cit.title)
            if not cit.url and not cit.doi:
                issues["no_links"].append(cit.title)

        return issues


# Example usage
if __name__ == "__main__":
    verifier = CitationVerifier()

    # GOOD: Verified citation with URL
    good_cit = verifier.create_safe_citation(
        title="Rethinking Lossy Compression: The Rate-Distortion-Perception Tradeoff",
        authors="Blau & Michaeli",
        year=2019,
        venue="ICML",
        url="https://arxiv.org/abs/1901.07821",
        source="search"
    )

    print("GOOD CITATION:")
    print(verifier.format_citation(good_cit, style="inline"))
    print()

    # BAD: Unverified DOI that doesn't match title
    bad_cit = verifier.create_safe_citation(
        title="Persistent Homology Classifies Parameter Dependence of Patterns in Turing Systems",
        doi="s41467-023-36796-3",  # This is actually GraphST!
        year=2024,
        source="hallucination"
    )

    print("BAD CITATION (will reject DOI):")
    print(verifier.format_citation(bad_cit, style="full"))
    print()

    # Audit
    print("AUDIT REPORT:")
    issues = verifier.audit_citation_list([good_cit, bad_cit])
    for category, cits in issues.items():
        if cits:
            print(f"  {category}: {len(cits)} issues")
            for c in cits:
                print(f"    - {c[:60]}...")
