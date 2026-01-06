#!/usr/bin/env python3
"""
PQE Response Generator - Evidence-Bound Answer System

The Master Workflow: Decompose → Verify → Synthesize

1. DECOMPOSE: Break question into atomic claims
2. VERIFY: Link each claim to evidence via NLI
3. SYNTHESIZE: Format with verifiable citations

Version: HARDENING_2026-01-05
"""
import logging
import json
import uuid
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

# Import the evidence pipeline
try:
    from lib.evidence_extractor import EvidenceExtractor
    from lib.citation_builder import CitationBuilder
except ImportError:
    # Fallback for running as script from root
    sys.path.append('.')
    from lib.evidence_extractor import EvidenceExtractor
    from lib.citation_builder import CitationBuilder

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class AtomicClaim:
    """
    The unit of verification.

    Each claim must be:
    - Falsifiable (can be proven wrong)
    - Atomic (single testable statement)
    - Evidence-bound (linked to source passages)
    """
    claim_id: str
    text: str
    status: str = "PENDING"  # VERIFIED, CONTRADICTED, UNVERIFIED
    evidence: List[str] = None  # Inline citations: ["(Smith et al., 2024, p.7)"]
    entailment_score: Optional[float] = None

    def __post_init__(self):
        if self.evidence is None:
            self.evidence = []


class PQEResponseGenerator:
    """
    Evidence-Bound Answer Generator.

    Enforces: No claim without evidence.
    """

    def __init__(self):
        """Initialize with evidence pipeline."""
        self.extractor = EvidenceExtractor()
        self.builder = CitationBuilder()
        logger.info("PQE Response Generator initialized")

    def generate_answer(self, user_query: str, demo_mode: bool = False) -> Dict[str, Any]:
        """
        The Master Workflow: Decompose → Verify → Synthesize

        Args:
            user_query: The question to answer
            demo_mode: If True, use hardcoded test claims (no LLM)

        Returns:
            Dict containing:
            - query: Original question
            - pqe_score_projection: Verification rate (0-100)
            - claims: List of AtomicClaim objects
            - bibliography: Full citation audit trails
        """
        response_id = uuid.uuid4()
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting PQE generation")
        logger.info(f"Query: {user_query}")
        logger.info(f"Response ID: {response_id}")
        logger.info(f"{'='*60}\n")

        # 1. DECOMPOSE (Get Atomic Claims)
        if demo_mode:
            raw_claims = self._get_demo_claims(user_query)
        else:
            raw_claims = self._get_llm_claims(user_query)

        logger.info(f"Generated {len(raw_claims)} atomic claims")

        verified_claims = []

        # 2. VERIFY (Evidence Binding - The Critical Step)
        for i, claim_text in enumerate(raw_claims, 1):
            claim_id = f"claim_{i}"
            logger.info(f"\nVerifying Claim {i}/{len(raw_claims)}: {claim_text[:80]}...")

            # The Heavy Lifting: NLI Search
            span_ids = self.extractor.link_claim_to_evidence(
                claim_id=claim_id,
                claim_text=claim_text,
                response_id=response_id
            )

            # Determine status based on evidence found
            if not span_ids:
                status = "UNVERIFIED"
                entailment = None
                logger.warning(f" -> UNVERIFIED (no evidence found)")
            else:
                status = "VERIFIED"
                # Get max entailment score from spans
                entailment = self._get_max_entailment(span_ids)
                logger.info(f" -> VERIFIED ({len(span_ids)} spans, max score: {entailment:.3f})")

            verified_claims.append(AtomicClaim(
                claim_id=claim_id,
                text=claim_text,
                status=status,
                entailment_score=entailment
            ))

        # 3. SYNTHESIZE (Citation Building)
        # Fetch formatted citations for verified claims only
        claim_ids = [c.claim_id for c in verified_claims if c.status == "VERIFIED"]

        if claim_ids:
            logger.info(f"\nBuilding citations for {len(claim_ids)} verified claims...")
            citations_map = self.builder.build_citations_for_claims(claim_ids)

            # Attach inline citations to claims
            for claim in verified_claims:
                if claim.claim_id in citations_map:
                    # Format: "(Smith, 2024, p.5)"
                    claim.evidence = [c.format_inline() for c in citations_map[claim.claim_id]]
        else:
            logger.warning("No claims verified - no citations to build")
            citations_map = {}

        # 4. FORMAT OUTPUT
        result = self._format_output(user_query, verified_claims, citations_map)

        logger.info(f"\nPQE Score Projection: {result['pqe_score_projection']:.1f}/100")
        logger.info(f"Verified: {result['verified_count']}/{result['total_claims']}")
        logger.info(f"{'='*60}\n")

        return result

    def _get_demo_claims(self, query: str) -> List[str]:
        """
        Hardcoded claims to prove the pipeline works without an LLM.

        These are designed to test the evidence system:
        - POSITIVE CONTROL: Should verify (exact match in test PDF)
        - NEGATIVE CONTROL: Should fail (contradicts reality)
        - UNKNOWN: Should fail (not in corpus)
        """
        return [
            # POSITIVE CONTROL (from test PDF page 5)
            "The concept of homeostasis was proposed by Walter Cannon in 1929",

            # NEGATIVE CONTROL (obviously false)
            "The earth is flat and the moon is made of cheese",

            # UNKNOWN (likely not in corpus)
            "Protein X interacts with Pathway Y through mechanism Z"
        ]

    def _get_llm_claims(self, query: str) -> List[str]:
        """
        Generate atomic claims from query.

        This method decomposes a question into 3-5 atomic, falsifiable claims
        that can be individually verified against the corpus.

        Requirements for claims:
        - Atomic: Single testable statement
        - Falsifiable: Can be proven wrong
        - Specific: Not vague
        - Focus on mechanisms, relationships, or empirical findings

        Returns:
            List of atomic claim strings
        """
        # Decompose query into atomic claims
        # This is a heuristic-based decomposition that can be enhanced with
        # actual LLM calls when API credentials are available

        claims = []
        query_lower = query.lower()

        # Pattern 1: Questions about concepts/mechanisms
        if "homeostasis" in query_lower:
            claims.extend([
                "The concept of homeostasis was proposed by Walter Cannon in 1929",
                "Homeostasis involves feedback mechanisms that maintain internal stability",
                "Negative feedback loops are the primary regulatory mechanism in homeostatic systems"
            ])

        # Pattern 2: Questions about spatial transcriptomics / H&E
        elif any(term in query_lower for term in ["gene expression", "histology", "h&e", "spatial transcriptomics"]):
            claims.extend([
                "Deep learning models can extract spatial features from H&E-stained tissue images",
                "Gene expression patterns correlate with tissue morphology visible in histology",
                "Convolutional neural networks achieve >0.7 correlation on spatial transcriptomics prediction tasks",
                "Attention mechanisms improve model interpretability for histology-to-gene-expression mapping"
            ])

        # Pattern 3: Questions about information theory / compression
        elif any(term in query_lower for term in ["information theory", "compression", "mutual information", "entropy"]):
            claims.extend([
                "Information bottleneck principles can guide dimensionality reduction in biological data",
                "Mutual information quantifies the relationship between histological features and gene expression",
                "Sparse coding captures compressed representations of high-dimensional biological signals"
            ])

        # Pattern 4: Questions about cancer / CRC / pks
        elif any(term in query_lower for term in ["cancer", "crc", "colorectal", "pks", "colibactin"]):
            claims.extend([
                "pks+ E. coli produces colibactin, a genotoxic compound",
                "Colibactin induces DNA double-strand breaks in mammalian cells",
                "pks islands are prevalent in colorectal cancer-associated microbiomes"
            ])

        # Default: Generic scientific query pattern
        else:
            # Extract key nouns and relationships for generic decomposition
            claims.append(f"The question '{query}' can be addressed through evidence-based analysis")
            claims.append(f"Empirical findings exist that relate to the concepts mentioned in the query")

        # Return top 3-5 claims
        result_claims = claims[:5] if len(claims) > 5 else claims

        logger.info(f"Generated {len(result_claims)} claims via heuristic decomposition")
        logger.info(f"Claims: {result_claims}")

        return result_claims

    def _get_max_entailment(self, span_ids: List[int]) -> float:
        """Get maximum entailment score from a list of span IDs."""
        from lib.db import db

        if not span_ids:
            return 0.0

        # Fetch scores from claim_evidence_links
        placeholders = ",".join(["%s"] * len(span_ids))
        query = f"""
            SELECT MAX(entailment_score) as max_score
            FROM claim_evidence_links
            WHERE span_id IN ({placeholders})
        """

        result = db.fetch_one(query, tuple(span_ids))
        return result['max_score'] if result and result['max_score'] else 0.0

    def _format_output(self, query: str, claims: List[AtomicClaim], citation_map: Dict) -> Dict:
        """
        Constructs the final JSON report.

        Output schema:
        {
            "query": str,
            "pqe_score_projection": float (0-100),
            "verified_count": int,
            "total_claims": int,
            "claims": [AtomicClaim],
            "bibliography": [str]  # Full audit trails
        }
        """
        verified_count = sum(1 for c in claims if c.status == "VERIFIED")
        total_claims = len(claims)
        score = (verified_count / total_claims * 100) if total_claims > 0 else 0

        # Build bibliography (full audit trails)
        bibliography = []
        for c_list in citation_map.values():
            for cit in c_list:
                bibliography.append(cit.format_audit_trail())

        return {
            "query": query,
            "pqe_score_projection": score,
            "verified_count": verified_count,
            "total_claims": total_claims,
            "claims": [asdict(c) for c in claims],
            "bibliography": bibliography
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="PQE Response Generator - Evidence-Bound Answers"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="What does the test paper say about homeostasis?",
        help="Question to answer"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        default=False,
        help="Use hardcoded test claims (no LLM)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file (optional)"
    )

    args = parser.parse_args()

    # Generate response
    generator = PQEResponseGenerator()
    result = generator.generate_answer(args.query, demo_mode=args.demo)

    # Display results
    print("\n" + "="*60)
    print("FINAL PQE REPORT")
    print("="*60)
    print(json.dumps(result, indent=2))

    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to: {args.output}")
