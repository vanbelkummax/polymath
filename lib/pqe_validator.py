#!/usr/bin/env python3
"""
PQE Response Validation Checklist

Integrates all quality gates to prevent failures:
1. Citation Verification (no hallucinated DOIs)
2. Formalism Definition (s → y → z → ŝ)
3. Tautology Checker (non-circular kill tests)

Usage:
    validator = PQEValidator()
    result = validator.validate_response(pqe_response)
    if result.is_passing():
        print("Ready to submit")
    else:
        print(result.failure_report())
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum
import re

from citation_verifier import CitationVerifier, Citation
from formalism_template import InformationFlow, Variable
from tautology_checker import TautologyChecker, KillTest


class ValidationLevel(Enum):
    """Severity of validation issues."""
    CRITICAL = "CRITICAL"    # Will fail peer review
    WARNING = "WARNING"      # May get flagged
    INFO = "INFO"           # Style/preference


@dataclass
class ValidationIssue:
    """Single validation issue."""
    level: ValidationLevel
    category: str  # "citation", "formalism", "kill_test", "completeness"
    message: str
    location: Optional[str] = None  # Section where issue occurs
    suggested_fix: Optional[str] = None


@dataclass
class ValidationResult:
    """Complete validation result."""
    issues: List[ValidationIssue]
    citations_validated: int
    formalism_validated: bool
    kill_test_validated: bool
    all_sections_present: bool

    def is_passing(self) -> bool:
        """Check if response passes all critical checks."""
        critical_issues = [i for i in self.issues if i.level == ValidationLevel.CRITICAL]
        return len(critical_issues) == 0 and self.all_sections_present

    def score_estimate(self) -> int:
        """Estimate PQE score based on issues."""
        base_score = 100

        # Deduct for critical issues
        for issue in self.issues:
            if issue.level == ValidationLevel.CRITICAL:
                base_score -= 10
            elif issue.level == ValidationLevel.WARNING:
                base_score -= 5
            elif issue.level == ValidationLevel.INFO:
                base_score -= 2

        # Missing sections
        if not self.all_sections_present:
            base_score -= 20

        return max(0, base_score)

    def failure_report(self) -> str:
        """Generate detailed failure report."""
        report = []
        report.append("="*80)
        report.append("PQE VALIDATION REPORT")
        report.append("="*80)

        if self.is_passing():
            report.append(f"\n✅ PASSING (Estimated score: {self.score_estimate()}/100)")
            report.append("\nNo critical issues detected. Response is ready for submission.")
        else:
            report.append(f"\n❌ FAILING (Estimated score: {self.score_estimate()}/100)")
            report.append("\nCritical issues must be fixed before submission:")

        # Group by level
        critical = [i for i in self.issues if i.level == ValidationLevel.CRITICAL]
        warnings = [i for i in self.issues if i.level == ValidationLevel.WARNING]
        info = [i for i in self.issues if i.level == ValidationLevel.INFO]

        if critical:
            report.append(f"\n🔴 CRITICAL ISSUES ({len(critical)}):")
            for i, issue in enumerate(critical, 1):
                report.append(f"\n{i}. [{issue.category}] {issue.message}")
                if issue.location:
                    report.append(f"   Location: {issue.location}")
                if issue.suggested_fix:
                    report.append(f"   Fix: {issue.suggested_fix}")

        if warnings:
            report.append(f"\n⚠️  WARNINGS ({len(warnings)}):")
            for i, issue in enumerate(warnings, 1):
                report.append(f"\n{i}. [{issue.category}] {issue.message}")
                if issue.suggested_fix:
                    report.append(f"   Fix: {issue.suggested_fix}")

        if info:
            report.append(f"\n💡 SUGGESTIONS ({len(info)}):")
            for i, issue in enumerate(info, 1):
                report.append(f"\n{i}. [{issue.category}] {issue.message}")

        # Summary
        report.append(f"\n{'='*80}")
        report.append("SUMMARY:")
        report.append(f"  Citations validated: {self.citations_validated}")
        report.append(f"  Formalism validated: {'✅' if self.formalism_validated else '❌'}")
        report.append(f"  Kill test validated: {'✅' if self.kill_test_validated else '❌'}")
        report.append(f"  All sections present: {'✅' if self.all_sections_present else '❌'}")
        report.append(f"  Estimated score: {self.score_estimate()}/100")
        report.append("="*80)

        return "\n".join(report)


@dataclass
class PQEResponse:
    """Structured PQE response."""
    title: str
    citations: List[Citation]
    formalism: Optional[InformationFlow]
    mechanism_table: Optional[str]
    predictions: List[str]
    kill_test: Optional[KillTest]
    implementation: Optional[str]
    limitations: Optional[str]
    extensions: Optional[str]


class PQEValidator:
    """Comprehensive PQE response validator."""

    def __init__(self):
        self.citation_verifier = CitationVerifier()
        self.tautology_checker = TautologyChecker()

    def validate_response(self, response: PQEResponse) -> ValidationResult:
        """Run all validation checks."""
        issues = []

        # 1. Check citations
        citation_issues = self._validate_citations(response.citations)
        issues.extend(citation_issues)

        # 2. Check formalism
        formalism_issues = self._validate_formalism(response.formalism)
        issues.extend(formalism_issues)

        # 3. Check kill test
        kill_test_issues = self._validate_kill_test(response.kill_test)
        issues.extend(kill_test_issues)

        # 4. Check completeness
        completeness_issues = self._validate_completeness(response)
        issues.extend(completeness_issues)

        return ValidationResult(
            issues=issues,
            citations_validated=len(response.citations) - len(citation_issues),
            formalism_validated=len(formalism_issues) == 0 and response.formalism is not None,
            kill_test_validated=len(kill_test_issues) == 0 and response.kill_test is not None,
            all_sections_present=len(completeness_issues) == 0
        )

    def _validate_citations(self, citations: List[Citation]) -> List[ValidationIssue]:
        """Validate all citations."""
        issues = []

        for cit in citations:
            # Check for unverified DOIs
            if cit.doi and not cit.verified:
                issues.append(ValidationIssue(
                    level=ValidationLevel.CRITICAL,
                    category="citation",
                    message=f"Unverified DOI: {cit.doi} for '{cit.title[:50]}...'",
                    suggested_fix="Use WebSearch to verify DOI matches title, or cite by title+URL only"
                ))

            # Check for missing year
            if not cit.year:
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    category="citation",
                    message=f"Missing year for '{cit.title[:50]}...'",
                    suggested_fix="Add publication year from search results"
                ))

            # Check for missing authors
            if not cit.authors:
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    category="citation",
                    message=f"Missing authors for '{cit.title[:50]}...'",
                    suggested_fix="Add first author from search results"
                ))

        return issues

    def _validate_formalism(self, formalism: Optional[InformationFlow]) -> List[ValidationIssue]:
        """Validate formalism definition."""
        issues = []

        if formalism is None:
            issues.append(ValidationIssue(
                level=ValidationLevel.CRITICAL,
                category="formalism",
                message="No formalism defined (s → y → z → ŝ missing)",
                location="Section 2: Formalism",
                suggested_fix="Use formalism_template.py to define information flow"
            ))
            return issues

        # Run formalism validation
        formalism_errors = formalism.validate()
        for err in formalism_errors:
            level = ValidationLevel.CRITICAL if "ERROR" in err else ValidationLevel.WARNING
            issues.append(ValidationIssue(
                level=level,
                category="formalism",
                message=err,
                location="Section 2: Formalism",
                suggested_fix="Fix variable definitions (see formalism_template.py examples)"
            ))

        return issues

    def _validate_kill_test(self, kill_test: Optional[KillTest]) -> List[ValidationIssue]:
        """Validate kill test for tautology."""
        issues = []

        if kill_test is None:
            issues.append(ValidationIssue(
                level=ValidationLevel.CRITICAL,
                category="kill_test",
                message="No kill test defined",
                location="Section 6: Experimental Kill Test",
                suggested_fix="Add experimental validation with independent ground truth"
            ))
            return issues

        # Run tautology check
        tautology_errors = self.tautology_checker.check_tautology(kill_test)
        for err in tautology_errors:
            issues.append(ValidationIssue(
                level=ValidationLevel.CRITICAL,
                category="kill_test",
                message=err,
                location="Section 6: Experimental Kill Test",
                suggested_fix=self.tautology_checker.suggest_fix(kill_test, [err])
            ))

        return issues

    def _validate_completeness(self, response: PQEResponse) -> List[ValidationIssue]:
        """Check if all required sections are present."""
        issues = []

        required_sections = {
            "citations": response.citations,
            "formalism": response.formalism,
            "mechanism_table": response.mechanism_table,
            "predictions": response.predictions,
            "kill_test": response.kill_test,
            "implementation": response.implementation,
            "limitations": response.limitations,
            "extensions": response.extensions
        }

        for section_name, section_content in required_sections.items():
            if section_content is None or (isinstance(section_content, list) and len(section_content) == 0):
                issues.append(ValidationIssue(
                    level=ValidationLevel.CRITICAL,
                    category="completeness",
                    message=f"Missing required section: {section_name}",
                    location=f"Section: {section_name}",
                    suggested_fix=f"Add {section_name} section per PQE schema"
                ))

        return issues


# Example usage
if __name__ == "__main__":
    from formalism_template import FormalismBuilder
    from tautology_checker import *

    # Example 1: GOOD response (should pass)
    print("="*80)
    print("EXAMPLE 1: VALID PQE RESPONSE")
    print("="*80)

    good_citations = [
        Citation(
            title="Rethinking Lossy Compression: The Rate-Distortion-Perception Tradeoff",
            authors="Blau, Michaeli",
            year=2019,
            venue="ICML",
            url="https://arxiv.org/abs/1901.07821",
            verified=False,  # No DOI, so not verified (but that's OK with URL)
            source="search"
        )
    ]

    good_formalism = FormalismBuilder.build_lossy_biology_formalism()

    good_kill_test = KillTest(
        input_source=HE_IMAGE,
        reconstruction_target=SPATIAL_TRANSCRIPTOMICS,
        pass_criterion="Topological features predict vascular density from CD31 IF",
        pass_criterion_measures={MeasurementType.VASCULAR},
        independent_validation=IF_VASCULAR
    )

    good_response = PQEResponse(
        title="Lossy Biology via Rate-Distortion",
        citations=good_citations,
        formalism=good_formalism,
        mechanism_table="Domain A | Domain B | Correspondence",
        predictions=["Prediction 1", "Prediction 2", "Prediction 3"],
        kill_test=good_kill_test,
        implementation="Implementation plan",
        limitations="Limitation 1, 2, 3",
        extensions="Extension 1, 2, 3"
    )

    validator = PQEValidator()
    result = validator.validate_response(good_response)
    print(result.failure_report())

    # Example 2: BAD response (from PQE failure)
    print("\n\n")
    print("="*80)
    print("EXAMPLE 2: FAILING PQE RESPONSE (PQE v1 errors)")
    print("="*80)

    bad_citations = [
        Citation(
            title="Persistent Homology Classifies Parameter Dependence of Patterns in Turing Systems",
            doi="s41467-023-36796-3",  # WRONG DOI (this is GraphST!)
            year=2024,
            verified=False,  # NOT VERIFIED!
            source="hallucination"
        )
    ]

    # Bad formalism (variable conflation)
    s_bad = Variable(
        symbol="s",
        name="Cell Centroids",
        domain="ℤ^2",
        interpretation="Cell locations",
        type=None,
        measured=True  # WRONG: latent shouldn't be measured!
    )
    y_bad = Variable(symbol="y", name="Cell Centroids", domain="ℤ^2", interpretation="Cell locations", type=None, measured=True)
    z_bad = Variable(symbol="z", name="Persistence", domain="𝒫", interpretation="TDA", type=None, measured=False)
    s_hat_bad = Variable(symbol="ŝ", name="Reconstructed", domain="ℤ^2", interpretation="Predicted centroids", type=None, measured=False)
    bad_formalism = InformationFlow(s=s_bad, y=y_bad, z=z_bad, s_hat=s_hat_bad)

    # Bad kill test (tautology)
    bad_kill_test = KillTest(
        input_source=CELL_CENTROIDS,
        reconstruction_target=CELL_CENTROIDS,
        pass_criterion="Reconstruction error correlates with persistence diagram distance",
        pass_criterion_measures={MeasurementType.TOPOLOGY}
    )

    bad_response = PQEResponse(
        title="Lossy Biology",
        citations=bad_citations,
        formalism=bad_formalism,
        mechanism_table=None,  # MISSING!
        predictions=["Only 1 prediction"],  # Too few
        kill_test=bad_kill_test,
        implementation=None,  # MISSING!
        limitations=None,  # MISSING!
        extensions=None  # MISSING!
    )

    result2 = validator.validate_response(bad_response)
    print(result2.failure_report())
