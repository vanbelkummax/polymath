#!/usr/bin/env python3
"""
Tautology Checker for Kill Tests

Prevents circular logic in experimental kill tests.
A kill test is tautological if:
1. Pass criterion measures the same thing as reconstruction target
2. "Independent" target is derived from same input
3. Success condition is trivially satisfied by construction

Example of CIRCULAR test:
- Input: Cell centroids
- Compressed: Persistence diagram
- Reconstruction: Cell centroids
- Kill test: "Reconstruction error correlates with persistence diagram distance"
- Problem: BOTH measure topological similarity → circular!

Example of VALID test:
- Input: H&E image
- Compressed: Topology features
- Reconstruction: Gene expression
- Kill test: "Topology predicts vascular density from IF (independent modality)"
- Valid: Vascular IF is NOT derived from H&E topology
"""

from dataclasses import dataclass
from typing import List, Optional, Set
from enum import Enum


class MeasurementType(Enum):
    """Types of measurements in biology."""
    MORPHOLOGY = "morphology"  # Cell shapes, tissue architecture
    TOPOLOGY = "topology"      # Betti numbers, persistence
    EXPRESSION = "expression"  # RNA/protein levels
    SPATIAL = "spatial"        # Cell positions, point patterns
    VASCULAR = "vascular"      # Blood vessels, vasculature
    IMMUNE = "immune"          # Immune cell infiltration
    MOLECULAR = "molecular"    # Mutations, pathways
    FUNCTIONAL = "functional"  # Survival, drug response


@dataclass
class DataSource:
    """Defines a data modality."""
    name: str
    modality: str  # "H&E", "IF", "RNA-seq", "Visium", etc.
    measurement_types: Set[MeasurementType]
    examples: List[str]


@dataclass
class KillTest:
    """Defines an experimental kill test."""
    input_source: DataSource
    reconstruction_target: DataSource
    pass_criterion: str
    pass_criterion_measures: Set[MeasurementType]
    independent_validation: Optional[DataSource] = None


class TautologyChecker:
    """Detects circular logic in kill tests."""

    @staticmethod
    def check_tautology(test: KillTest) -> List[str]:
        """
        Check if kill test is tautological.

        Returns:
            List of tautology errors (empty if valid)
        """
        errors = []

        # Rule 1: Pass criterion shouldn't measure same thing as reconstruction
        overlap = test.reconstruction_target.measurement_types & test.pass_criterion_measures
        if overlap:
            errors.append(
                f"TAUTOLOGY: Pass criterion measures {overlap}, "
                f"which overlaps with reconstruction target {test.reconstruction_target.measurement_types}. "
                f"This is circular - you're testing if X predicts X."
            )

        # Rule 2: If using independent validation, it must be from different modality
        if test.independent_validation:
            if test.independent_validation.modality == test.input_source.modality:
                errors.append(
                    f"PSEUDO-INDEPENDENCE: Validation uses same modality as input "
                    f"({test.independent_validation.modality}). Use orthogonal channel (e.g., IF if input is H&E)."
                )

            # Check if validation measures same thing as input
            val_input_overlap = test.independent_validation.measurement_types & test.input_source.measurement_types
            if val_input_overlap == test.independent_validation.measurement_types:
                errors.append(
                    f"DERIVED VALIDATION: Validation measures {test.independent_validation.measurement_types}, "
                    f"which is fully derivable from input {test.input_source.measurement_types}. "
                    f"Use truly independent ground truth."
                )

        # Rule 3: Reconstruction target should be different from input
        if test.reconstruction_target.modality == test.input_source.modality:
            if test.reconstruction_target.measurement_types == test.input_source.measurement_types:
                errors.append(
                    f"IDENTITY RECONSTRUCTION: Input and output measure the same thing "
                    f"({test.input_source.measurement_types}). This is trivial autoencoding."
                )

        return errors

    @staticmethod
    def suggest_fix(test: KillTest, errors: List[str]) -> Optional[str]:
        """Suggest how to fix tautological test."""
        if not errors:
            return None

        suggestions = []

        # If measuring topology → topology
        if MeasurementType.TOPOLOGY in test.reconstruction_target.measurement_types and \
           MeasurementType.TOPOLOGY in test.pass_criterion_measures:
            suggestions.append(
                "FIX: Use independent biological outcome. Examples:\n"
                "  - Predict vascular density from IF (CD31 stain)\n"
                "  - Predict patient survival (clinical endpoint)\n"
                "  - Predict immune infiltration (CD3/CD8 IF)"
            )

        # If measuring expression → expression
        if MeasurementType.EXPRESSION in test.reconstruction_target.measurement_types and \
           MeasurementType.EXPRESSION in test.pass_criterion_measures:
            suggestions.append(
                "FIX: Use orthogonal measurement. Examples:\n"
                "  - Predict spatial morphology (H&E) from expression\n"
                "  - Predict drug response (viability assay) from expression\n"
                "  - Predict mutation status (DNA-seq) from expression"
            )

        # If using same modality for validation
        if test.independent_validation and \
           test.independent_validation.modality == test.input_source.modality:
            suggestions.append(
                f"FIX: Use different imaging modality. Examples:\n"
                f"  - Input: H&E → Validation: IF (CD31, CD3, etc.)\n"
                f"  - Input: H&E → Validation: Visium ST (gene expression)\n"
                f"  - Input: Brightfield → Validation: Fluorescence"
            )

        return "\n\n".join(suggestions) if suggestions else None


# Example data sources
HE_IMAGE = DataSource(
    name="H&E Histology",
    modality="H&E",
    measurement_types={MeasurementType.MORPHOLOGY, MeasurementType.SPATIAL, MeasurementType.TOPOLOGY},
    examples=["Cell nuclei", "Tissue architecture", "Gland structures"]
)

IF_VASCULAR = DataSource(
    name="CD31 Immunofluorescence",
    modality="IF",
    measurement_types={MeasurementType.VASCULAR},
    examples=["Blood vessel density", "Vascular networks", "Endothelial cells"]
)

IF_IMMUNE = DataSource(
    name="CD3/CD8 Immunofluorescence",
    modality="IF",
    measurement_types={MeasurementType.IMMUNE},
    examples=["T cell density", "Immune infiltration", "Tumor-immune interface"]
)

SPATIAL_TRANSCRIPTOMICS = DataSource(
    name="Visium HD",
    modality="Spatial-RNA",
    measurement_types={MeasurementType.EXPRESSION, MeasurementType.SPATIAL},
    examples=["Gene expression per spot", "Spatial gene patterns", "Cell type abundance"]
)

CELL_CENTROIDS = DataSource(
    name="Cell Segmentation",
    modality="H&E-derived",
    measurement_types={MeasurementType.SPATIAL, MeasurementType.TOPOLOGY},
    examples=["Cell positions", "Spatial point pattern", "Cell density"]
)

PERSISTENCE_DIAGRAM = DataSource(
    name="Persistent Homology",
    modality="Topology-derived",
    measurement_types={MeasurementType.TOPOLOGY},
    examples=["Betti numbers", "Birth-death pairs", "Topological features"]
)

CLINICAL_OUTCOME = DataSource(
    name="Patient Survival",
    modality="Clinical",
    measurement_types={MeasurementType.FUNCTIONAL},
    examples=["Overall survival", "Disease-free survival", "Treatment response"]
)


def validate_kill_test(test: KillTest, verbose: bool = True) -> bool:
    """Validate kill test and print results."""
    if verbose:
        print(f"\n=== KILL TEST: {test.pass_criterion} ===")
        print(f"Input: {test.input_source.name} ({test.input_source.modality})")
        print(f"Reconstruction: {test.reconstruction_target.name} ({test.reconstruction_target.modality})")
        if test.independent_validation:
            print(f"Validation: {test.independent_validation.name} ({test.independent_validation.modality})")

    checker = TautologyChecker()
    errors = checker.check_tautology(test)

    if errors:
        if verbose:
            print("\n❌ TAUTOLOGICAL TEST DETECTED:")
            for err in errors:
                print(f"  • {err}")

            suggestion = checker.suggest_fix(test, errors)
            if suggestion:
                print(f"\n💡 SUGGESTED FIXES:\n{suggestion}")
        return False
    else:
        if verbose:
            print("\n✅ Kill test is valid (non-tautological)")
        return True


# Example tests
if __name__ == "__main__":
    print("="*80)
    print("EXAMPLE 1: TAUTOLOGICAL TEST (from PQE failure)")
    print("="*80)

    # BAD: Topology → Topology (circular)
    bad_test = KillTest(
        input_source=CELL_CENTROIDS,
        reconstruction_target=CELL_CENTROIDS,  # Same as input!
        pass_criterion="Reconstruction error correlates with persistence diagram distance",
        pass_criterion_measures={MeasurementType.TOPOLOGY}  # Measures topology!
    )
    validate_kill_test(bad_test)

    print("\n" + "="*80)
    print("EXAMPLE 2: VALID TEST (Fixed version)")
    print("="*80)

    # GOOD: Topology → Vascular (independent)
    good_test = KillTest(
        input_source=HE_IMAGE,
        reconstruction_target=SPATIAL_TRANSCRIPTOMICS,
        pass_criterion="Topological features predict vascular density from CD31 IF",
        pass_criterion_measures={MeasurementType.VASCULAR},
        independent_validation=IF_VASCULAR  # Independent modality!
    )
    validate_kill_test(good_test)

    print("\n" + "="*80)
    print("EXAMPLE 3: PSEUDO-INDEPENDENT TEST (Subtle error)")
    print("="*80)

    # BAD: Using H&E-derived validation (not truly independent)
    pseudo_test = KillTest(
        input_source=HE_IMAGE,
        reconstruction_target=SPATIAL_TRANSCRIPTOMICS,
        pass_criterion="Reconstruction quality measured on held-out H&E patches",
        pass_criterion_measures={MeasurementType.MORPHOLOGY},
        independent_validation=HE_IMAGE  # Same modality as input!
    )
    validate_kill_test(pseudo_test)

    print("\n" + "="*80)
    print("EXAMPLE 4: CLINICAL ENDPOINT (Gold standard)")
    print("="*80)

    # BEST: Using true clinical outcome
    clinical_test = KillTest(
        input_source=HE_IMAGE,
        reconstruction_target=SPATIAL_TRANSCRIPTOMICS,
        pass_criterion="Compressed features predict patient survival (Cox model p<0.05)",
        pass_criterion_measures={MeasurementType.FUNCTIONAL},
        independent_validation=CLINICAL_OUTCOME  # Truly independent!
    )
    validate_kill_test(clinical_test)
