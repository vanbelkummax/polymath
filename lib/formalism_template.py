#!/usr/bin/env python3
"""
Formalism Definition Template

Enforces rigorous variable definition for PQE responses.
Prevents variable conflation (e.g., setting x = solution instead of x = observation).

Required Variables:
- s: Latent state (ground truth, unobserved)
- y: Observed measurement (actual input data)
- z: Compressed representation (information bottleneck)
- ŝ: Reconstructed state (output prediction)

Flow: s → y → z → ŝ
"""

from dataclasses import dataclass
from typing import Optional, List, Dict
from enum import Enum


class VariableType(Enum):
    """Types of variables in information flow."""
    LATENT = "s"          # Ground truth (unobserved)
    OBSERVED = "y"        # Measurement (input)
    COMPRESSED = "z"      # Representation (bottleneck)
    RECONSTRUCTED = "s_hat"  # Prediction (output)


@dataclass
class Variable:
    """Formal variable definition."""
    symbol: str
    name: str
    domain: str  # e.g., "ℝ^{H×W×3}" (pixels), "Δ^G" (simplex), "ℤ^2" (point process)
    interpretation: str  # Biological/physical meaning
    type: VariableType
    measured: bool  # Is this directly observed?
    examples: Optional[List[str]] = None


@dataclass
class InformationFlow:
    """Defines the s → y → z → ŝ flow."""
    s: Variable  # Latent state
    y: Variable  # Observation
    z: Variable  # Compressed representation
    s_hat: Variable  # Reconstruction

    def validate(self) -> List[str]:
        """Check for common formalism errors."""
        errors = []

        # Rule 1: s must be unobserved
        if self.s.measured:
            errors.append("ERROR: s (latent state) cannot be directly measured")

        # Rule 2: y must be observed
        if not self.y.measured:
            errors.append("ERROR: y (observation) must be directly measured")

        # Rule 3: s and y cannot be the same
        if self.s.domain == self.y.domain and self.s.interpretation == self.y.interpretation:
            errors.append("ERROR: s and y appear identical (variable conflation)")

        # Rule 4: z must be lower-dimensional than y (for compression)
        # This is heuristic - flag if not obvious
        if "×" in self.y.domain and "×" not in self.z.domain:
            pass  # Likely compressed (e.g., image → vector)
        elif self.z.domain == self.y.domain:
            errors.append("WARNING: z and y have same domain (is this compressed?)")

        # Rule 5: ŝ should match s's domain (reconstruction target)
        if self.s_hat.domain != self.s.domain:
            errors.append("WARNING: ŝ domain doesn't match s (reconstruction mismatch)")

        return errors

    def __str__(self) -> str:
        """Pretty-print the information flow."""
        return f"""
=== INFORMATION FLOW ===

s (Latent State):
  Domain: {self.s.domain}
  Meaning: {self.s.interpretation}
  Measured: {self.s.measured}

y (Observation):
  Domain: {self.y.domain}
  Meaning: {self.y.interpretation}
  Measured: {self.y.measured}

z (Compressed):
  Domain: {self.z.domain}
  Meaning: {self.z.interpretation}
  Measured: {self.z.measured}

ŝ (Reconstruction):
  Domain: {self.s_hat.domain}
  Meaning: {self.s_hat.interpretation}
  Measured: {self.s_hat.measured}

Flow: {self.s.symbol} → {self.y.symbol} → {self.z.symbol} → {self.s_hat.symbol}
"""


class FormalismBuilder:
    """Interactive builder for PQE formalism section."""

    @staticmethod
    def build_img2st_formalism() -> InformationFlow:
        """Example: Image-to-spatial-transcriptomics formalism."""

        s = Variable(
            symbol="s",
            name="Spatial Gene Expression",
            domain="ℝ^{N×G}",  # N spots, G genes
            interpretation="True spatial transcriptome (unobserved at 2μm)",
            type=VariableType.LATENT,
            measured=False,
            examples=["RNA counts per spot", "Visium HD ground truth"]
        )

        y = Variable(
            symbol="y",
            name="H&E Image",
            domain="ℝ^{H×W×3}",  # H×W pixels, RGB
            interpretation="Histology image (observed input)",
            type=VariableType.OBSERVED,
            measured=True,
            examples=["5μm resolution H&E", "224×224 patches"]
        )

        z = Variable(
            symbol="z",
            name="Tissue Representation",
            domain="ℝ^{N×D}",  # N spots, D-dim embedding
            interpretation="Compressed morphology features",
            type=VariableType.COMPRESSED,
            measured=False,
            examples=["UNI embeddings (1024-d)", "CONCH features (512-d)"]
        )

        s_hat = Variable(
            symbol="ŝ",
            name="Predicted Expression",
            domain="ℝ^{N×G}",  # Same as s
            interpretation="Reconstructed spatial transcriptome",
            type=VariableType.RECONSTRUCTED,
            measured=False,
            examples=["Predicted RNA from H&E", "Virtual ST"]
        )

        return InformationFlow(s=s, y=y, z=z, s_hat=s_hat)

    @staticmethod
    def build_lossy_biology_formalism() -> InformationFlow:
        """Example: Lossy biology formalism (corrected from PQE failure)."""

        s = Variable(
            symbol="s",
            name="Tissue Morphology State",
            domain="ℝ^{K}",  # K topological features
            interpretation="Ground truth tissue architecture (Betti numbers, cycles)",
            type=VariableType.LATENT,
            measured=False,
            examples=["β₀=connected components", "β₁=holes", "β₂=voids"]
        )

        y = Variable(
            symbol="y",
            name="Cell Centroid Image",
            domain="ℤ^2 point process",  # Spatial point pattern
            interpretation="Observed cell locations from segmentation",
            type=VariableType.OBSERVED,
            measured=True,
            examples=["Cellpose outputs", "QuPath detections"]
        )

        z = Variable(
            symbol="z",
            name="Persistence Diagram",
            domain="𝒫 = {(birth, death)}",  # Persistence pairs
            interpretation="Topological summary statistics",
            type=VariableType.COMPRESSED,
            measured=False,
            examples=["TDA features", "Persistent homology"]
        )

        s_hat = Variable(
            symbol="ŝ",
            name="Reconstructed Morphology",
            domain="ℝ^{K}",  # Same as s
            interpretation="Inferred tissue state from persistence",
            type=VariableType.RECONSTRUCTED,
            measured=False,
            examples=["Predicted Betti numbers", "Topological classification"]
        )

        return InformationFlow(s=s, y=y, z=z, s_hat=s_hat)


def validate_formalism(flow: InformationFlow) -> None:
    """Validate and print formalism."""
    print(flow)

    errors = flow.validate()
    if errors:
        print("\n❌ VALIDATION ERRORS:")
        for err in errors:
            print(f"  • {err}")
    else:
        print("\n✅ Formalism is valid")


# Example usage
if __name__ == "__main__":
    print("=== EXAMPLE 1: Image-to-ST (Correct) ===")
    img2st = FormalismBuilder.build_img2st_formalism()
    validate_formalism(img2st)

    print("\n" + "="*60 + "\n")

    print("=== EXAMPLE 2: Lossy Biology (Corrected) ===")
    lossy = FormalismBuilder.build_lossy_biology_formalism()
    validate_formalism(lossy)

    print("\n" + "="*60 + "\n")

    # BAD EXAMPLE: Variable conflation
    print("=== EXAMPLE 3: Variable Conflation (ERROR) ===")

    s_bad = Variable(
        symbol="s",
        name="Cell Centroids",  # This is the OBSERVATION, not latent!
        domain="ℤ^2",
        interpretation="Cell locations",
        type=VariableType.LATENT,
        measured=True,  # WRONG: latent shouldn't be measured
        examples=[]
    )

    y_bad = Variable(
        symbol="y",
        name="Cell Centroids",  # Same as s!
        domain="ℤ^2",
        interpretation="Cell locations",
        type=VariableType.OBSERVED,
        measured=True,
        examples=[]
    )

    z_bad = Variable(
        symbol="z",
        name="Persistence",
        domain="𝒫",
        interpretation="TDA features",
        type=VariableType.COMPRESSED,
        measured=False,
        examples=[]
    )

    s_hat_bad = Variable(
        symbol="ŝ",
        name="Reconstructed Centroids",
        domain="ℤ^2",
        interpretation="Predicted cell locations",
        type=VariableType.RECONSTRUCTED,
        measured=False,
        examples=[]
    )

    bad_flow = InformationFlow(s=s_bad, y=y_bad, z=z_bad, s_hat=s_hat_bad)
    validate_formalism(bad_flow)
