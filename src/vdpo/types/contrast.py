from dataclasses import dataclass

from vdpo.types.enums import GroundingMode, ImageVariant


@dataclass(frozen=True)
class ContrastSpec:
    name: str
    source: tuple[GroundingMode, ImageVariant]
    target: tuple[GroundingMode, ImageVariant]

    @classmethod
    def from_name(cls, name: str) -> "ContrastSpec":
        for spec in (PRIOR_CIRCUIT_CONTRAST, VISUAL_CIRCUIT_CONTRAST):
            if spec.name == name:
                return spec
        raise ValueError(
            f"Unknown contrast {name!r}. "
            f"Expected one of: prior_circuit, visual_circuit."
        )


PRIOR_CIRCUIT_CONTRAST = ContrastSpec(
    name="prior_circuit",
    source=(GroundingMode.PRIOR, ImageVariant.COUNTERFACTUAL),
    target=(GroundingMode.VISUAL, ImageVariant.COUNTERFACTUAL),
)


VISUAL_CIRCUIT_CONTRAST = ContrastSpec(
    name="visual_circuit",
    source=(GroundingMode.VISUAL, ImageVariant.ORIGINAL),
    target=(GroundingMode.VISUAL, ImageVariant.COUNTERFACTUAL),
)
