from vdpo.types.contrast import (
    PRIOR_CIRCUIT_CONTRAST,
    VISUAL_CIRCUIT_CONTRAST,
    ContrastSpec,
)
from vdpo.types.enums import GroundingMode, ImageVariant


def test_prior_circuit_contrast_is_prompt_varying():
    assert PRIOR_CIRCUIT_CONTRAST.name == "prior_circuit"
    assert PRIOR_CIRCUIT_CONTRAST.source == (
        GroundingMode.PRIOR,
        ImageVariant.COUNTERFACTUAL,
    )
    assert PRIOR_CIRCUIT_CONTRAST.target == (
        GroundingMode.VISUAL,
        ImageVariant.COUNTERFACTUAL,
    )


def test_visual_circuit_contrast_is_image_varying():
    assert VISUAL_CIRCUIT_CONTRAST.name == "visual_circuit"
    assert VISUAL_CIRCUIT_CONTRAST.source == (
        GroundingMode.VISUAL,
        ImageVariant.ORIGINAL,
    )
    assert VISUAL_CIRCUIT_CONTRAST.target == (
        GroundingMode.VISUAL,
        ImageVariant.COUNTERFACTUAL,
    )


def test_from_name_round_trip():
    for spec in (PRIOR_CIRCUIT_CONTRAST, VISUAL_CIRCUIT_CONTRAST):
        assert ContrastSpec.from_name(spec.name) == spec


def test_from_name_rejects_unknown():
    import pytest

    with pytest.raises(ValueError, match="Unknown contrast"):
        ContrastSpec.from_name("nonsense")
