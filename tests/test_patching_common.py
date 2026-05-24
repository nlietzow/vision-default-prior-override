import json
from pathlib import Path

from scripts.analysis._patching_common import (
    compute_restoration_scores,
    has_patching_data,
    is_visual_circuit_result,
    load_intersection_example_ids,
)


def _write_inference(
    inference_dir: Path,
    family: str,
    size: str,
    example_id: str,
    cell: str,
    next_token_id: int,
    correct: set[int],
    incorrect: set[int],
):
    """cell is e.g. 'visual_counterfactual'."""
    p = inference_dir / family / size / example_id / f"{cell}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {
                "next_token_id": next_token_id,
                "correct_token_ids": sorted(correct),
                "incorrect_token_id": sorted(incorrect),
            }
        )
    )


def test_intersection_keeps_only_three_way_correct(tmp_path):
    inf = tmp_path / "inference"
    # Example A satisfies all three (V,orig)->orig, (V,CF)->CF, (P,CF)->orig
    _write_inference(
        inf,
        "qwen",
        "3B",
        "A",
        "visual_original",
        next_token_id=10,
        correct={10},
        incorrect={11},
    )
    _write_inference(
        inf,
        "qwen",
        "3B",
        "A",
        "visual_counterfactual",
        next_token_id=11,
        correct={10},
        incorrect={11},
    )
    _write_inference(
        inf,
        "qwen",
        "3B",
        "A",
        "prior_counterfactual",
        next_token_id=10,
        correct={10},
        incorrect={11},
    )
    # Example B fails (V, original) -> CF instead of original
    _write_inference(
        inf,
        "qwen",
        "3B",
        "B",
        "visual_original",
        next_token_id=11,
        correct={10},
        incorrect={11},
    )
    _write_inference(
        inf,
        "qwen",
        "3B",
        "B",
        "visual_counterfactual",
        next_token_id=11,
        correct={10},
        incorrect={11},
    )
    _write_inference(
        inf,
        "qwen",
        "3B",
        "B",
        "prior_counterfactual",
        next_token_id=10,
        correct={10},
        incorrect={11},
    )

    ids = load_intersection_example_ids(inference_dir=inf)
    assert ids == {"qwen/3B": {"A"}}


def test_is_visual_circuit_result_detects_new_field_names():
    assert is_visual_circuit_result({"contrast": "visual_circuit"}) is True
    assert is_visual_circuit_result({"results_p2v": []}) is False
    # Missing both signals -> treat as legacy prior_circuit
    assert is_visual_circuit_result({}) is False


def test_has_patching_data_supports_both_field_names():
    assert has_patching_data({"results_p2v": [{}]}) is True
    assert has_patching_data({"results_p2v": []}) is False
    assert (
        has_patching_data(
            {"contrast": "visual_circuit", "results_source_to_target": [{}]}
        )
        is True
    )
    assert (
        has_patching_data(
            {"contrast": "visual_circuit", "results_source_to_target": []}
        )
        is False
    )


def _make_prior_circuit_result_s2t_full():
    """A prior-circuit result where p2v patching fully restores the source diff."""
    return {
        # source = prior baseline diff = +1.0 (prior logit > visual)
        "prior_token_logit_prior_run": 1.5,
        "visual_token_logit_prior_run": 0.5,
        # target = visual baseline diff = -1.0
        "prior_token_logit_visual_run": 0.5,
        "visual_token_logit_visual_run": 1.5,
        "results_p2v": [
            {
                "layer_idx_to_patch": 7,
                # patched diff matches source baseline (+1.0) -> full restoration
                "logit_prior": 1.5,
                "logit_visual": 0.5,
            }
        ],
    }


def test_compute_restoration_scores_prior_circuit_full_restoration():
    r = _make_prior_circuit_result_s2t_full()
    scores = compute_restoration_scores(r, "p2v")
    assert len(scores) == 1
    assert scores[0]["layer_idx"] == 7
    assert abs(scores[0]["score"] - 1.0) < 1e-6


def test_compute_restoration_scores_visual_circuit_full_restoration():
    r = {
        "contrast": "visual_circuit",
        "source_token_logit_source_run": 1.5,
        "target_token_logit_source_run": 0.5,
        "source_token_logit_target_run": 0.5,
        "target_token_logit_target_run": 1.5,
        "results_source_to_target": [
            {
                "layer_idx_to_patch": 12,
                "logit_source": 1.5,
                "logit_target": 0.5,
            }
        ],
    }
    scores = compute_restoration_scores(r, "s2t")
    assert len(scores) == 1
    assert abs(scores[0]["score"] - 1.0) < 1e-6


def test_compute_restoration_scores_visual_circuit_rejects_bad_direction():
    import pytest

    r = {"contrast": "visual_circuit", "results_source_to_target": [{}]}
    with pytest.raises(ValueError):
        compute_restoration_scores(r, "p2v")
