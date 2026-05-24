"""Analyze difference logit lens for classified attention heads.

For each promoting/suppressing head, checks whether the visual or prior
answer token appears in the pre-computed top-k/bottom-k of the difference
logit lens (visual - prior head output projected through W_O and W_U).
"""

import json
from pathlib import Path

import numpy as np

from scripts.analysis._patching_common import (
    load_correct_example_tokens,
    sort_models,
)
from vdpo.settings import settings

EXTRACT_DIR = settings.output_dir / "extract_attention_weights"
CLASSIFICATIONS_PATH = (
    Path(__file__).parent.parent.parent / "data" / "classify_attn_heads.json"
)


def load_classifications() -> dict:
    with CLASSIFICATIONS_PATH.open() as f:
        return json.load(f)


def load_extraction_examples(model_dir: Path) -> dict[str, tuple[dict, Path]]:
    """Load extraction metadata JSONs and return {example_id: (metadata, npz_path)}."""
    examples = {}
    for json_path in model_dir.glob("*.json"):
        npz_path = json_path.with_suffix(".npz")
        if not npz_path.exists():
            continue
        with json_path.open() as f:
            meta = json.load(f)
        examples[meta["example_id"]] = (meta, npz_path)
    return examples


def compute_hit_rates(
    classifications: dict,
    example_tokens: dict[str, dict[str, tuple[int, int]]],
) -> dict[str, dict]:
    """Compute per-head hit rates for classified heads across models.

    Uses inference-sourced answer tokens (visual=counterfactual color,
    prior=original color) as the ground truth, independent of what the
    extraction runner may have cached.

    Returns: {model_key: {
        "n_examples": int,
        "promoting": [{"layer", "head", "visual_in_top_k",
                        "prior_in_bottom_k", "n_examples"}],
        "suppressing": [{"layer", "head", "prior_in_top_k",
                         "visual_in_bottom_k", "n_examples"}],
    }}
    """
    results = {}

    for model_key in sort_models(list(classifications.keys())):
        clf = classifications[model_key]
        family, size = model_key.split("/")
        model_dir = EXTRACT_DIR / family / size

        if not model_dir.exists():
            continue

        examples = load_extraction_examples(model_dir)

        # Filter to correctly-conflicting examples using inference-sourced tokens
        valid_tokens = example_tokens.get(model_key, {})
        filtered = {
            eid: (meta, npz_path)
            for eid, (meta, npz_path) in examples.items()
            if eid in valid_tokens
        }

        n_examples = len(filtered)
        if n_examples == 0:
            continue

        # Accumulate counts per head: keyed by (layer, head)
        promoting_counts: dict[tuple[int, int], dict[str, int]] = {
            (h["layer"], h["head"]): {"visual_in_top": 0, "prior_in_bottom": 0}
            for h in clf["promoting"]
        }
        suppressing_counts: dict[tuple[int, int], dict[str, int]] = {
            (h["layer"], h["head"]): {"prior_in_top": 0, "visual_in_bottom": 0}
            for h in clf["suppressing"]
        }

        # Single pass over examples — load each npz once
        for eid, (_meta, npz_path) in filtered.items():
            visual_token, prior_token = valid_tokens[eid]
            with np.load(npz_path) as data:
                for layer, head in promoting_counts:
                    top_ids = data[f"diff_top_ids_layer_{layer}"][head]
                    bottom_ids = data[f"diff_bottom_ids_layer_{layer}"][head]
                    if visual_token in top_ids:
                        promoting_counts[(layer, head)]["visual_in_top"] += 1
                    if prior_token in bottom_ids:
                        promoting_counts[(layer, head)]["prior_in_bottom"] += 1

                for layer, head in suppressing_counts:
                    top_ids = data[f"diff_top_ids_layer_{layer}"][head]
                    bottom_ids = data[f"diff_bottom_ids_layer_{layer}"][head]
                    if prior_token in top_ids:
                        suppressing_counts[(layer, head)]["prior_in_top"] += 1
                    if visual_token in bottom_ids:
                        suppressing_counts[(layer, head)]["visual_in_bottom"] += 1

        # Build result lists in original order
        promoting_results = [
            {
                "layer": h["layer"],
                "head": h["head"],
                "visual_in_top_k": promoting_counts[(h["layer"], h["head"])][
                    "visual_in_top"
                ],
                "prior_in_bottom_k": promoting_counts[(h["layer"], h["head"])][
                    "prior_in_bottom"
                ],
                "n_examples": n_examples,
            }
            for h in clf["promoting"]
        ]

        suppressing_results = [
            {
                "layer": h["layer"],
                "head": h["head"],
                "prior_in_top_k": suppressing_counts[(h["layer"], h["head"])][
                    "prior_in_top"
                ],
                "visual_in_bottom_k": suppressing_counts[(h["layer"], h["head"])][
                    "visual_in_bottom"
                ],
                "n_examples": n_examples,
            }
            for h in clf["suppressing"]
        ]

        results[model_key] = {
            "n_examples": n_examples,
            "promoting": promoting_results,
            "suppressing": suppressing_results,
        }

    return results


def print_results(results: dict[str, dict]):
    for model_key in sort_models(list(results.keys())):
        r = results[model_key]
        n = r["n_examples"]

        print(f"\n{'=' * 70}")
        print(f"  Difference Logit Lens: {model_key} (n={n})")
        print(f"{'=' * 70}")

        # Promoting heads
        if r["promoting"]:
            print("\n  Promoting heads — visual answer in top-20:")
            print(f"  {'Head':<12} {'Hit rate':>10} {'Examples':>12}")
            print(f"  {'-' * 34}")
            for h in r["promoting"]:
                rate = h["visual_in_top_k"] / h["n_examples"] * 100
                print(
                    f"  L{h['layer']}H{h['head']:<8}"
                    f" {rate:>9.1f}%"
                    f" {h['visual_in_top_k']:>6}/{h['n_examples']}"
                )

            print("\n  Promoting heads — prior answer in bottom-20:")
            print(f"  {'Head':<12} {'Hit rate':>10} {'Examples':>12}")
            print(f"  {'-' * 34}")
            for h in r["promoting"]:
                rate = h["prior_in_bottom_k"] / h["n_examples"] * 100
                print(
                    f"  L{h['layer']}H{h['head']:<8}"
                    f" {rate:>9.1f}%"
                    f" {h['prior_in_bottom_k']:>6}/{h['n_examples']}"
                )

        # Suppressing heads
        if r["suppressing"]:
            print("\n  Suppressing heads — prior answer in top-20:")
            print(f"  {'Head':<12} {'Hit rate':>10} {'Examples':>12}")
            print(f"  {'-' * 34}")
            for h in r["suppressing"]:
                rate = h["prior_in_top_k"] / h["n_examples"] * 100
                print(
                    f"  L{h['layer']}H{h['head']:<8}"
                    f" {rate:>9.1f}%"
                    f" {h['prior_in_top_k']:>6}/{h['n_examples']}"
                )

            print("\n  Suppressing heads — visual answer in bottom-20:")
            print(f"  {'Head':<12} {'Hit rate':>10} {'Examples':>12}")
            print(f"  {'-' * 34}")
            for h in r["suppressing"]:
                rate = h["visual_in_bottom_k"] / h["n_examples"] * 100
                print(
                    f"  L{h['layer']}H{h['head']:<8}"
                    f" {rate:>9.1f}%"
                    f" {h['visual_in_bottom_k']:>6}/{h['n_examples']}"
                )

        # Summary
        print("\n  Summary:")
        if r["promoting"]:
            above = sum(
                1
                for h in r["promoting"]
                if h["visual_in_top_k"] / h["n_examples"] > 0.5
            )
            print(
                f"    Promoting with visual top-20 hit >50%:"
                f"     {above}/{len(r['promoting'])}"
            )
        if r["suppressing"]:
            above = sum(
                1
                for h in r["suppressing"]
                if h["prior_in_top_k"] / h["n_examples"] > 0.5
            )
            print(
                f"    Suppressing with prior top-20 hit >50%:"
                f"    {above}/{len(r['suppressing'])}"
            )


def save_results(results: dict[str, dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved results to {output_path}")


def main():
    print("Loading classifications...")
    classifications = load_classifications()
    print(f"  Models: {', '.join(sort_models(list(classifications.keys())))}")

    print("Loading correct example tokens...")
    example_tokens = load_correct_example_tokens()
    for mk in sort_models(list(example_tokens.keys())):
        print(f"  {mk}: {len(example_tokens[mk])} correctly-conflicting examples")

    print("\nComputing hit rates...")
    results = compute_hit_rates(classifications, example_tokens)
    print_results(results)
    save_results(
        results,
        Path(__file__).parent.parent.parent / "data" / "difference_logit_lens.json",
    )


if __name__ == "__main__":
    main()
