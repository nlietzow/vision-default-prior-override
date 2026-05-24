"""Analyze MLP knockout experiment results.

Loads all JSON outputs from outputs/knockout_mlp/ and prints:
1. Example counts per model
2. Group knockout flip rates (headline asymmetry test)
3. Individual layer knockout flip rates
4. Logit margin analysis
"""

import json
from collections import defaultdict
from pathlib import Path

from scripts.analysis._patching_common import (
    load_correct_example_ids,
    load_intersection_example_ids,
    print_section,
    sort_models,
)
from vdpo.settings import settings

KNOCKOUT_MLP_DIR = settings.output_dir / "knockout_mlp"
_DATA_DIR = Path(__file__).parent.parent.parent / "data"


def _knockout_dir_for_contrast(contrast: str) -> Path:
    if contrast == "prior_circuit":
        return KNOCKOUT_MLP_DIR
    return KNOCKOUT_MLP_DIR / contrast


def _classifications_path_for_contrast(contrast: str) -> Path:
    if contrast == "prior_circuit":
        return _DATA_DIR / "classify_mlp_layers.json"
    return _DATA_DIR / f"classify_mlp_layers_{contrast}.json"


def _correct_ids_for_contrast(contrast: str) -> dict[str, set[str]]:
    if contrast == "prior_circuit":
        return load_correct_example_ids()
    return load_intersection_example_ids()


def load_knockout_results(output_dir: Path, contrast: str) -> list[dict]:
    """Load all knockout MLP JSON files into a flat list."""
    results = []
    for json_path in output_dir.rglob("*.json"):
        parts = json_path.relative_to(output_dir).parts
        if contrast == "prior_circuit" and parts[0] == "visual_circuit":
            continue
        model_family = parts[0]
        model_size = parts[1]
        with json_path.open() as f:
            data = json.load(f)
        data["model_family"] = model_family
        data["model_size"] = model_size
        results.append(data)
    return results


def model_key(result: dict) -> str:
    return f"{result['model_family']}/{result['model_size']}"


def compute_flip_rates(results: list[dict]) -> dict[str, dict[str, float]]:
    """Compute flip rates for each knockout config within a single model.

    Returns: {knockout_id: {
        "n": int,
        "visual_flips": int,
        "prior_flips": int,
        "visual_flip_rate": float,
        "prior_flip_rate": float,
    }}
    """
    grouped = defaultdict(list)
    for r in results:
        grouped[r["knockout_id"]].append(r)

    rates = {}
    for knockout_id, rs in grouped.items():
        n = len(rs)
        visual_flips = sum(
            1 for r in rs if r["visual_grounding"]["predicted"] != "visual"
        )
        prior_flips = sum(1 for r in rs if r["prior_grounding"]["predicted"] != "prior")
        rates[knockout_id] = {
            "n": n,
            "visual_flips": visual_flips,
            "prior_flips": prior_flips,
            "visual_flip_rate": visual_flips / n if n > 0 else 0.0,
            "prior_flip_rate": prior_flips / n if n > 0 else 0.0,
        }
    return rates


def print_example_counts(results: list[dict], classifications: dict):
    """Section 1: Example counts per model."""
    print_section("1. EXAMPLE COUNTS")

    grouped = defaultdict(set)
    for r in results:
        grouped[model_key(r)].add(r["example_id"])

    models = sort_models(list(grouped.keys()))

    print(f"\n  {'Model':<22} {'Examples':>10} {'Promoting':>12} {'Suppressing':>14}")
    print(f"  {'-' * 60}")

    for model in models:
        n = len(grouped[model])
        cls = classifications.get(model, {})
        n_promoting = len(cls.get("promoting", []))
        n_suppressing = len(cls.get("suppressing", []))
        promoting_layers = ", ".join(f"L{layer}" for layer in cls.get("promoting", []))
        suppressing_layers = ", ".join(
            f"L{layer}" for layer in cls.get("suppressing", [])
        )
        print(
            f"  {model:<22} {n:>10}"
            f" {n_promoting:>12} ({promoting_layers})"
            f" {n_suppressing:>14} ({suppressing_layers})"
        )


def print_group_knockout_flip_rates(results: list[dict]):
    """Section 2: Group knockout flip rates — the headline asymmetry test."""
    print_section("2. GROUP KNOCKOUT FLIP RATES")
    print(
        "  Visual flip: knockout causes visual grounding to flip away from visual answer"
    )
    print(
        "  Prior flip: knockout causes prior grounding to flip away from prior answer"
    )

    grouped = defaultdict(list)
    for r in results:
        grouped[model_key(r)].append(r)

    models = sort_models(list(grouped.keys()))

    print(
        f"\n  {'Model':<22} {'Knockout':>16}"
        f" {'Visual flip':>14} {'Prior flip':>14} {'n':>6}"
    )
    print(f"  {'-' * 74}")

    for model in models:
        rates = compute_flip_rates(grouped[model])
        for knockout_id in ["grp_promoting", "grp_suppressing"]:
            if knockout_id not in rates:
                continue
            r = rates[knockout_id]
            print(
                f"  {model:<22} {knockout_id:>16}"
                f" {r['visual_flip_rate']:>12.1%}"
                f" {r['prior_flip_rate']:>12.1%}"
                f" {r['n']:>6}"
            )


def print_individual_knockout_flip_rates(results: list[dict], classifications: dict):
    """Section 3: Per-layer knockout flip rates."""
    print_section("3. INDIVIDUAL LAYER KNOCKOUT FLIP RATES")

    grouped = defaultdict(list)
    for r in results:
        grouped[model_key(r)].append(r)

    models = sort_models(list(grouped.keys()))

    for model in models:
        rates = compute_flip_rates(grouped[model])
        cls = classifications.get(model, {})
        promoting_set = set(cls.get("promoting", []))
        suppressing_set = set(cls.get("suppressing", []))

        # Collect individual knockouts, sorted by layer
        ind_knockouts = sorted(
            [(kid, r) for kid, r in rates.items() if kid.startswith("ind_")],
            key=lambda x: int(x[0].split("L")[1]),
        )

        if not ind_knockouts:
            continue

        print(f"\n  {model}")
        print(
            f"  {'Layer':>8} {'Type':>14}"
            f" {'Visual flip':>14} {'Prior flip':>14} {'n':>6}"
        )
        print(f"  {'-' * 58}")

        for kid, r in ind_knockouts:
            layer_idx = int(kid.split("L")[1])
            if layer_idx in promoting_set:
                head_type = "promoting"
            elif layer_idx in suppressing_set:
                head_type = "suppressing"
            else:
                head_type = "?"
            print(
                f"  {'L' + str(layer_idx):>8} {head_type:>14}"
                f" {r['visual_flip_rate']:>12.1%}"
                f" {r['prior_flip_rate']:>12.1%}"
                f" {r['n']:>6}"
            )


def print_logit_margin_analysis(results: list[dict]):
    """Section 4: Mean logit margin shift for group knockouts."""
    print_section("4. LOGIT MARGIN ANALYSIS (group knockouts)")
    print("  Margin = logit_visual - logit_prior (positive = favors visual)")

    grouped = defaultdict(list)
    for r in results:
        grouped[model_key(r)].append(r)

    models = sort_models(list(grouped.keys()))

    print(
        f"\n  {'Model':<22} {'Knockout':>16} {'Visual margin':>16} {'Prior margin':>16}"
    )
    print(f"  {'-' * 72}")

    for model in models:
        by_knockout = defaultdict(list)
        for r in grouped[model]:
            if r["knockout_mode"] == "group":
                by_knockout[r["knockout_id"]].append(r)

        for knockout_id in ["grp_promoting", "grp_suppressing"]:
            rs = by_knockout.get(knockout_id, [])
            if not rs:
                continue

            visual_margins = [
                r["visual_grounding"]["logit_visual"]
                - r["visual_grounding"]["logit_prior"]
                for r in rs
            ]
            prior_margins = [
                r["prior_grounding"]["logit_visual"]
                - r["prior_grounding"]["logit_prior"]
                for r in rs
            ]

            mean_visual = sum(visual_margins) / len(visual_margins)
            mean_prior = sum(prior_margins) / len(prior_margins)

            print(
                f"  {model:<22} {knockout_id:>16}"
                f" {mean_visual:>+14.2f}"
                f" {mean_prior:>+14.2f}"
            )


def filter_to_correct_examples(
    results: list[dict], correct_ids: dict[str, set[str]]
) -> list[dict]:
    """Keep only correctly-conflicting examples (same filter as patching analyses)."""
    return [
        r for r in results if r["example_id"] in correct_ids.get(model_key(r), set())
    ]


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contrast",
        type=str,
        default="prior_circuit",
        choices=["prior_circuit", "visual_circuit"],
    )
    args = parser.parse_args()

    output_dir = _knockout_dir_for_contrast(args.contrast)
    print(f"Loading MLP knockout results from {output_dir} ...")
    all_results = load_knockout_results(output_dir, args.contrast)
    print(f"Loaded {len(all_results)} result files.")

    print(f"Loading subset example IDs for contrast={args.contrast} ...")
    correct_ids = _correct_ids_for_contrast(args.contrast)
    for mk, ids in sorted(correct_ids.items()):
        print(f"  {mk}: {len(ids)} examples")

    results = filter_to_correct_examples(all_results, correct_ids)
    print(f"Filtered to {len(results)} result files.")

    classifications_path = _classifications_path_for_contrast(args.contrast)
    print(f"Loading MLP classifications from {classifications_path} ...")
    with classifications_path.open() as f:
        classifications = json.load(f)

    print_example_counts(results, classifications)
    print_group_knockout_flip_rates(results)
    print_individual_knockout_flip_rates(results, classifications)
    print_logit_margin_analysis(results)


if __name__ == "__main__":
    main()
