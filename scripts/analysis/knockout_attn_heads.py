"""Analyze attention head knockout experiment results.

Loads all JSON outputs from outputs/knockout_attn_heads/ and prints:
1. Example counts per model
2. Group knockout flip rates (headline asymmetry test)
3. Individual head knockout flip rates
4. Logit margin analysis
"""

import json
import re
from collections import defaultdict
from pathlib import Path

from scripts.analysis._patching_common import (
    load_correct_example_ids,
    load_intersection_example_ids,
    print_section,
    sort_models,
)
from vdpo.settings import settings

KNOCKOUT_ATTN_HEADS_DIR = settings.output_dir / "knockout_attn_heads"
_DATA_DIR = Path(__file__).parent.parent.parent / "data"


def _knockout_dir_for_contrast(contrast: str) -> Path:
    if contrast == "prior_circuit":
        return KNOCKOUT_ATTN_HEADS_DIR
    return KNOCKOUT_ATTN_HEADS_DIR / contrast


def _classifications_path_for_contrast(contrast: str) -> Path:
    if contrast == "prior_circuit":
        return _DATA_DIR / "classify_attn_heads.json"
    return _DATA_DIR / f"classify_attn_heads_{contrast}.json"


def _correct_ids_for_contrast(contrast: str) -> dict[str, set[str]]:
    if contrast == "prior_circuit":
        return load_correct_example_ids()
    return load_intersection_example_ids()


def load_knockout_results(output_dir: Path, contrast: str) -> list[dict]:
    """Load all knockout attn heads JSON files into a flat list."""
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


def filter_results(results: list[dict], correct_ids: dict[str, set[str]]) -> list[dict]:
    """Keep only examples whose example_id is in correct_ids for the model."""
    return [
        r for r in results if r["example_id"] in correct_ids.get(model_key(r), set())
    ]


def compute_flip_rates(results: list[dict]) -> dict[str, dict]:
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
        promoting = cls.get("promoting", [])
        suppressing = cls.get("suppressing", [])
        n_promoting = len(promoting)
        n_suppressing = len(suppressing)
        promoting_heads = ", ".join(f"L{h['layer']}H{h['head']}" for h in promoting)
        suppressing_heads = ", ".join(f"L{h['layer']}H{h['head']}" for h in suppressing)
        print(
            f"  {model:<22} {n:>10}"
            f" {n_promoting:>12} ({promoting_heads})"
            f" {n_suppressing:>14} ({suppressing_heads})"
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


def print_individual_knockout_flip_rates(
    results: list[dict], classifications: dict, top_n: int = 10
):
    """Section 3: Per-head knockout flip rates, sorted by prior_flip_rate."""
    print_section("3. INDIVIDUAL HEAD KNOCKOUT FLIP RATES")

    grouped = defaultdict(list)
    for r in results:
        grouped[model_key(r)].append(r)

    models = sort_models(list(grouped.keys()))

    for model in models:
        rates = compute_flip_rates(grouped[model])
        cls = classifications.get(model, {})
        promoting_set = {(h["layer"], h["head"]) for h in cls.get("promoting", [])}
        suppressing_set = {(h["layer"], h["head"]) for h in cls.get("suppressing", [])}

        # Collect individual knockouts
        ind_knockouts = []
        for kid, r in rates.items():
            m = re.match(r"ind_L(\d+)H(\d+)", kid)
            if not m:
                continue
            layer_idx = int(m.group(1))
            head_idx = int(m.group(2))
            if (layer_idx, head_idx) in promoting_set:
                head_type = "promoting"
            elif (layer_idx, head_idx) in suppressing_set:
                head_type = "suppressing"
            else:
                head_type = "?"
            ind_knockouts.append((kid, layer_idx, head_idx, head_type, r))

        if not ind_knockouts:
            continue

        # Sort by prior_flip_rate descending, take top N
        ind_knockouts.sort(key=lambda x: x[4]["prior_flip_rate"], reverse=True)
        ind_knockouts = ind_knockouts[:top_n]

        print(f"\n  {model} (top {min(top_n, len(ind_knockouts))} by prior flip rate)")
        print(
            f"  {'Head':>10} {'Type':>14}"
            f" {'Visual flip':>14} {'Prior flip':>14} {'n':>6}"
        )
        print(f"  {'-' * 60}")

        for _kid, layer_idx, head_idx, head_type, r in ind_knockouts:
            head_label = f"L{layer_idx}H{head_idx}"
            print(
                f"  {head_label:>10} {head_type:>14}"
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


def print_compensation_analysis(results: list[dict], classifications: dict):
    """Section 5: Compensation analysis for group-flipping examples.

    For examples where grp_promoting flips prior grounding, checks how many
    individual promoting-head knockouts also flip prior grounding.
    Buckets into: 0 (fully compensated), 1, 2+ individual heads.
    Also prints per-head coverage (top 5 per model).
    """
    print_section("5. COMPENSATION ANALYSIS (grp_promoting prior flips)")
    print("  'Fully compensated' = group flips prior, but 0 individual heads do.")
    print("  Coverage = fraction of group-flipped examples also flipped by that head.")

    grouped = defaultdict(list)
    for r in results:
        grouped[model_key(r)].append(r)

    models = sort_models(list(grouped.keys()))

    for model in models:
        cls = classifications.get(model, {})
        promoting_set = {(h["layer"], h["head"]) for h in cls.get("promoting", [])}

        # Group by example_id → {knockout_id: result}
        by_example: dict[str, dict[str, dict]] = defaultdict(dict)
        for r in grouped[model]:
            by_example[r["example_id"]][r["knockout_id"]] = r

        # Find examples where grp_promoting flips prior grounding
        group_flipped_ids = [
            eid
            for eid, ko_map in by_example.items()
            if "grp_promoting" in ko_map
            and ko_map["grp_promoting"]["prior_grounding"]["predicted"] != "prior"
        ]

        if not group_flipped_ids:
            print(f"\n  {model}: no group-flipped examples found.")
            continue

        n_group_flipped = len(group_flipped_ids)

        # For each group-flipped example, count how many individual
        # promoting-head knockouts also flip prior grounding
        bucket_0, bucket_1, bucket_2plus = 0, 0, 0
        per_head_coverage: dict[str, int] = defaultdict(int)

        for eid in group_flipped_ids:
            ko_map = by_example[eid]
            ind_flips = 0
            for kid, r in ko_map.items():
                m = re.match(r"ind_L(\d+)H(\d+)", kid)
                if not m:
                    continue
                layer_idx = int(m.group(1))
                head_idx = int(m.group(2))
                if (layer_idx, head_idx) not in promoting_set:
                    continue
                if r["prior_grounding"]["predicted"] != "prior":
                    ind_flips += 1
                    per_head_coverage[kid] += 1
            if ind_flips == 0:
                bucket_0 += 1
            elif ind_flips == 1:
                bucket_1 += 1
            else:
                bucket_2plus += 1

        print(f"\n  {model} (group-flipped examples: {n_group_flipped})")
        print(f"  {'Bucket':<26} {'Count':>8} {'%':>8}")
        print(f"  {'-' * 44}")
        for label, count in [
            ("0 ind. heads flip (compensated)", bucket_0),
            ("1 ind. head flips", bucket_1),
            ("2+ ind. heads flip", bucket_2plus),
        ]:
            pct = count / n_group_flipped if n_group_flipped > 0 else 0.0
            print(f"  {label:<26} {count:>8} {pct:>7.1%}")

        # Per-head coverage — top 5
        top_heads = sorted(per_head_coverage.items(), key=lambda x: x[1], reverse=True)[
            :5
        ]
        if top_heads:
            print(f"\n  Top heads by coverage (of {n_group_flipped} group-flipped):")
            print(f"  {'Head':>10} {'Count':>8} {'Coverage':>10}")
            print(f"  {'-' * 30}")
            for kid, count in top_heads:
                cov = count / n_group_flipped
                print(f"  {kid:>10} {count:>8} {cov:>9.1%}")


def print_compensation_margin_analysis(results: list[dict], classifications: dict):
    """Section 6: Logit margin shift for fully-compensated examples.

    For the 'fully compensated' subset (grp_promoting flips prior, but 0
    individual promoting-head knockouts do), measures the mean margin shift
    caused by each individual head: baseline_margin - post_knockout_margin,
    where margin = logit_prior - logit_visual (positive = prior wins).
    Positive shift = knockout weakened prior dominance.
    """
    print_section("6. MARGIN SHIFT FOR FULLY-COMPENSATED EXAMPLES")
    print("  Margin = logit_prior - logit_visual (positive = prior wins).")
    print(
        "  Shift = baseline_margin - post_knockout_margin"
        " (positive = knockout weakened prior)."
    )

    inference_dir = settings.output_dir / "inference"

    grouped = defaultdict(list)
    for r in results:
        grouped[model_key(r)].append(r)

    models = sort_models(list(grouped.keys()))

    for model in models:
        cls = classifications.get(model, {})
        promoting_set = {(h["layer"], h["head"]) for h in cls.get("promoting", [])}

        # Group by example_id → {knockout_id: result}
        by_example: dict[str, dict[str, dict]] = defaultdict(dict)
        for r in grouped[model]:
            by_example[r["example_id"]][r["knockout_id"]] = r

        # Identify fully-compensated examples
        fully_compensated_ids = []
        for eid, ko_map in by_example.items():
            if "grp_promoting" not in ko_map:
                continue
            if ko_map["grp_promoting"]["prior_grounding"]["predicted"] == "prior":
                continue  # group didn't flip prior
            ind_flips = 0
            for kid, r in ko_map.items():
                m = re.match(r"ind_L(\d+)H(\d+)", kid)
                if not m:
                    continue
                layer_idx = int(m.group(1))
                head_idx = int(m.group(2))
                if (layer_idx, head_idx) not in promoting_set:
                    continue
                if r["prior_grounding"]["predicted"] != "prior":
                    ind_flips += 1
            if ind_flips == 0:
                fully_compensated_ids.append(eid)

        if not fully_compensated_ids:
            print(f"\n  {model}: no fully-compensated examples found.")
            continue

        # Load inference baselines for these examples
        model_family, model_size = model.split("/")
        baseline_margins: dict[str, float] = {}
        for eid in fully_compensated_ids:
            inf_path = (
                inference_dir
                / model_family
                / model_size
                / eid
                / "prior_counterfactual.json"
            )
            if not inf_path.exists():
                continue
            with inf_path.open() as f:
                inf = json.load(f)
            # correct = original color (prior answer), incorrect = counterfactual
            baseline_margins[eid] = (
                inf["correct_max_logit"] - inf["incorrect_max_logit"]
            )

        if not baseline_margins:
            print(
                f"\n  {model}: no inference baselines found"
                " for fully-compensated examples."
            )
            continue

        # Compute mean margin shift per individual promoting head
        head_shifts: dict[str, list[float]] = defaultdict(list)
        for eid in fully_compensated_ids:
            if eid not in baseline_margins:
                continue
            base_margin = baseline_margins[eid]
            ko_map = by_example[eid]
            for kid, r in ko_map.items():
                m = re.match(r"ind_L(\d+)H(\d+)", kid)
                if not m:
                    continue
                layer_idx = int(m.group(1))
                head_idx = int(m.group(2))
                if (layer_idx, head_idx) not in promoting_set:
                    continue
                pg = r["prior_grounding"]
                post_margin = pg["logit_prior"] - pg["logit_visual"]
                shift = base_margin - post_margin
                head_shifts[kid].append(shift)

        if not head_shifts:
            print(
                f"\n  {model}: no individual head data for fully-compensated examples."
            )
            continue

        n_fc = len(fully_compensated_ids)
        print(
            f"\n  {model}"
            f" (fully-compensated examples: {n_fc},"
            f" baselines loaded: {len(baseline_margins)})"
        )
        print(f"  {'Head':>10} {'n':>6} {'Mean shift':>12}")
        print(f"  {'-' * 30}")

        sorted_heads = sorted(
            head_shifts.items(),
            key=lambda x: sum(x[1]) / len(x[1]),
            reverse=True,
        )
        for kid, shifts in sorted_heads:
            mean_shift = sum(shifts) / len(shifts)
            print(f"  {kid:>10} {len(shifts):>6} {mean_shift:>+11.3f}")


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
    print(f"Loading attention head knockout results from {output_dir} ...")
    results = load_knockout_results(output_dir, args.contrast)
    print(f"Loaded {len(results)} result files.")

    print(f"Loading subset example IDs for contrast={args.contrast} ...")
    correct_ids = _correct_ids_for_contrast(args.contrast)
    for mk, ids in sorted(correct_ids.items()):
        print(f"  {mk}: {len(ids)} examples")

    results = filter_results(results, correct_ids)
    print(f"Filtered to {len(results)} results.")

    classifications_path = _classifications_path_for_contrast(args.contrast)
    print(f"Loading attention head classifications from {classifications_path} ...")
    with classifications_path.open() as f:
        classifications = json.load(f)

    print_example_counts(results, classifications)
    print_group_knockout_flip_rates(results)
    print_individual_knockout_flip_rates(results, classifications)
    print_logit_margin_analysis(results)
    print_compensation_analysis(results, classifications)
    print_compensation_margin_analysis(results, classifications)


if __name__ == "__main__":
    main()
