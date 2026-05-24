"""Analyze inference runner results.

Loads all JSON outputs from outputs/inference/ and prints:
1. Per-model accuracy across all 4 conditions (grounding x image variant)
2. Mean logit differences (original - counterfactual) per condition
3. Conflict analysis: prior+counterfactual condition in detail
4. Cross-model comparison table

Terminology:
- "original color" = the real-world color of the object (e.g., yellow for banana)
- "counterfactual color" = the manipulated color in the edited image (e.g., blue banana)
- In the JSON outputs, original color tokens are stored as "correct_*"
  and counterfactual color tokens as "incorrect_*" — a legacy naming convention
  from the dataset. We map these to original/counterfactual throughout.
"""

import json
from collections import defaultdict
from pathlib import Path

from vdpo.settings import settings
from vdpo.types.enums import GroundingMode, ImageVariant, ModelFamily


def load_inference_results(output_dir: Path) -> list[dict]:
    """Load all inference JSON files into a flat list."""
    inference_dir = output_dir / "inference"
    results = []
    for json_path in inference_dir.rglob("*.json"):
        # inference/{model_family}/{model_size}/{example_id}/{grounding}_{variant}.json
        parts = json_path.relative_to(inference_dir).parts
        model_family = parts[0]
        model_size = parts[1]
        with json_path.open() as f:
            data = json.load(f)
        data["model_family"] = model_family
        data["model_size"] = model_size
        results.append(data)
    return results


def predicts_expected(result: dict) -> bool:
    """Check if the model's argmax matches the expected answer for this condition.

    Under visual+counterfactual the model should report what it sees (the
    counterfactual color). Under all other conditions it should report the
    original color.
    """
    if (
        result["grounding_mode"] == GroundingMode.VISUAL
        and result["image_variant"] == ImageVariant.COUNTERFACTUAL
    ):
        # Expected answer is the counterfactual color (stored as "incorrect" in JSON)
        return result["next_token_id"] in result["incorrect_token_id"]
    # Expected answer is the original color (stored as "correct" in JSON)
    return result["next_token_id"] in result["correct_token_ids"]


def logit_diff_original_minus_counterfactual(result: dict) -> float:
    """Original color logit minus counterfactual color logit.

    Positive = model prefers the original (real-world) color.
    Negative = model prefers the counterfactual (manipulated) color.
    """
    return result["correct_max_logit"] - result["incorrect_max_logit"]


def predicts_original(result: dict) -> bool:
    """True if argmax lands on an original-color token."""
    return result["next_token_id"] in result["correct_token_ids"]


def predicts_counterfactual(result: dict) -> bool:
    """True if argmax lands on a counterfactual-color token."""
    return result["next_token_id"] in result["incorrect_token_id"]


def model_key(result: dict) -> str:
    return f"{result['model_family']}/{result['model_size']}"


def condition_key(result: dict) -> str:
    return f"{result['grounding_mode']}+{result['image_variant']}"


CONDITIONS = [f"{g.value}+{v.value}" for g in GroundingMode for v in ImageVariant]

MODEL_ORDER = [
    f"{f.value}/{s}" for f in ModelFamily for s in ("3B", "7B", "10B", "32B", "72B")
]


def print_section(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def analyze_accuracy(results: list[dict]):
    """Print accuracy table: models x conditions."""
    print_section("1. ACCURACY (% expected answer)")

    grouped = defaultdict(lambda: defaultdict(list))
    for r in results:
        grouped[model_key(r)][condition_key(r)].append(predicts_expected(r))

    models = sorted(
        grouped.keys(), key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 999
    )

    col_w = 16
    header = f"{'Model':<22}" + "".join(f"{c:>{col_w}}" for c in CONDITIONS)
    print(f"\n{header}")
    print("-" * len(header))

    for model in models:
        row = f"{model:<22}"
        for cond in CONDITIONS:
            vals = grouped[model][cond]
            if vals:
                acc = sum(vals) / len(vals) * 100
                row += f"{acc:>{col_w}.1f}%"
            else:
                row += f"{'N/A':>{col_w}}"
        n = len(grouped[model][CONDITIONS[0]]) if grouped[model][CONDITIONS[0]] else 0
        print(f"{row}  (n={n})")


def analyze_logit_differences(results: list[dict]):
    """Print mean logit difference table: models x conditions."""
    print_section("2. MEAN LOGIT DIFFERENCE (original - counterfactual)")

    grouped = defaultdict(lambda: defaultdict(list))
    for r in results:
        grouped[model_key(r)][condition_key(r)].append(
            logit_diff_original_minus_counterfactual(r)
        )

    models = sorted(
        grouped.keys(), key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 999
    )

    col_w = 16
    header = f"{'Model':<22}" + "".join(f"{c:>{col_w}}" for c in CONDITIONS)
    print(f"\n{header}")
    print("-" * len(header))

    for model in models:
        row = f"{model:<22}"
        for cond in CONDITIONS:
            vals = grouped[model][cond]
            if vals:
                mean = sum(vals) / len(vals)
                row += f"{mean:>{col_w}.2f}"
            else:
                row += f"{'N/A':>{col_w}}"
        print(row)


def analyze_conflict_condition(results: list[dict]):
    """Deep dive into prior+counterfactual: the core conflict condition."""
    print_section("3. CONFLICT ANALYSIS: prior + counterfactual")
    print("  (Model sees counterfactual image but is asked for memorized knowledge)")

    conflict_cond = f"{GroundingMode.PRIOR}+{ImageVariant.COUNTERFACTUAL}"
    baseline_cond = f"{GroundingMode.PRIOR}+{ImageVariant.ORIGINAL}"

    grouped = defaultdict(lambda: defaultdict(list))
    for r in results:
        grouped[model_key(r)][condition_key(r)].append(r)

    models = sorted(
        grouped.keys(), key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 999
    )

    for model in models:
        conflict_results = grouped[model][conflict_cond]
        baseline_results = grouped[model][baseline_cond]
        if not conflict_results:
            continue

        n = len(conflict_results)
        conflict_acc = sum(predicts_expected(r) for r in conflict_results) / n * 100
        conflict_mean_ld = (
            sum(logit_diff_original_minus_counterfactual(r) for r in conflict_results)
            / n
        )

        baseline_acc = 0.0
        baseline_mean_ld = 0.0
        if baseline_results:
            baseline_acc = (
                sum(predicts_expected(r) for r in baseline_results)
                / len(baseline_results)
                * 100
            )
            baseline_mean_ld = sum(
                logit_diff_original_minus_counterfactual(r) for r in baseline_results
            ) / len(baseline_results)

        print(f"\n  {model} (n={n})")
        print(
            f"    Prior+Original (baseline):       acc={baseline_acc:5.1f}%   mean_logit_diff={baseline_mean_ld:+.2f}"
        )
        print(
            f"    Prior+Counterfactual (conflict):  acc={conflict_acc:5.1f}%   mean_logit_diff={conflict_mean_ld:+.2f}"
        )
        print(
            f"    Accuracy drop from conflict:      {conflict_acc - baseline_acc:+.1f} pp"
        )

        # Logit difference distribution in conflict condition
        lds = sorted(
            logit_diff_original_minus_counterfactual(r) for r in conflict_results
        )
        q25 = lds[len(lds) // 4]
        q50 = lds[len(lds) // 2]
        q75 = lds[3 * len(lds) // 4]
        print(
            f"    Conflict logit diff quartiles:    Q25={q25:+.2f}  Q50={q50:+.2f}  Q75={q75:+.2f}"
        )

        # Count flips: predicted original on baseline -> predicted counterfactual on conflict
        if baseline_results:
            baseline_by_id = {r["example_id"]: r for r in baseline_results}
            flips = 0
            matched = 0
            for r in conflict_results:
                b = baseline_by_id.get(r["example_id"])
                if b and predicts_expected(b):
                    matched += 1
                    if not predicts_expected(r):
                        flips += 1
            if matched > 0:
                print(
                    f"    Flips (original -> counterfactual): {flips}/{matched} ({flips / matched * 100:.1f}%)"
                )


def analyze_argmax_breakdown(results: list[dict]):
    """In the conflict condition, where does the argmax land?"""
    print_section("4. ARGMAX BREAKDOWN (conflict condition)")
    print("  (Does the model predict the original color, the counterfactual color,")
    print("   or something else entirely?)")

    conflict_cond = f"{GroundingMode.PRIOR}+{ImageVariant.COUNTERFACTUAL}"

    grouped = defaultdict(list)
    for r in results:
        if condition_key(r) == conflict_cond:
            grouped[model_key(r)].append(r)

    models = sorted(
        grouped.keys(), key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 999
    )

    print(
        f"\n  {'Model':<22} {'Original (knowledge)':>22} {'Counterfactual (visual)':>24} {'Other':>10} {'Total':>8}"
    )
    print(f"  {'-' * 86}")

    for model in models:
        rs = grouped[model]
        n = len(rs)
        n_original = sum(1 for r in rs if predicts_original(r))
        n_counterfactual = sum(1 for r in rs if predicts_counterfactual(r))
        n_other = n - n_original - n_counterfactual
        print(
            f"  {model:<22} {n_original:>16} ({n_original / n * 100:4.1f}%)"
            f" {n_counterfactual:>18} ({n_counterfactual / n * 100:4.1f}%)"
            f" {n_other:>5} ({n_other / n * 100:4.1f}%)"
            f" {n:>8}"
        )


def analyze_cross_condition_summary(results: list[dict]):
    """Summary table comparing all conditions to highlight the conflict."""
    print_section("5. CROSS-CONDITION SUMMARY")

    grouped = defaultdict(lambda: defaultdict(list))
    for r in results:
        grouped[model_key(r)][condition_key(r)].append(r)

    models = sorted(
        grouped.keys(), key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 999
    )

    print("\n  Expected behavior if model always follows the prompt:")
    print("    visual+original:        report original color (no conflict)")
    print("    visual+counterfactual:   report counterfactual color (no conflict)")
    print("    prior+original:          report original color (no conflict)")
    print(
        "    prior+counterfactual:    report original color (conflict — visual input contradicts knowledge)"
    )
    print()

    col_w = 14
    header = f"  {'Model':<22}" + "".join(f"{c:>{col_w}}" for c in CONDITIONS)
    print(header)
    print(f"  {'-' * (22 + col_w * len(CONDITIONS))}")

    for model in models:
        row = f"  {model:<22}"
        for cond in CONDITIONS:
            vals = grouped[model][cond]
            if vals:
                acc = sum(predicts_expected(r) for r in vals) / len(vals) * 100
                row += f"{acc:>{col_w - 1}.1f}%"
            else:
                row += f"{'N/A':>{col_w}}"
        print(row)


def main():
    print("Loading inference results...")
    results = load_inference_results(settings.output_dir)
    print(f"Loaded {len(results)} result files.")

    analyze_accuracy(results)
    analyze_logit_differences(results)
    analyze_conflict_condition(results)
    analyze_argmax_breakdown(results)
    analyze_cross_condition_summary(results)


if __name__ == "__main__":
    main()
