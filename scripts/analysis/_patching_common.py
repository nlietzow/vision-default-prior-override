"""Shared utilities for patching analysis scripts.

Used by both patching_last_token_res_stream.py and patching_last_token_mlp.py.
The JSON output format is identical across patching runners.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from vdpo.settings import settings
from vdpo.types.enums import ModelFamily

INFERENCE_DIR = settings.output_dir / "inference"

MODEL_ORDER = [
    f"{f.value}/{s}" for f in ModelFamily for s in ("3B", "7B", "10B", "32B", "72B")
]


def load_patching_results(output_dir: Path) -> list[dict]:
    """Load all patching JSON files into a flat list."""
    results = []
    for json_path in output_dir.rglob("*.json"):
        parts = json_path.relative_to(output_dir).parts
        model_family = parts[0]
        model_size = parts[1]
        with json_path.open() as f:
            data = json.load(f)
        data["model_family"] = model_family
        data["model_size"] = model_size
        results.append(data)
    return results


def load_correct_example_ids(
    inference_dir: Path | None = None,
) -> dict[str, set[str]]:
    """Load example IDs where the model correctly answers both groundings.

    An example is 'correctly conflicting' if:
    - visual+counterfactual: model predicts the counterfactual color
    - prior+counterfactual: model predicts the knowledge color

    Returns: {model_key: set of example_ids}
    """
    if inference_dir is None:
        inference_dir = INFERENCE_DIR

    correct_ids = defaultdict(set)

    for model_family_dir in inference_dir.iterdir():
        if not model_family_dir.is_dir():
            continue
        for model_size_dir in model_family_dir.iterdir():
            if not model_size_dir.is_dir():
                continue
            mk = f"{model_family_dir.name}/{model_size_dir.name}"

            for example_dir in model_size_dir.iterdir():
                if not example_dir.is_dir():
                    continue
                example_id = example_dir.name

                visual_cf = example_dir / "visual_counterfactual.json"
                prior_cf = example_dir / "prior_counterfactual.json"
                if not visual_cf.exists() or not prior_cf.exists():
                    continue

                with visual_cf.open() as f:
                    v_data = json.load(f)
                with prior_cf.open() as f:
                    p_data = json.load(f)

                visual_ok = v_data["next_token_id"] in v_data["incorrect_token_id"]
                prior_ok = p_data["next_token_id"] in p_data["correct_token_ids"]

                if visual_ok and prior_ok:
                    correct_ids[mk].add(example_id)

    return correct_ids


def load_intersection_example_ids(
    inference_dir: Path | None = None,
) -> dict[str, set[str]]:
    """Keep examples that satisfy all three:
    - (V, original)        -> original color
    - (V, counterfactual)  -> CF color
    - (P, counterfactual)  -> original color

    This is the intersection of the visual-circuit and prior-circuit subsets,
    so analyses on both contrasts run on the same examples and can be compared
    head-set-to-head-set without confounding from differing sample membership.
    """
    if inference_dir is None:
        inference_dir = INFERENCE_DIR

    ids = defaultdict(set)

    for model_family_dir in inference_dir.iterdir():
        if not model_family_dir.is_dir():
            continue
        for model_size_dir in model_family_dir.iterdir():
            if not model_size_dir.is_dir():
                continue
            mk = f"{model_family_dir.name}/{model_size_dir.name}"

            for example_dir in model_size_dir.iterdir():
                if not example_dir.is_dir():
                    continue
                example_id = example_dir.name

                v_orig = example_dir / "visual_original.json"
                v_cf = example_dir / "visual_counterfactual.json"
                p_cf = example_dir / "prior_counterfactual.json"
                if not (v_orig.exists() and v_cf.exists() and p_cf.exists()):
                    continue

                with v_orig.open() as f:
                    v_orig_d = json.load(f)
                with v_cf.open() as f:
                    v_cf_d = json.load(f)
                with p_cf.open() as f:
                    p_cf_d = json.load(f)

                visual_original_ok = (
                    v_orig_d["next_token_id"] in v_orig_d["correct_token_ids"]
                )
                visual_cf_ok = v_cf_d["next_token_id"] in v_cf_d["incorrect_token_id"]
                prior_cf_ok = p_cf_d["next_token_id"] in p_cf_d["correct_token_ids"]

                if visual_original_ok and visual_cf_ok and prior_cf_ok:
                    ids[mk].add(example_id)

    return ids


def is_visual_circuit_result(result: dict) -> bool:
    return result.get("contrast") == "visual_circuit"


def load_correct_example_tokens(
    inference_dir: Path | None = None,
) -> dict[str, dict[str, tuple[int, int]]]:
    """Load answer token IDs for correctly-conflicting examples from inference outputs.

    Uses inference outputs as the source of truth, independent of what any
    other runner may have cached. This avoids issues with non-deterministic
    borderline examples where different runs produce different predictions.

    Returns: {model_key: {example_id: (visual_token_id, prior_token_id)}}
    where visual_token_id is the counterfactual-color token and
    prior_token_id is the original-color token.
    """
    if inference_dir is None:
        inference_dir = INFERENCE_DIR

    tokens: dict[str, dict[str, tuple[int, int]]] = defaultdict(dict)

    for model_family_dir in inference_dir.iterdir():
        if not model_family_dir.is_dir():
            continue
        for model_size_dir in model_family_dir.iterdir():
            if not model_size_dir.is_dir():
                continue
            mk = f"{model_family_dir.name}/{model_size_dir.name}"

            for example_dir in model_size_dir.iterdir():
                if not example_dir.is_dir():
                    continue
                example_id = example_dir.name

                visual_cf = example_dir / "visual_counterfactual.json"
                prior_cf = example_dir / "prior_counterfactual.json"
                if not visual_cf.exists() or not prior_cf.exists():
                    continue

                with visual_cf.open() as f:
                    v_data = json.load(f)
                with prior_cf.open() as f:
                    p_data = json.load(f)

                visual_ok = v_data["next_token_id"] in v_data["incorrect_token_id"]
                prior_ok = p_data["next_token_id"] in p_data["correct_token_ids"]

                if visual_ok and prior_ok:
                    tokens[mk][example_id] = (
                        v_data["next_token_id"],
                        p_data["next_token_id"],
                    )

    return tokens


def model_key(result: dict) -> str:
    return f"{result['model_family']}/{result['model_size']}"


def has_patching_data(result: dict) -> bool:
    """Whether patching was run (source and target predictions disagree)."""
    if is_visual_circuit_result(result):
        return len(result.get("results_source_to_target", [])) > 0
    return len(result.get("results_p2v", [])) > 0


def filter_results(results: list[dict], correct_ids: dict[str, set[str]]) -> list[dict]:
    """Keep only examples that are correctly conflicting and have patching data."""
    return [
        r
        for r in results
        if r["example_id"] in correct_ids.get(model_key(r), set())
        and has_patching_data(r)
    ]


def sort_models(models: list[str]) -> list[str]:
    return sorted(
        models,
        key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 999,
    )


def compute_restoration_scores(result: dict, direction: str) -> list[dict]:
    """Compute per-layer normalized restoration score.

    direction: "p2v" / "v2p" for prior-circuit results, or
               "s2t" / "t2s" for visual-circuit results.

    Restoration score: 0 = no effect (same as unpatched target run),
                       1 = full restoration (matches the source run's logit diff).
    """
    if is_visual_circuit_result(result):
        return _compute_restoration_scores_visual_circuit(result, direction)
    return _compute_restoration_scores_prior_circuit(result, direction)


def _compute_restoration_scores_prior_circuit(
    result: dict, direction: str
) -> list[dict]:
    key = f"results_{direction}"
    results_list = result.get(key, [])
    if not results_list:
        return []

    prior_baseline = (
        result["prior_token_logit_prior_run"] - result["visual_token_logit_prior_run"]
    )
    visual_baseline = (
        result["prior_token_logit_visual_run"] - result["visual_token_logit_visual_run"]
    )

    if direction == "p2v":
        baseline = visual_baseline
        target = prior_baseline
    else:  # v2p
        baseline = prior_baseline
        target = visual_baseline

    denom = target - baseline
    if abs(denom) < 1e-8:
        return []

    scores = []
    for layer_result in results_list:
        patched_diff = layer_result["logit_prior"] - layer_result["logit_visual"]
        score = (patched_diff - baseline) / denom
        scores.append(
            {
                "layer_idx": layer_result["layer_idx_to_patch"],
                "score": score,
            }
        )
    return scores


def _compute_restoration_scores_visual_circuit(
    result: dict, direction: str
) -> list[dict]:
    if direction == "s2t":
        key = "results_source_to_target"
    elif direction == "t2s":
        key = "results_target_to_source"
    else:
        raise ValueError(
            f"direction must be 's2t' or 't2s' for visual_circuit, got {direction!r}"
        )
    results_list = result.get(key, [])
    if not results_list:
        return []

    source_baseline = (
        result["source_token_logit_source_run"]
        - result["target_token_logit_source_run"]
    )
    target_baseline = (
        result["source_token_logit_target_run"]
        - result["target_token_logit_target_run"]
    )

    if direction == "s2t":
        baseline = target_baseline
        target = source_baseline
    else:  # t2s
        baseline = source_baseline
        target = target_baseline

    denom = target - baseline
    if abs(denom) < 1e-8:
        return []

    scores = []
    for layer_result in results_list:
        patched_diff = layer_result["logit_source"] - layer_result["logit_target"]
        score = (patched_diff - baseline) / denom
        scores.append(
            {
                "layer_idx": layer_result["layer_idx_to_patch"],
                "score": score,
            }
        )
    return scores


def compute_mean_restoration_curves(
    rs: list[dict],
) -> tuple[list[float], list[float]]:
    """Return per-layer mean restoration scores for the two patching directions.

    For prior_circuit results: returns (P2V_mean, V2P_mean).
    For visual_circuit results: returns (S2T_mean, T2S_mean) i.e. (O→C, C→O).

    rs must be homogeneous (all the same contrast).
    """
    if not rs:
        return [], []
    if is_visual_circuit_result(rs[0]):
        return _compute_mean_restoration_curves(rs, "s2t", "t2s")
    return _compute_mean_restoration_curves(rs, "p2v", "v2p")


def _compute_mean_restoration_curves(
    rs: list[dict], dir_a: str, dir_b: str
) -> tuple[list[float], list[float]]:
    a_key = "results_source_to_target" if dir_a == "s2t" else "results_p2v"
    n_layers = len(rs[0][a_key])
    a_by_layer = defaultdict(list)
    b_by_layer = defaultdict(list)
    for r in rs:
        for s in compute_restoration_scores(r, dir_a):
            a_by_layer[s["layer_idx"]].append(s["score"])
        for s in compute_restoration_scores(r, dir_b):
            b_by_layer[s["layer_idx"]].append(s["score"])
    a = [
        np.mean(a_by_layer[layer_idx]) if a_by_layer[layer_idx] else 0.0
        for layer_idx in range(n_layers)
    ]
    b = [
        np.mean(b_by_layer[layer_idx]) if b_by_layer[layer_idx] else 0.0
        for layer_idx in range(n_layers)
    ]
    return a, b


def compute_critical_window(
    scores: list[float], threshold: float = 0.80
) -> tuple[int, int, float]:
    """Find the smallest consecutive layer range covering >threshold of total incremental restoration.

    Incremental contribution of layer L = scores[L] - scores[L-1] (with scores[-1] = 0).
    Returns (start_layer, end_layer, fraction_covered).
    """
    increments = [scores[0]] + [
        scores[i] - scores[i - 1] for i in range(1, len(scores))
    ]
    total = sum(increments)
    if total <= 0:
        return 0, len(scores) - 1, 0.0

    best_start, best_end = 0, len(scores) - 1
    best_len = len(scores)

    for start in range(len(increments)):
        cumsum = 0.0
        for end in range(start, len(increments)):
            cumsum += increments[end]
            if cumsum / total >= threshold:
                window_len = end - start + 1
                if window_len < best_len:
                    best_start, best_end, best_len = start, end, window_len
                break

    covered = sum(increments[best_start : best_end + 1]) / total
    return best_start, best_end, covered


def compute_flip_rates(
    rs: list[dict],
) -> tuple[list[float], list[float]]:
    """Return per-layer flip rates (0-1) for P2V and V2P."""
    n_layers = len(rs[0]["results_p2v"])
    n = len(rs)
    p2v_rates = []
    v2p_rates = []
    for layer_idx in range(n_layers):
        p2v_flips = sum(
            1
            for r in rs
            if r["results_p2v"][layer_idx]["logit_prior"]
            > r["results_p2v"][layer_idx]["logit_visual"]
        )
        v2p_flips = sum(
            1
            for r in rs
            if r["results_v2p"][layer_idx]["logit_visual"]
            > r["results_v2p"][layer_idx]["logit_prior"]
        )
        p2v_rates.append(p2v_flips / n)
        v2p_rates.append(v2p_flips / n)
    return p2v_rates, v2p_rates


# --- Printing helpers ---


def print_section(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_overview(all_results: list[dict], correct_ids: dict[str, set[str]]):
    """Print overview: how many examples pass the filter per model."""
    print_section("1. OVERVIEW (filtered to correctly-conflicting examples)")

    grouped = defaultdict(list)
    for r in all_results:
        grouped[model_key(r)].append(r)

    models = sort_models(list(grouped.keys()))

    print(f"\n  {'Model':<22} {'Total':>8} {'Correctly conflicting':>22}")
    print(f"  {'-' * 54}")

    for model in models:
        rs = grouped[model]
        n = len(rs)
        ids = correct_ids.get(model, set())
        n_correct = sum(1 for r in rs if r["example_id"] in ids)
        print(f"  {model:<22} {n:>8} {n_correct:>22}")


def print_restoration_scores(results: list[dict]):
    """Print mean normalized restoration score per layer."""
    print_section(
        "2. PER-LAYER MEAN RESTORATION SCORE (0=no effect, 1=full restoration)"
    )

    grouped = defaultdict(list)
    for r in results:
        grouped[model_key(r)].append(r)

    models = sort_models(list(grouped.keys()))

    for model in models:
        rs = grouped[model]
        n_layers = len(rs[0]["results_p2v"])
        p2v_scores, v2p_scores = compute_mean_restoration_curves(rs)

        print(f"\n  {model} (n={len(rs)}, {n_layers} layers)")
        print(
            f"  {'Layer':>6} {'P2V score':>12} {'V2P score':>12} {'Gap (V2P-P2V)':>14}"
        )
        print(f"  {'-' * 46}")

        for layer_idx in range(n_layers):
            gap = v2p_scores[layer_idx] - p2v_scores[layer_idx]
            print(
                f"  {layer_idx:>6} {p2v_scores[layer_idx]:>12.3f}"
                f" {v2p_scores[layer_idx]:>12.3f} {gap:>+14.3f}"
            )


def print_critical_window(results: list[dict]):
    """Print critical window for each model."""
    print_section(
        "3. CRITICAL WINDOW (smallest consecutive range with >80% of restoration)"
    )

    grouped = defaultdict(list)
    for r in results:
        grouped[model_key(r)].append(r)

    models = sort_models(list(grouped.keys()))

    print(
        f"\n  {'Model':<22} {'Layers':>6} {'P2V window':>14}"
        f" {'V2P window':>14} {'Combined window':>16} {'Depth range':>14}"
    )
    print(f"  {'-' * 88}")

    for model in models:
        rs = grouped[model]
        n_layers = len(rs[0]["results_p2v"])
        p2v_scores, v2p_scores = compute_mean_restoration_curves(rs)
        avg_scores = [(p + v) / 2 for p, v in zip(p2v_scores, v2p_scores, strict=False)]

        p2v_start, p2v_end, _ = compute_critical_window(p2v_scores)
        v2p_start, v2p_end, _ = compute_critical_window(v2p_scores)
        avg_start, avg_end, _ = compute_critical_window(avg_scores)

        depth_start = avg_start / (n_layers - 1) * 100
        depth_end = avg_end / (n_layers - 1) * 100

        print(
            f"  {model:<22} {n_layers:>6}"
            f" {p2v_start:>5}-{p2v_end:<5}"
            f" {v2p_start:>5}-{v2p_end:<5}"
            f" {avg_start:>6}-{avg_end:<6}"
            f" {depth_start:>5.0f}-{depth_end:.0f}%"
        )


def print_flip_rates(results: list[dict]):
    """Print per-layer flip rates."""
    print_section("4. PER-LAYER FLIP RATE")
    print("  Fraction of examples where patching at layer L flips the argmax")

    grouped = defaultdict(list)
    for r in results:
        grouped[model_key(r)].append(r)

    models = sort_models(list(grouped.keys()))

    for model in models:
        rs = grouped[model]
        n_layers = len(rs[0]["results_p2v"])
        p2v_rates, v2p_rates = compute_flip_rates(rs)

        print(f"\n  {model} (n={len(rs)})")
        print(f"  {'Layer':>6} {'P2V flips':>12} {'V2P flips':>12} {'Gap':>10}")
        print(f"  {'-' * 42}")

        for layer_idx in range(n_layers):
            p = p2v_rates[layer_idx] * 100
            v = v2p_rates[layer_idx] * 100
            print(f"  {layer_idx:>6} {p:>10.1f}% {v:>10.1f}% {v - p:>+9.1f}")


def load_and_filter(output_dir: Path, runner_label: str) -> list[dict]:
    """Load patching results, filter to correctly-conflicting examples, print overview."""
    print(f"Loading {runner_label} results...")
    all_results = load_patching_results(output_dir)
    print(f"Loaded {len(all_results)} result files.")

    print("Loading inference results for filtering...")
    correct_ids = load_correct_example_ids()
    for mk, ids in sorted(correct_ids.items()):
        print(f"  {mk}: {len(ids)} correctly-conflicting examples")

    print_overview(all_results, correct_ids)

    results = filter_results(all_results, correct_ids)
    print(f"\nFiltered to {len(results)} examples for analysis.")
    return results


def compute_head_restoration_scores(result: dict, direction: str) -> list[dict]:
    """Compute per-head normalized restoration score.

    Same formula as compute_restoration_scores but for attn head results
    which use layer_idx + head_idx instead of layer_idx_to_patch.
    """
    if is_visual_circuit_result(result):
        return _compute_head_restoration_scores_visual_circuit(result, direction)
    return _compute_head_restoration_scores_prior_circuit(result, direction)


def _compute_head_restoration_scores_prior_circuit(
    result: dict, direction: str
) -> list[dict]:
    key = f"results_{direction}"
    results_list = result.get(key, [])
    if not results_list:
        return []

    prior_baseline = (
        result["prior_token_logit_prior_run"] - result["visual_token_logit_prior_run"]
    )
    visual_baseline = (
        result["prior_token_logit_visual_run"] - result["visual_token_logit_visual_run"]
    )

    if direction == "p2v":
        baseline = visual_baseline
        target = prior_baseline
    else:  # v2p
        baseline = prior_baseline
        target = visual_baseline

    denom = target - baseline
    if abs(denom) < 1e-8:
        return []

    scores = []
    for entry in results_list:
        patched_diff = entry["logit_prior"] - entry["logit_visual"]
        score = (patched_diff - baseline) / denom
        scores.append(
            {
                "layer_idx": entry["layer_idx"],
                "head_idx": entry["head_idx"],
                "score": score,
            }
        )
    return scores


def _compute_head_restoration_scores_visual_circuit(
    result: dict, direction: str
) -> list[dict]:
    if direction == "s2t":
        key = "results_source_to_target"
    elif direction == "t2s":
        key = "results_target_to_source"
    else:
        raise ValueError(
            f"direction must be 's2t' or 't2s' for visual_circuit, got {direction!r}"
        )
    results_list = result.get(key, [])
    if not results_list:
        return []

    source_baseline = (
        result["source_token_logit_source_run"]
        - result["target_token_logit_source_run"]
    )
    target_baseline = (
        result["source_token_logit_target_run"]
        - result["target_token_logit_target_run"]
    )

    if direction == "s2t":
        baseline = target_baseline
        target = source_baseline
    else:  # t2s
        baseline = source_baseline
        target = target_baseline

    denom = target - baseline
    if abs(denom) < 1e-8:
        return []

    scores = []
    for entry in results_list:
        patched_diff = entry["logit_source"] - entry["logit_target"]
        score = (patched_diff - baseline) / denom
        scores.append(
            {
                "layer_idx": entry["layer_idx"],
                "head_idx": entry["head_idx"],
                "score": score,
            }
        )
    return scores


def compute_mean_head_restoration_grid(
    rs: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Return 2D arrays (n_layers, n_heads) of mean restoration scores.

    For prior_circuit:  returns (P2V_grid, V2P_grid).
    For visual_circuit: returns (S2T_grid, T2S_grid) i.e. (O→C, C→O).
    rs must be homogeneous (all the same contrast).
    """
    if not rs:
        return np.zeros((0, 0)), np.zeros((0, 0))
    if is_visual_circuit_result(rs[0]):
        return _compute_mean_head_restoration_grid(
            rs, "s2t", "t2s", "results_source_to_target"
        )
    return _compute_mean_head_restoration_grid(rs, "p2v", "v2p", "results_p2v")


def _compute_mean_head_restoration_grid(
    rs: list[dict], dir_a: str, dir_b: str, sample_key: str
) -> tuple[np.ndarray, np.ndarray]:
    first = rs[0][sample_key]
    n_layers = max(e["layer_idx"] for e in first) + 1
    n_heads = max(e["head_idx"] for e in first) + 1

    a_sums = np.zeros((n_layers, n_heads))
    b_sums = np.zeros((n_layers, n_heads))
    a_counts = np.zeros((n_layers, n_heads))
    b_counts = np.zeros((n_layers, n_heads))

    for r in rs:
        for s in compute_head_restoration_scores(r, dir_a):
            li, hi = s["layer_idx"], s["head_idx"]
            a_sums[li, hi] += s["score"]
            a_counts[li, hi] += 1
        for s in compute_head_restoration_scores(r, dir_b):
            li, hi = s["layer_idx"], s["head_idx"]
            b_sums[li, hi] += s["score"]
            b_counts[li, hi] += 1

    a_counts[a_counts == 0] = 1
    b_counts[b_counts == 0] = 1
    return a_sums / a_counts, b_sums / b_counts


def compute_head_flip_rates(
    rs: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Return 2D arrays (n_layers, n_heads) of flip rates (0-1).

    P2V flip: patching makes logit_prior > logit_visual (shifted toward prior).
    V2P flip: patching makes logit_visual > logit_prior (shifted toward visual).
    """
    first_p2v = rs[0]["results_p2v"]
    n_layers = max(e["layer_idx"] for e in first_p2v) + 1
    n_heads = max(e["head_idx"] for e in first_p2v) + 1
    n = len(rs)

    p2v_flips = np.zeros((n_layers, n_heads))
    v2p_flips = np.zeros((n_layers, n_heads))

    for r in rs:
        for entry in r["results_p2v"]:
            if entry["logit_prior"] > entry["logit_visual"]:
                p2v_flips[entry["layer_idx"], entry["head_idx"]] += 1
        for entry in r["results_v2p"]:
            if entry["logit_visual"] > entry["logit_prior"]:
                v2p_flips[entry["layer_idx"], entry["head_idx"]] += 1

    return p2v_flips / n, v2p_flips / n
