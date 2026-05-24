"""Classify attention heads into promoting and suppressing via PCA on restoration scores.

For each model, each head is represented as a 2D point (P2V score, V2P score).
PCA reduces to 1D along the principal axis.
Heads >2 sigma along PC1 are classified as promoting, <-2 sigma as suppressing.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.analysis._patching_common import (
    compute_mean_head_restoration_grid,
    filter_results,
    load_correct_example_ids,
    load_patching_results,
    model_key,
    sort_models,
)
from vdpo.settings import settings

PATCHING_BASE_DIR = settings.output_dir / "patching_last_token_attn_heads"


def _output_dir_for_contrast(contrast: str) -> Path:
    if contrast == "prior_circuit":
        return PATCHING_BASE_DIR
    return PATCHING_BASE_DIR / contrast


def _correct_ids_for_contrast(contrast: str) -> dict[str, set[str]]:
    if contrast == "prior_circuit":
        return load_correct_example_ids()
    # visual_circuit uses the intersection subset so comparisons are clean
    from scripts.analysis._patching_common import load_intersection_example_ids

    return load_intersection_example_ids()


def _classification_output_path(contrast: str) -> Path:
    root = Path(__file__).parent.parent.parent / "data"
    if contrast == "prior_circuit":
        return root / "classify_attn_heads.json"
    return root / f"classify_attn_heads_{contrast}.json"


def classify_attn_heads(
    results: list[dict],
    sigma_threshold: float = 2.0,
) -> dict[str, dict]:
    """Classify attention heads per model using PCA on (P2V, V2P) restoration scores.

    Returns: {model_key: {
        "n_examples": int,
        "n_layers": int,
        "n_heads": int,
        "p2v_grid": ndarray (n_layers, n_heads),
        "v2p_grid": ndarray (n_layers, n_heads),
        "pc1_scores": ndarray (n_layers * n_heads,),
        "pc1_mean": float,
        "pc1_std": float,
        "pc1_direction": (float, float),
        "promoting": [(layer, head), ...],
        "suppressing": [(layer, head), ...],
    }}
    """
    grouped = defaultdict(list)
    for r in results:
        grouped[model_key(r)].append(r)

    classifications = {}
    for model in sort_models(list(grouped.keys())):
        rs = grouped[model]
        p2v_grid, v2p_grid = compute_mean_head_restoration_grid(rs)
        n_layers, n_heads = p2v_grid.shape

        # Flatten to (n_layers * n_heads, 2) for PCA
        p2v_flat = p2v_grid.ravel()
        v2p_flat = v2p_grid.ravel()
        points = np.column_stack([p2v_flat, v2p_flat])

        mean = points.mean(axis=0)
        centered = points - mean

        cov = np.cov(centered.T)
        _eigenvalues, eigenvectors = np.linalg.eigh(cov)
        pc1 = eigenvectors[:, -1]

        if pc1.sum() < 0:
            pc1 = -pc1

        pc1_scores = centered @ pc1
        pc1_mean = pc1_scores.mean()
        pc1_std = pc1_scores.std()

        promoting = []
        suppressing = []
        for i in range(len(pc1_scores)):
            if pc1_scores[i] > pc1_mean + sigma_threshold * pc1_std:
                layer = i // n_heads
                head = i % n_heads
                promoting.append((layer, head))
            elif pc1_scores[i] < pc1_mean - sigma_threshold * pc1_std:
                layer = i // n_heads
                head = i % n_heads
                suppressing.append((layer, head))

        classifications[model] = {
            "n_examples": len(rs),
            "n_layers": n_layers,
            "n_heads": n_heads,
            "p2v_grid": p2v_grid,
            "v2p_grid": v2p_grid,
            "pc1_scores": pc1_scores,
            "pc1_mean": float(pc1_mean),
            "pc1_std": float(pc1_std),
            "pc1_direction": (float(pc1[0]), float(pc1[1])),
            "promoting": promoting,
            "suppressing": suppressing,
        }

    return classifications


def print_classifications(classifications: dict[str, dict]):
    print(f"\n{'=' * 70}")
    print("  ATTENTION HEAD CLASSIFICATION (PCA on P2V/V2P restoration scores)")
    print(f"{'=' * 70}")

    for model in sort_models(list(classifications.keys())):
        c = classifications[model]
        n = c["n_layers"] * c["n_heads"]
        print(
            f"\n  {model} ({c['n_layers']} layers x {c['n_heads']} heads"
            f" = {n} heads, n={c['n_examples']} examples)"
        )
        print(
            f"    PC1 direction: ({c['pc1_direction'][0]:.3f}, {c['pc1_direction'][1]:.3f})"
        )
        print(f"    PC1 mean={c['pc1_mean']:.4f}, std={c['pc1_std']:.4f}")
        print(f"    Threshold: ±2σ = ±{2 * c['pc1_std']:.4f}")  # noqa: RUF001

        if c["promoting"]:
            print(f"    Promoting ({len(c['promoting'])}):")
            for layer, head in c["promoting"]:
                p2v = c["p2v_grid"][layer, head]
                v2p = c["v2p_grid"][layer, head]
                idx = layer * c["n_heads"] + head
                print(
                    f"      L{layer}H{head}: P2V={p2v:.3f},"
                    f" V2P={v2p:.3f},"
                    f" PC1={c['pc1_scores'][idx]:.4f}"
                )
        else:
            print("    Promoting: none")

        if c["suppressing"]:
            print(f"    Suppressing ({len(c['suppressing'])}):")
            for layer, head in c["suppressing"]:
                p2v = c["p2v_grid"][layer, head]
                v2p = c["v2p_grid"][layer, head]
                idx = layer * c["n_heads"] + head
                print(
                    f"      L{layer}H{head}: P2V={p2v:.3f},"
                    f" V2P={v2p:.3f},"
                    f" PC1={c['pc1_scores'][idx]:.4f}"
                )
        else:
            print("    Suppressing: none")


def save_classifications(classifications: dict[str, dict], output_path: Path):
    """Save classifications to JSON for downstream use (e.g., knockout experiments)."""
    serializable = {}
    for model, c in classifications.items():
        serializable[model] = {
            "promoting": [
                {"layer": layer_idx, "head": h} for layer_idx, h in c["promoting"]
            ],
            "suppressing": [
                {"layer": layer_idx, "head": h} for layer_idx, h in c["suppressing"]
            ],
            "pc1_direction": c["pc1_direction"],
            "pc1_std": c["pc1_std"],
            "n_layers": c["n_layers"],
            "n_heads": c["n_heads"],
            "heads": [
                {
                    "layer_idx": i // c["n_heads"],
                    "head_idx": i % c["n_heads"],
                    "p2v_score": float(c["p2v_grid"].ravel()[i]),
                    "v2p_score": float(c["v2p_grid"].ravel()[i]),
                    "pc1_score": float(c["pc1_scores"][i]),
                }
                for i in range(len(c["pc1_scores"]))
            ],
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Saved classifications to {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contrast",
        type=str,
        default="prior_circuit",
        choices=["prior_circuit", "visual_circuit"],
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=2.0,
        help="Threshold for promoting/suppressing classification.",
    )
    args = parser.parse_args()

    output_dir = _output_dir_for_contrast(args.contrast)
    print(f"Loading attention head patching results from {output_dir} ...")
    all_results = load_patching_results(output_dir)
    correct_ids = _correct_ids_for_contrast(args.contrast)
    results = filter_results(all_results, correct_ids)
    print(f"Filtered to {len(results)} examples (contrast={args.contrast}).")

    classifications = classify_attn_heads(results, sigma_threshold=args.sigma)
    print_classifications(classifications)
    save_classifications(
        classifications,
        _classification_output_path(args.contrast),
    )


if __name__ == "__main__":
    main()
