"""Classify MLP layers into promoting and suppressing via PCA on restoration scores.

For each model, each MLP layer is represented as a 2D point (P2V score, V2P score).
PCA reduces to 1D along the principal axis (expected to be roughly the diagonal).
Layers >2 sigma along PC1 are classified as promoting, <-2 sigma as suppressing.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.analysis._patching_common import (
    compute_mean_restoration_curves,
    filter_results,
    load_correct_example_ids,
    load_patching_results,
    model_key,
    sort_models,
)
from vdpo.settings import settings

PATCHING_BASE_DIR = settings.output_dir / "patching_last_token_mlp"


def _output_dir_for_contrast(contrast: str) -> Path:
    if contrast == "prior_circuit":
        return PATCHING_BASE_DIR
    return PATCHING_BASE_DIR / contrast


def _correct_ids_for_contrast(contrast: str) -> dict[str, set[str]]:
    if contrast == "prior_circuit":
        return load_correct_example_ids()
    from scripts.analysis._patching_common import load_intersection_example_ids

    return load_intersection_example_ids()


def _classification_output_path(contrast: str) -> Path:
    root = Path(__file__).parent.parent.parent / "data"
    if contrast == "prior_circuit":
        return root / "classify_mlp_layers.json"
    return root / f"classify_mlp_layers_{contrast}.json"


def classify_mlp_layers(
    results: list[dict],
    sigma_threshold: float = 2.0,
) -> dict[str, dict]:
    """Classify MLP layers per model using PCA on (P2V, V2P) restoration scores.

    Returns: {model_key: {
        "layers": [...],
        "p2v_scores": [...],
        "v2p_scores": [...],
        "pc1_scores": [...],
        "pc1_mean": float,
        "pc1_std": float,
        "pc1_direction": (float, float),
        "promoting": [layer_idx, ...],
        "suppressing": [layer_idx, ...],
    }}
    """
    grouped = defaultdict(list)
    for r in results:
        grouped[model_key(r)].append(r)

    classifications = {}
    for model in sort_models(list(grouped.keys())):
        rs = grouped[model]
        p2v_scores, v2p_scores = compute_mean_restoration_curves(rs)
        n_layers = len(p2v_scores)

        points = np.array(
            list(zip(p2v_scores, v2p_scores, strict=False))
        )  # (n_layers, 2)
        mean = points.mean(axis=0)
        centered = points - mean

        # PCA: eigenvectors of covariance matrix
        cov = np.cov(centered.T)
        _eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # eigh returns sorted ascending, take the last (largest)
        pc1 = eigenvectors[:, -1]

        # Ensure PC1 points toward positive (promoting) direction
        if pc1.sum() < 0:
            pc1 = -pc1

        pc1_scores = centered @ pc1
        pc1_mean = pc1_scores.mean()
        pc1_std = pc1_scores.std()

        promoting = [
            i
            for i in range(n_layers)
            if pc1_scores[i] > pc1_mean + sigma_threshold * pc1_std
        ]
        suppressing = [
            i
            for i in range(n_layers)
            if pc1_scores[i] < pc1_mean - sigma_threshold * pc1_std
        ]

        classifications[model] = {
            "n_examples": len(rs),
            "layers": list(range(n_layers)),
            "p2v_scores": p2v_scores,
            "v2p_scores": v2p_scores,
            "pc1_scores": pc1_scores.tolist(),
            "pc1_mean": float(pc1_mean),
            "pc1_std": float(pc1_std),
            "pc1_direction": (float(pc1[0]), float(pc1[1])),
            "promoting": promoting,
            "suppressing": suppressing,
        }

    return classifications


def print_classifications(classifications: dict[str, dict]):
    print(f"\n{'=' * 70}")
    print("  MLP LAYER CLASSIFICATION (PCA on P2V/V2P restoration scores)")
    print(f"{'=' * 70}")

    for model in sort_models(list(classifications.keys())):
        c = classifications[model]
        n = len(c["layers"])
        print(f"\n  {model} ({n} layers, n={c['n_examples']} examples)")
        print(
            f"    PC1 direction: ({c['pc1_direction'][0]:.3f}, {c['pc1_direction'][1]:.3f})"
        )
        print(f"    PC1 mean={c['pc1_mean']:.4f}, std={c['pc1_std']:.4f}")
        print(f"    Threshold: ±2σ = ±{2 * c['pc1_std']:.4f}")  # noqa: RUF001

        if c["promoting"]:
            print(f"    Promoting ({len(c['promoting'])}): {c['promoting']}")
            for layer_idx in c["promoting"]:
                print(
                    f"      Layer {layer_idx}: P2V={c['p2v_scores'][layer_idx]:.3f},"
                    f" V2P={c['v2p_scores'][layer_idx]:.3f},"
                    f" PC1={c['pc1_scores'][layer_idx]:.4f}"
                )
        else:
            print("    Promoting: none")

        if c["suppressing"]:
            print(f"    Suppressing ({len(c['suppressing'])}): {c['suppressing']}")
            for layer_idx in c["suppressing"]:
                print(
                    f"      Layer {layer_idx}: P2V={c['p2v_scores'][layer_idx]:.3f},"
                    f" V2P={c['v2p_scores'][layer_idx]:.3f},"
                    f" PC1={c['pc1_scores'][layer_idx]:.4f}"
                )
        else:
            print("    Suppressing: none")


def save_classifications(classifications: dict[str, dict], output_path: Path):
    """Save classifications to JSON for downstream use (e.g., knockout experiments)."""
    serializable = {}
    for model, c in classifications.items():
        serializable[model] = {
            "promoting": c["promoting"],
            "suppressing": c["suppressing"],
            "pc1_direction": c["pc1_direction"],
            "pc1_std": c["pc1_std"],
            "layers": [
                {
                    "layer_idx": layer_idx,
                    "p2v_score": c["p2v_scores"][layer_idx],
                    "v2p_score": c["v2p_scores"][layer_idx],
                    "pc1_score": c["pc1_scores"][layer_idx],
                }
                for layer_idx in c["layers"]
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
    parser.add_argument("--sigma", type=float, default=2.0)
    args = parser.parse_args()

    output_dir = _output_dir_for_contrast(args.contrast)
    print(f"Loading MLP patching results from {output_dir} ...")
    all_results = load_patching_results(output_dir)
    correct_ids = _correct_ids_for_contrast(args.contrast)
    results = filter_results(all_results, correct_ids)
    print(f"Filtered to {len(results)} examples (contrast={args.contrast}).")

    classifications = classify_mlp_layers(results, sigma_threshold=args.sigma)
    print_classifications(classifications)
    save_classifications(
        classifications,
        _classification_output_path(args.contrast),
    )


if __name__ == "__main__":
    main()
