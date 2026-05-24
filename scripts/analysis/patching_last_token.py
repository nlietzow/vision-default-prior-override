"""Analyze activation patching results across all three granularities."""

from collections import defaultdict

from scripts.analysis._patching_common import (
    compute_head_flip_rates,
    compute_mean_head_restoration_grid,
    load_and_filter,
    model_key,
    print_critical_window,
    print_flip_rates,
    print_restoration_scores,
    print_section,
    sort_models,
)
from scripts.analysis.classify_attn_heads import classify_attn_heads
from scripts.analysis.classify_attn_heads import (
    print_classifications as print_head_classifications,
)
from scripts.analysis.classify_mlp_layers import classify_mlp_layers
from scripts.analysis.classify_mlp_layers import (
    print_classifications as print_mlp_classifications,
)
from vdpo.settings import settings

RES_STREAM_DIR = settings.output_dir / "patching_last_token_res_stream"
ATTN_HEADS_DIR = settings.output_dir / "patching_last_token_attn_heads"
MLP_DIR = settings.output_dir / "patching_last_token_mlp"


def print_top_heads_by_restoration(results: list[dict], top_n: int = 15):
    """Print the top-N heads by restoration score per model."""
    print_section("TOP HEADS BY RESTORATION SCORE")

    grouped = defaultdict(list)
    for r in results:
        grouped[model_key(r)].append(r)

    for model in sort_models(list(grouped.keys())):
        rs = grouped[model]
        p2v_grid, v2p_grid = compute_mean_head_restoration_grid(rs)
        n_layers, n_heads = p2v_grid.shape

        heads = []
        for li in range(n_layers):
            for hi in range(n_heads):
                heads.append(
                    {
                        "layer": li,
                        "head": hi,
                        "p2v": p2v_grid[li, hi],
                        "v2p": v2p_grid[li, hi],
                        "mean": (p2v_grid[li, hi] + v2p_grid[li, hi]) / 2,
                    }
                )

        heads.sort(key=lambda h: abs(h["mean"]), reverse=True)

        print(f"\n  {model} (n={len(rs)}, {n_layers}L x {n_heads}H)")
        print(f"  {'Head':>8} {'P2V':>8} {'V2P':>8} {'Mean':>8}")
        print(f"  {'-' * 34}")
        for h in heads[:top_n]:
            print(
                f"  L{h['layer']:>2}H{h['head']:<2}"
                f"  {h['p2v']:>+8.3f} {h['v2p']:>+8.3f} {h['mean']:>+8.3f}"
            )


def print_top_heads_by_flip_rate(results: list[dict], top_n: int = 15):
    """Print the top-N heads by flip rate per model."""
    print_section("TOP HEADS BY FLIP RATE")

    grouped = defaultdict(list)
    for r in results:
        grouped[model_key(r)].append(r)

    for model in sort_models(list(grouped.keys())):
        rs = grouped[model]
        p2v_rates, v2p_rates = compute_head_flip_rates(rs)
        n_layers, n_heads = p2v_rates.shape

        heads = []
        for li in range(n_layers):
            for hi in range(n_heads):
                heads.append(
                    {
                        "layer": li,
                        "head": hi,
                        "p2v": p2v_rates[li, hi],
                        "v2p": v2p_rates[li, hi],
                        "max": max(p2v_rates[li, hi], v2p_rates[li, hi]),
                    }
                )

        heads.sort(key=lambda h: h["max"], reverse=True)

        print(f"\n  {model} (n={len(rs)})")
        print(f"  {'Head':>8} {'P2V':>8} {'V2P':>8}")
        print(f"  {'-' * 26}")
        for h in heads[:top_n]:
            print(
                f"  L{h['layer']:>2}H{h['head']:<2}  {h['p2v']:>7.1%} {h['v2p']:>7.1%}"
            )


def main():
    # 1. Residual stream
    print("\n" + "=" * 70)
    print("  RESIDUAL STREAM PATCHING")
    print("=" * 70)
    res_results = load_and_filter(RES_STREAM_DIR, "residual stream patching")
    print_restoration_scores(res_results)
    print_critical_window(res_results)
    print_flip_rates(res_results)

    # 2. Attention heads
    print("\n" + "=" * 70)
    print("  ATTENTION HEAD PATCHING")
    print("=" * 70)
    attn_results = load_and_filter(ATTN_HEADS_DIR, "attention head patching")
    print_top_heads_by_restoration(attn_results)
    print_top_heads_by_flip_rate(attn_results)
    print_section("HEAD CLASSIFICATION")
    head_classifications = classify_attn_heads(attn_results)
    print_head_classifications(head_classifications)

    # 3. MLP
    print("\n" + "=" * 70)
    print("  MLP PATCHING")
    print("=" * 70)
    mlp_results = load_and_filter(MLP_DIR, "MLP patching")
    print_restoration_scores(mlp_results)
    print_flip_rates(mlp_results)
    print_section("MLP CLASSIFICATION")
    mlp_classifications = classify_mlp_layers(mlp_results)
    print_mlp_classifications(mlp_classifications)


if __name__ == "__main__":
    main()
