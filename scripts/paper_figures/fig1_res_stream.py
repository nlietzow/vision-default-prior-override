"""Figure 1: Residual stream restoration scores for 3 representative models.

Three-panel line plot showing P2V (dashed) and V2P (solid) restoration curves
with critical window shading. Models: Qwen 7B, LLaVA-NeXT 7B, PaliGemma 3B.

Output: figures/res_stream_restoration.pdf
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from scripts.analysis._patching_common import (
    compute_critical_window,
    compute_mean_restoration_curves,
    filter_results,
    load_correct_example_ids,
    load_patching_results,
)
from scripts.paper_figures._common import (
    COLUMN_WIDTH,
    CRITICAL_WINDOW_COLOR,
    P2V_COLOR,
    V2P_COLOR,
    save_fig,
    setup_style,
)
from vdpo.settings import settings

OUTPUT_DIR = settings.output_dir / "patching_last_token_res_stream"

# Three representative models for main paper
REPRESENTATIVE_MODELS = ["qwen/3B", "llava_next/7B", "paligemma_2/3B"]


def main():
    setup_style()

    print("Loading results...")
    all_results = load_patching_results(OUTPUT_DIR)
    correct_ids = load_correct_example_ids()
    results = filter_results(all_results, correct_ids)
    print(f"Filtered to {len(results)} correctly-conflicting examples.")

    # Group by model
    from collections import defaultdict

    from scripts.analysis._patching_common import model_key

    grouped = defaultdict(list)
    for r in results:
        grouped[model_key(r)].append(r)

    plt.rcParams["figure.constrained_layout.use"] = False
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 0.45),
        sharey=True,
    )
    fig.subplots_adjust(wspace=0.08, left=0.13, right=0.98, top=0.88, bottom=0.22)

    panel_titles = {
        "qwen/3B": "(a) Qwen-VL 3B",
        "llava_next/7B": "(b) LLaVA-NeXT 7B",
        "paligemma_2/3B": "(c) PaliGemma 3B",
    }

    for ax, model in zip(axes, REPRESENTATIVE_MODELS, strict=False):
        rs = grouped[model]
        n_layers = len(rs[0]["results_p2v"])
        p2v_vals, v2p_vals = compute_mean_restoration_curves(rs)

        # Critical window boundaries as dashed verticals
        avg = [(p + v) / 2 for p, v in zip(p2v_vals, v2p_vals, strict=False)]
        cw_start, cw_end, _ = compute_critical_window(avg)
        for cw_x in (cw_start, cw_end):
            ax.axvline(
                cw_x,
                color=CRITICAL_WINDOW_COLOR,
                linestyle="--",
                linewidth=0.6,
                alpha=0.7,
                zorder=0.5,
            )

        layers = list(range(n_layers))
        ax.fill_between(
            layers,
            p2v_vals,
            v2p_vals,
            where=[v > p for v, p in zip(v2p_vals, p2v_vals, strict=False)],
            alpha=0.35,
            color="#DDCC77",
            linewidth=0,
            zorder=0.7,
        )
        ax.plot(layers, v2p_vals, color=V2P_COLOR, linewidth=1.0, label="V2P")
        ax.plot(
            layers,
            p2v_vals,
            color=P2V_COLOR,
            linewidth=1.0,
            linestyle="--",
            label="P2V",
        )
        ax.set_xlabel("Layer", fontsize=7)
        ax.set_xlim(0, n_layers - 1)
        ax.set_ylim(0, 1.05)
        ax.set_title(panel_titles[model], fontsize=7, pad=3)
        ax.tick_params(labelsize=6)
        ax.tick_params(which="minor", length=2)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        # Draw grid lines manually on top of shading
        for y in [0.25, 0.5, 0.75, 1.0]:
            ax.axhline(y=y, color="gray", linewidth=0.35, alpha=0.6, zorder=2)
        for y in [0.125, 0.375, 0.625, 0.875]:
            ax.axhline(y=y, color="gray", linewidth=0.2, alpha=0.35, zorder=2)

    axes[0].set_ylabel("Restoration score", fontsize=7)
    axes[0].legend(fontsize=5, loc="upper left", frameon=False)

    save_fig(fig, "res_stream_restoration.pdf")
    print("Done.")


if __name__ == "__main__":
    main()
