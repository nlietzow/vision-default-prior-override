"""Figure 2: Attention head classification scatter for 3 representative models.

PCA scatter with P2V vs V2P mean restoration scores per head. Heads beyond
+/-2 sigma on PC1 are classified as promoting or suppressing.
Models: Qwen 7B, LLaVA-NeXT 7B, PaliGemma 3B (matches Fig. 1 coverage).

Output: figures/head_classification.pdf
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

from scripts.analysis._patching_common import (
    filter_results,
    load_correct_example_ids,
    load_patching_results,
)
from scripts.analysis.classify_attn_heads import PATCHING_BASE_DIR, classify_attn_heads
from scripts.paper_figures._common import (
    COLUMN_WIDTH,
    PROMOTING_COLOR,
    SUPPRESSING_COLOR,
    save_fig,
    setup_style,
)

REPRESENTATIVE_MODELS = ["qwen/3B", "llava_next/7B", "paligemma_2/3B"]


def main():
    setup_style()

    print("Loading results...")
    all_results = load_patching_results(PATCHING_BASE_DIR)
    correct_ids = load_correct_example_ids()
    results = filter_results(all_results, correct_ids)
    print(f"Filtered to {len(results)} correctly-conflicting examples.")

    classifications = classify_attn_heads(results)

    plt.rcParams["figure.constrained_layout.use"] = False
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 0.50),
        sharey=True,
    )
    fig.subplots_adjust(wspace=0.08, left=0.13, right=0.98, top=0.82, bottom=0.22)

    panel_titles = {
        "qwen/3B": "(a) Qwen-VL 3B",
        "llava_next/7B": "(b) LLaVA-NeXT 7B",
        "paligemma_2/3B": "(c) PaliGemma 3B",
    }

    for ax, model in zip(axes, REPRESENTATIVE_MODELS, strict=False):
        c = classifications[model]
        p2v_flat = c["p2v_grid"].ravel()
        v2p_flat = c["v2p_grid"].ravel()
        n_heads_per_layer = c["n_heads"]

        promoting_set = set(c["promoting"])
        suppressing_set = set(c["suppressing"])

        # Neutral points
        neutral_idx = [
            i
            for i in range(len(p2v_flat))
            if (i // n_heads_per_layer, i % n_heads_per_layer) not in promoting_set
            and (i // n_heads_per_layer, i % n_heads_per_layer) not in suppressing_set
        ]
        ax.scatter(
            p2v_flat[neutral_idx],
            v2p_flat[neutral_idx],
            s=1,
            color="#BBBBBB",
            alpha=0.5,
            zorder=2,
            linewidths=0,
        )

        # Promoting
        if promoting_set:
            idx = [layer_idx * n_heads_per_layer + h for layer_idx, h in promoting_set]
            ax.scatter(
                p2v_flat[idx],
                v2p_flat[idx],
                s=5,
                color=PROMOTING_COLOR,
                zorder=3,
                label="Promoting",
                linewidths=0.2,
                edgecolors="white",
            )

        # Suppressing
        if suppressing_set:
            idx = [
                layer_idx * n_heads_per_layer + h for layer_idx, h in suppressing_set
            ]
            ax.scatter(
                p2v_flat[idx],
                v2p_flat[idx],
                s=5,
                color=SUPPRESSING_COLOR,
                zorder=3,
                label="Suppressing",
                linewidths=0.2,
                edgecolors="white",
            )

        # Reference lines
        all_vals = np.concatenate([p2v_flat, v2p_flat])
        lo, hi = all_vals.min() - 0.01, all_vals.max() + 0.01
        dx, dy = c["pc1_direction"]
        mean_p2v = float(p2v_flat.mean())
        mean_v2p = float(v2p_flat.mean())
        t = hi - lo
        ax.plot(
            [mean_p2v - t * dx, mean_p2v + t * dx],
            [mean_v2p - t * dy, mean_v2p + t * dy],
            color="gray",
            linestyle="--",
            alpha=0.4,
            linewidth=0.5,
        )
        ax.axhline(y=0, color="black", linewidth=0.3, alpha=0.4)
        ax.axvline(x=0, color="black", linewidth=0.3, alpha=0.4)

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("P2V restoration", fontsize=7)
        n_total = c["n_layers"] * c["n_heads"]
        n_class = len(promoting_set) + len(suppressing_set)
        ax.set_title(
            f"{panel_titles[model]}\n({n_class}/{n_total})",
            fontsize=7,
            pad=3,
        )
        ax.tick_params(labelsize=6)
        ax.tick_params(which="minor", length=2)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    # Draw horizontal grid lines after the shared y-axis has settled, so all
    # panels use the same tick positions.
    for ax in axes:
        for val in ax.get_yticks():
            if val != 0:
                ax.axhline(y=val, color="gray", linewidth=0.35, alpha=0.4, zorder=1)

    axes[0].set_ylabel("V2P restoration", fontsize=7)
    leg = axes[-1].legend(
        fontsize=5,
        loc="lower right",
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#888888",
        borderpad=0.3,
        handletextpad=0.3,
    )
    leg.get_frame().set_linewidth(0.3)

    save_fig(fig, "head_classification.pdf")
    print("Done.")


if __name__ == "__main__":
    main()
