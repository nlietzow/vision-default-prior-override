"""Appendix figure: Difference logit lens hit rates for all classified heads.

Horizontal bar chart showing top-20 hit rates for all classified heads
across all 5 models. Late-layer heads show high hit rates, earlier heads
show 0%.

Output: figures/logit_lens_all.pdf
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator

from scripts.analysis._patching_common import sort_models
from scripts.paper_figures._common import (
    COLORS,
    MODEL_LABELS_SHORT,
    TEXT_WIDTH,
    save_fig,
    setup_style,
)

RESULTS_PATH = Path("data") / "difference_logit_lens.json"


def main():
    setup_style()

    print("Loading results...")
    with RESULTS_PATH.open() as f:
        results = json.load(f)

    models = sort_models(list(results.keys()))
    n = len(models)

    plt.rcParams["figure.constrained_layout.use"] = False
    fig, axes = plt.subplots(
        1,
        n,
        figsize=(TEXT_WIDTH, TEXT_WIDTH * 0.55),
    )
    fig.subplots_adjust(wspace=0.4, left=0.05, right=0.90, top=0.92, bottom=0.06)

    bar_height = 0.35

    for ax, model in zip(axes, models, strict=False):
        r = results[model]
        n_ex = r["n_examples"]

        labels = []
        top_rates = []
        bottom_rates = []
        colors = []

        for h in r["promoting"]:
            labels.append(f"L{h['layer']}H{h['head']}")
            top_rates.append(h["visual_in_top_k"] / n_ex * 100)
            bottom_rates.append(h["prior_in_bottom_k"] / n_ex * 100)
            colors.append(COLORS["rose"])

        for h in r["suppressing"]:
            labels.append(f"L{h['layer']}H{h['head']}")
            top_rates.append(
                h["prior_in_top_k"] / n_ex * 100 if "prior_in_top_k" in h else 0
            )
            bottom_rates.append(
                h["visual_in_bottom_k"] / n_ex * 100 if "visual_in_bottom_k" in h else 0
            )
            colors.append(COLORS["indigo"])

        y_pos = np.arange(len(labels))

        # Top-k bars (solid)
        ax.barh(
            y_pos - bar_height / 2,
            top_rates,
            color=colors,
            height=bar_height,
        )
        # Bottom-k bars (hatched)
        ax.barh(
            y_pos + bar_height / 2,
            bottom_rates,
            color=colors,
            height=bar_height,
            alpha=0.4,
            hatch="//",
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=4)
        ax.set_xlabel("Hit rate (%)", fontsize=6)
        ax.set_xlim(0, 105)
        ax.axvline(x=50, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)
        ax.set_title(
            MODEL_LABELS_SHORT[model],
            fontsize=7,
            pad=3,
        )
        ax.tick_params(labelsize=5)
        ax.tick_params(which="minor", length=2)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.invert_yaxis()

    # Legend on last axis
    legend_elements = [
        Patch(facecolor=COLORS["rose"], label="Promoting"),
        Patch(facecolor=COLORS["indigo"], label="Suppressing"),
        Patch(facecolor="#999999", label="Top-20"),
        Patch(facecolor="#999999", alpha=0.4, hatch="//", label="Bottom-20"),
    ]
    axes[-1].legend(
        handles=legend_elements,
        fontsize=5,
        frameon=False,
        bbox_to_anchor=(1.0, 0.5),
        loc="center left",
    )

    save_fig(fig, "logit_lens_all.pdf")
    print("Done.")


if __name__ == "__main__":
    main()
