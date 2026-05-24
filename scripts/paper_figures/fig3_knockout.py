"""Figure 3: Knockout flip rates for promoting-head group knockout.

Bar chart showing prior vs visual grounding flip rates across all 5 models
when all promoting attention heads are knocked out simultaneously.

Output: figures/knockout_flip_rates.pdf
"""

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

from scripts.analysis._patching_common import load_correct_example_ids, sort_models
from scripts.analysis.knockout_attn_heads import (
    KNOCKOUT_ATTN_HEADS_DIR,
    compute_flip_rates,
    filter_results,
    load_knockout_results,
    model_key,
)
from scripts.paper_figures._common import (
    COLUMN_WIDTH,
    PRIOR_COLOR,
    VISUAL_COLOR,
    save_fig,
    setup_style,
)

MODEL_LABELS = {
    "qwen/3B": "Qwen\n3B",
    "qwen/7B": "Qwen\n7B",
    "llava_next/7B": "LLaVA\n7B",
    "paligemma_2/3B": "PG\n3B",
    "paligemma_2/10B": "PG\n10B",
}


def main():
    setup_style()

    print("Loading knockout results...")
    results = load_knockout_results(KNOCKOUT_ATTN_HEADS_DIR, "prior_circuit")
    correct_ids = load_correct_example_ids()
    results = filter_results(results, correct_ids)
    print(f"Filtered to {len(results)} results.")

    by_model = defaultdict(list)
    for r in results:
        by_model[model_key(r)].append(r)

    models = sort_models(list(by_model.keys()))

    prior_rates = []
    visual_rates = []
    for model in models:
        rates = compute_flip_rates(by_model[model])
        gp = rates["grp_promoting"]
        prior_rates.append(gp["prior_flip_rate"] * 100)
        visual_rates.append(gp["visual_flip_rate"] * 100)

    plt.rcParams["figure.constrained_layout.use"] = False
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 0.45))
    fig.subplots_adjust(left=0.14, right=0.72, top=0.90, bottom=0.22)

    x = np.arange(len(models))
    width = 0.35

    ax.bar(
        x - width / 2,
        prior_rates,
        width,
        color=PRIOR_COLOR,
        label="Prior flip rate",
    )
    ax.bar(
        x + width / 2,
        visual_rates,
        width,
        color=VISUAL_COLOR,
        label="Visual flip rate",
    )

    # Value labels on prior bars
    for i, v in enumerate(prior_rates):
        ax.text(
            i - width / 2,
            v + 2,
            f"{v:.0f}",
            ha="center",
            va="bottom",
            fontsize=5,
            color="#555555",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [MODEL_LABELS[m] for m in models],
        fontsize=6,
    )
    ax.set_ylabel("Flip rate (%)", fontsize=7)
    ax.set_ylim(0, 105)
    ax.tick_params(labelsize=6)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="minor", length=2)
    ax.legend(fontsize=5, frameon=False, bbox_to_anchor=(1.0, 0.5), loc="center left")

    # Grid at major ticks
    for y in [20, 40, 60, 80, 100]:
        ax.axhline(y=y, color="gray", linewidth=0.35, alpha=0.5, zorder=0)

    save_fig(fig, "knockout_flip_rates.pdf")
    print("Done.")


if __name__ == "__main__":
    main()
