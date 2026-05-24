"""Appendix figure: MLP restoration scores across layers for all 5 models.

Five-panel line plot showing P2V (dashed) and V2P (solid) MLP patching
restoration curves.

Output: figures/mlp_restoration.pdf
"""

from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from scripts.analysis._patching_common import (
    compute_mean_restoration_curves,
    filter_results,
    load_correct_example_ids,
    load_patching_results,
    model_key,
    sort_models,
)
from scripts.paper_figures._common import (
    MODEL_LABELS_SHORT,
    P2V_COLOR,
    TEXT_WIDTH,
    V2P_COLOR,
    save_fig,
    setup_style,
)
from vdpo.settings import settings

OUTPUT_DIR = settings.output_dir / "patching_last_token_mlp"


def main():
    setup_style()

    print("Loading results...")
    all_results = load_patching_results(OUTPUT_DIR)
    correct_ids = load_correct_example_ids()
    results = filter_results(all_results, correct_ids)
    print(f"Filtered to {len(results)} correctly-conflicting examples.")

    grouped = defaultdict(list)
    for r in results:
        grouped[model_key(r)].append(r)

    models = sort_models(list(grouped.keys()))

    # Compute global y-range across all models
    all_vals = []
    for model in models:
        rs = grouped[model]
        p2v, v2p = compute_mean_restoration_curves(rs)
        all_vals.extend(p2v)
        all_vals.extend(v2p)
    y_lo = min(all_vals) - 0.02
    y_hi = max(all_vals) + 0.02

    plt.rcParams["figure.constrained_layout.use"] = False
    n = len(models)
    fig, axes = plt.subplots(
        1,
        n,
        figsize=(TEXT_WIDTH, TEXT_WIDTH * 0.22),
        sharey=True,
    )
    fig.subplots_adjust(wspace=0.08, left=0.06, right=0.93, top=0.85, bottom=0.22)

    for ax, model in zip(axes, models, strict=False):
        rs = grouped[model]
        n_layers = len(rs[0]["results_p2v"])
        p2v_vals, v2p_vals = compute_mean_restoration_curves(rs)

        layers = list(range(n_layers))
        ax.plot(layers, v2p_vals, color=V2P_COLOR, linewidth=1.0, label="V2P")
        ax.plot(
            layers,
            p2v_vals,
            color=P2V_COLOR,
            linewidth=1.0,
            linestyle="--",
            label="P2V",
        )
        ax.axhline(y=0, color="black", linewidth=0.3, alpha=0.5)
        ax.set_xlabel("Layer", fontsize=6)
        ax.set_xlim(0, n_layers - 1)
        ax.set_ylim(y_lo, y_hi)
        ax.set_title(MODEL_LABELS_SHORT[model], fontsize=6, pad=3)
        ax.tick_params(labelsize=5)
        ax.tick_params(which="minor", length=2)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        # Grid
        for y in ax.get_yticks():
            if y != 0:
                ax.axhline(y=y, color="gray", linewidth=0.3, alpha=0.5, zorder=0)

    axes[0].set_ylabel("Restoration score", fontsize=6)
    axes[-1].legend(
        fontsize=5, frameon=False, bbox_to_anchor=(1.0, 0.5), loc="center left"
    )

    save_fig(fig, "mlp_restoration.pdf")
    print("Done.")


if __name__ == "__main__":
    main()
