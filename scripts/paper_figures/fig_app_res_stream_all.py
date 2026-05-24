"""Appendix figure: Residual stream restoration scores for all 5 models.

Five-panel line plot showing P2V (dashed) and V2P (solid) restoration curves
with critical window shading.

Output: figures/res_stream_all_models.pdf
"""

from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from scripts.analysis._patching_common import (
    compute_critical_window,
    compute_mean_restoration_curves,
    filter_results,
    load_correct_example_ids,
    load_patching_results,
    model_key,
    sort_models,
)
from scripts.paper_figures._common import (
    CRITICAL_WINDOW_COLOR,
    MODEL_LABELS_SHORT,
    P2V_COLOR,
    TEXT_WIDTH,
    V2P_COLOR,
    save_fig,
    setup_style,
)
from vdpo.settings import settings

OUTPUT_DIR = settings.output_dir / "patching_last_token_res_stream"


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

    plt.rcParams["figure.constrained_layout.use"] = False
    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=(TEXT_WIDTH, TEXT_WIDTH * 0.22),
        sharey=True,
    )
    fig.subplots_adjust(wspace=0.08, left=0.06, right=0.93, top=0.85, bottom=0.22)

    for ax, model in zip(axes, models, strict=False):
        rs = grouped[model]
        n_layers = len(rs[0]["results_p2v"])
        p2v_vals, v2p_vals = compute_mean_restoration_curves(rs)

        avg = [(p + v) / 2 for p, v in zip(p2v_vals, v2p_vals, strict=False)]
        cw_start, cw_end, _ = compute_critical_window(avg)
        ax.axvspan(
            cw_start,
            cw_end,
            alpha=0.06,
            color=CRITICAL_WINDOW_COLOR,
            zorder=0,
        )

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
        ax.set_xlabel("Layer", fontsize=7)
        ax.set_xlim(0, n_layers - 1)
        ax.set_ylim(0, 1.05)
        ax.set_title(MODEL_LABELS_SHORT[model], fontsize=7, pad=3)
        ax.tick_params(labelsize=6)
        ax.tick_params(which="minor", length=2)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        for y in [0.25, 0.5, 0.75, 1.0]:
            ax.axhline(y=y, color="gray", linewidth=0.35, alpha=0.6, zorder=2)
        for y in [0.125, 0.375, 0.625, 0.875]:
            ax.axhline(y=y, color="gray", linewidth=0.2, alpha=0.35, zorder=2)

    axes[0].set_ylabel("Restoration score", fontsize=7)
    axes[-1].legend(
        fontsize=5, frameon=False, bbox_to_anchor=(1.0, 0.5), loc="center left"
    )

    save_fig(fig, "res_stream_all_models.pdf")
    print("Done.")


if __name__ == "__main__":
    main()
