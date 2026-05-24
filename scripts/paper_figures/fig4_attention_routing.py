"""Figure 4: Image-attention fraction for classified heads under both groundings.

Two separate PDFs (one per model) for a figure* environment:
  - attention_routing_qwen.pdf: Qwen 7B — attention routing mechanism
  - attention_routing_paligemma.pdf: PaliGemma 3B — value-space modulation

Output: figures/attention_routing_{qwen,paligemma}.pdf
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

from scripts.analysis._attention_common import (
    compute_image_attention_fractions,
    get_image_position,
)
from scripts.paper_figures._common import (
    COLORS,
    PRIOR_COLOR,
    TEXT_WIDTH,
    VISUAL_COLOR,
    save_fig,
    setup_style,
)

_DATA_DIR = Path("data")
_CLASSIFICATIONS_PATH = _DATA_DIR / "classify_attn_heads.json"


def _plot_single_model(
    classifications: dict,
    model: str,
    title: str,
    output_name: str,
):
    promoting = classifications[model]["promoting"]
    suppressing = classifications[model]["suppressing"]
    all_heads = promoting + suppressing

    image_pos = get_image_position(model)
    fractions = compute_image_attention_fractions(model, all_heads, image_pos)

    head_labels = [f"L{h['layer']}H{h['head']}" for h in all_heads]
    x = np.arange(len(all_heads))
    width = 0.35

    _nan = {"visual": float("nan"), "prior": float("nan")}
    visual_vals = [
        fractions.get((h["layer"], h["head"]), _nan)["visual"] for h in all_heads
    ]
    prior_vals = [
        fractions.get((h["layer"], h["head"]), _nan)["prior"] for h in all_heads
    ]

    plt.rcParams["figure.constrained_layout.use"] = False
    fig, ax = plt.subplots(
        figsize=(TEXT_WIDTH * 0.48, TEXT_WIDTH * 0.48 * 0.55),
    )
    fig.subplots_adjust(left=0.12, right=0.85, top=0.88, bottom=0.28)

    ax.bar(
        x - width / 2,
        visual_vals,
        width,
        color=VISUAL_COLOR,
        label="Visual",
    )
    ax.bar(
        x + width / 2,
        prior_vals,
        width,
        color=PRIOR_COLOR,
        label="Prior",
    )

    ax.set_ylim(0, 1.05)

    # Separator between promoting and suppressing
    n_promoting = len(promoting)
    if 0 < n_promoting < len(all_heads):
        sep_x = n_promoting - 0.5
        ax.axvline(x=sep_x, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)
        # Shaded backgrounds to distinguish groups
        ax.axvspan(-0.5, sep_x, alpha=0.03, color=COLORS["rose"], zorder=0)
        ax.axvspan(
            sep_x, len(all_heads) - 0.5, alpha=0.03, color=COLORS["indigo"], zorder=0
        )
        # Labels just below 1.0 line
        ax.text(
            (n_promoting - 1) / 2,
            0.98,
            "promoting",
            ha="center",
            va="top",
            fontsize=5,
            color="#777777",
            style="italic",
        )
        ax.text(
            n_promoting + (len(all_heads) - n_promoting - 1) / 2,
            0.98,
            "suppressing",
            ha="center",
            va="top",
            fontsize=5,
            color="#777777",
            style="italic",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(head_labels, rotation=90, ha="center", fontsize=4)
    ax.set_ylabel("Image-attn. fraction", fontsize=7)
    ax.set_title(title, fontsize=7, pad=3)
    ax.tick_params(labelsize=6)
    ax.tick_params(which="minor", length=2)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.legend(fontsize=5, frameon=False, bbox_to_anchor=(1.0, 1.0), loc="upper left")

    # Grid
    for y in [0.25, 0.5, 0.75, 1.0]:
        ax.axhline(y=y, color="gray", linewidth=0.3, alpha=0.5, zorder=0)

    save_fig(fig, output_name)


def main():
    setup_style()

    print("Loading classifications...")
    with _CLASSIFICATIONS_PATH.open() as f:
        classifications = json.load(f)

    _plot_single_model(
        classifications,
        "qwen/3B",
        "(a) Qwen-VL 3B",
        "attention_routing_qwen.pdf",
    )
    _plot_single_model(
        classifications,
        "paligemma_2/3B",
        "(b) PaliGemma 3B",
        "attention_routing_paligemma.pdf",
    )

    print("Done.")


if __name__ == "__main__":
    main()
