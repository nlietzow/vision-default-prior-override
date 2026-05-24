"""Appendix figure: Image-attention fraction for classified heads, all 5 models.

Horizontal bar chart per model showing visual vs prior image-attention
fractions for every classified head. Promoting and suppressing groups
are separated; the visual bar (indigo) and prior bar (rose) are paired
per head. Qwen-VL and LLaVA-NeXT show large deltas (attention routing);
PaliGemma stays high under both conditions (value-space routing).

Output: figures/attention_routing_all.pdf
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator

from scripts.analysis._attention_common import (
    compute_image_attention_fractions,
    get_image_position,
)
from scripts.analysis._patching_common import sort_models
from scripts.paper_figures._common import (
    MODEL_LABELS_SHORT,
    PRIOR_COLOR,
    TEXT_WIDTH,
    VISUAL_COLOR,
    save_fig,
    setup_style,
)

_CLASSIFICATIONS_PATH = Path("data") / "classify_attn_heads.json"


def main():
    setup_style()

    print("Loading classifications...")
    with _CLASSIFICATIONS_PATH.open() as f:
        classifications = json.load(f)

    models = sort_models(list(classifications.keys()))
    n = len(models)

    plt.rcParams["figure.constrained_layout.use"] = False
    fig, axes = plt.subplots(
        1,
        n,
        figsize=(TEXT_WIDTH, TEXT_WIDTH * 0.55),
    )
    fig.subplots_adjust(wspace=0.55, left=0.05, right=0.90, top=0.92, bottom=0.06)

    bar_height = 0.4

    for ax, model in zip(axes, models, strict=False):
        promoting = classifications[model]["promoting"]
        suppressing = classifications[model]["suppressing"]
        all_heads = promoting + suppressing
        n_promoting = len(promoting)

        image_pos = get_image_position(model)
        fractions = compute_image_attention_fractions(model, all_heads, image_pos)

        labels = [f"L{h['layer']}H{h['head']}" for h in all_heads]
        visual_vals = [fractions[(h["layer"], h["head"])]["visual"] for h in all_heads]
        prior_vals = [fractions[(h["layer"], h["head"])]["prior"] for h in all_heads]
        y_pos = np.arange(len(all_heads))

        ax.barh(
            y_pos - bar_height / 2,
            visual_vals,
            color=VISUAL_COLOR,
            height=bar_height,
            label="Visual",
        )
        ax.barh(
            y_pos + bar_height / 2,
            prior_vals,
            color=PRIOR_COLOR,
            height=bar_height,
            label="Prior",
        )

        if 0 < n_promoting < len(all_heads):
            sep_y = n_promoting - 0.5
            ax.axhline(y=sep_y, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=4)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Image-attn. frac.", fontsize=6)
        ax.set_title(MODEL_LABELS_SHORT[model], fontsize=7, pad=3)
        ax.tick_params(labelsize=5)
        ax.tick_params(which="minor", length=2)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.invert_yaxis()

        for x_val in [0.25, 0.5, 0.75, 1.0]:
            ax.axvline(x=x_val, color="gray", linewidth=0.3, alpha=0.4, zorder=0)

    legend_elements = [
        Patch(facecolor=VISUAL_COLOR, label="Visual"),
        Patch(facecolor=PRIOR_COLOR, label="Prior"),
    ]
    axes[-1].legend(
        handles=legend_elements,
        fontsize=5,
        frameon=False,
        bbox_to_anchor=(1.0, 0.5),
        loc="center left",
    )

    save_fig(fig, "attention_routing_all.pdf")
    print("Done.")


if __name__ == "__main__":
    main()
