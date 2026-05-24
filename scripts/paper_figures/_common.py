"""Shared configuration for paper-quality figures using tueplots."""

from pathlib import Path

import matplotlib.pyplot as plt
from tueplots import fonts, fontsizes

# ACL 2023 two-column format dimensions (inches)
COLUMN_WIDTH = 3.31
TEXT_WIDTH = 6.83

# Output directory for paper PDFs
FIGURES_DIR = Path(__file__).parent.parent.parent / "figures"

MODEL_LABELS = {
    "qwen/3B": "Qwen 2.5 VL 3B",
    "qwen/7B": "Qwen 2.5 VL 7B",
    "llava_next/7B": "LLaVA-NeXT 7B",
    "paligemma_2/3B": "PaliGemma 2 3B",
    "paligemma_2/10B": "PaliGemma 2 10B",
}

MODEL_LABELS_SHORT = {
    "qwen/3B": "Qwen-VL 3B",
    "qwen/7B": "Qwen-VL 7B",
    "llava_next/7B": "LLaVA-NeXT 7B",
    "paligemma_2/3B": "PaliGemma 3B",
    "paligemma_2/10B": "PaliGemma 10B",
}

# Paul Tol muted palette
# https://personal.sron.nl/~pault/#sec:qualitative
COLORS = {
    "indigo": "#332288",
    "cyan": "#88CCEE",
    "teal": "#44AA99",
    "green": "#117733",
    "olive": "#999933",
    "sand": "#DDCC77",
    "rose": "#CC6677",
    "wine": "#882255",
    "purple": "#AA4499",
    "grey": "#DDDDDD",
}

# Semantic assignments
P2V_COLOR = COLORS["rose"]
V2P_COLOR = COLORS["indigo"]
PROMOTING_COLOR = COLORS["rose"]
SUPPRESSING_COLOR = COLORS["indigo"]
VISUAL_COLOR = "#286EAA"  # matches \Visual macro: RGB(40, 110, 170)
PRIOR_COLOR = "#B43232"  # matches \Prior macro: RGB(180, 50, 50)
CRITICAL_WINDOW_COLOR = "#555555"


def setup_style():
    """Apply tueplots styling for ACL paper figures."""
    plt.rcParams.update(fonts.neurips2024())
    plt.rcParams.update(fontsizes.neurips2024())
    plt.rcParams.update(
        {
            "figure.constrained_layout.use": True,
            "figure.autolayout": False,
            "savefig.pad_inches": 0.015,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_fig(fig, name: str):
    """Save figure to paper figures directory as PDF."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
