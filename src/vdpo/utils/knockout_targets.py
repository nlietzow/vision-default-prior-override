"""Load knockout targets from PCA classification outputs."""

import json
from pathlib import Path

_DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"


def _classification_path(component: str, contrast: str) -> Path:
    if component not in ("attn_heads", "mlp"):
        raise ValueError(f"Unknown component {component!r}. Expected: attn_heads, mlp.")
    base = "classify_attn_heads" if component == "attn_heads" else "classify_mlp_layers"
    if contrast == "prior_circuit":
        return _DATA_DIR / f"{base}.json"
    return _DATA_DIR / f"{base}_{contrast}.json"


def load_knockout_targets(
    family: str,
    size: str,
    component: str,
    contrast: str = "prior_circuit",
) -> dict[str, list[dict]]:
    """Load knockout targets for a model.

    Args:
        family: Model family string (e.g., "qwen").
        size: Model size string (e.g., "3B").
        component: "attn_heads" or "mlp".
        contrast: "prior_circuit" (default) or "visual_circuit".

    Returns:
        Dict with keys "promoting" and "suppressing", each mapping to a list
        of target dicts. Attention heads: {"layer_idx": int, "head_idx": int}.
        MLP layers: {"layer_idx": int}.
    """
    path = _classification_path(component, contrast)
    with path.open() as f:
        raw = json.load(f)

    key = f"{family}/{size}"
    if key not in raw:
        raise KeyError(
            f"No knockout targets for {key} in {path.name}. "
            f"Available: {list(raw.keys())}"
        )

    entry = raw[key]

    if component == "attn_heads":
        return {
            "promoting": [
                {"layer_idx": h["layer"], "head_idx": h["head"]}
                for h in entry["promoting"]
            ],
            "suppressing": [
                {"layer_idx": h["layer"], "head_idx": h["head"]}
                for h in entry["suppressing"]
            ],
        }

    return {
        "promoting": [{"layer_idx": idx} for idx in entry["promoting"]],
        "suppressing": [{"layer_idx": idx} for idx in entry["suppressing"]],
    }
