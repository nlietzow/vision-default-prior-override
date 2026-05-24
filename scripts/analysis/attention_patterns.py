"""Analyze attention pattern results: image-token attention fractions for classified heads."""

import json
import math
from pathlib import Path

from scripts.analysis._attention_common import (
    compute_image_attention_fractions,
    get_image_position,
    load_attention_data,
    sort_models,
)
from scripts.analysis._patching_common import print_section

_DATA_DIR = Path(__file__).parent.parent.parent / "data"
_CLASSIFICATIONS_PATH = _DATA_DIR / "classify_attn_heads.json"


def main():
    with _CLASSIFICATIONS_PATH.open() as f:
        classifications: dict[str, dict[str, list[dict[str, int]]]] = json.load(f)

    print_section("ATTENTION PATTERN ANALYSIS: IMAGE-TOKEN ATTENTION FRACTIONS")

    for model in sort_models(list(classifications.keys())):
        promoting = classifications[model].get("promoting", [])
        suppressing = classifications[model].get("suppressing", [])
        all_heads = promoting + suppressing

        image_pos = get_image_position(model)
        example_ids, _ = load_attention_data(model)
        n_examples = len(example_ids)

        fractions = compute_image_attention_fractions(model, all_heads, image_pos)

        n_promoting = len(promoting)
        n_suppressing = len(suppressing)

        print(
            f"\n  {model}"
            f"  (n={n_examples},"
            f" promoting={n_promoting},"
            f" suppressing={n_suppressing},"
            f" image_pos={image_pos})"
        )
        print(f"  {'Head':>8}  {'Type':>12}  {'Visual':>8}  {'Prior':>8}  {'Delta':>8}")
        print(f"  {'-' * 52}")

        for h in promoting:
            key = (h["layer"], h["head"])
            vals = fractions.get(
                key,
                {"visual": float("nan"), "prior": float("nan"), "delta": float("nan")},
            )
            print(
                f"  L{h['layer']:>2}H{h['head']:<2}"
                f"  {'promoting':>12}"
                f"  {vals['visual']:>8.4f}"
                f"  {vals['prior']:>8.4f}"
                f"  {vals['delta']:>+8.4f}"
            )

        for h in suppressing:
            key = (h["layer"], h["head"])
            vals = fractions.get(
                key,
                {"visual": float("nan"), "prior": float("nan"), "delta": float("nan")},
            )
            print(
                f"  L{h['layer']:>2}H{h['head']:<2}"
                f"  {'suppressing':>12}"
                f"  {vals['visual']:>8.4f}"
                f"  {vals['prior']:>8.4f}"
                f"  {vals['delta']:>+8.4f}"
            )

        # Summary
        promoting_deltas = [
            fractions[(h["layer"], h["head"])]["delta"]
            for h in promoting
            if (h["layer"], h["head"]) in fractions
            and not math.isnan(fractions[(h["layer"], h["head"])]["delta"])
        ]
        suppressing_deltas = [
            fractions[(h["layer"], h["head"])]["delta"]
            for h in suppressing
            if (h["layer"], h["head"]) in fractions
            and not math.isnan(fractions[(h["layer"], h["head"])]["delta"])
        ]

        if promoting_deltas:
            mean_p = sum(promoting_deltas) / len(promoting_deltas)
            print(f"\n  Summary: promoting  mean delta = {mean_p:>+.4f}")
        if suppressing_deltas:
            mean_s = sum(suppressing_deltas) / len(suppressing_deltas)
            print(f"  Summary: suppressing mean delta = {mean_s:>+.4f}")


if __name__ == "__main__":
    main()
