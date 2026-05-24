"""Shared utilities for attention pattern analysis scripts.

Provides:
- get_image_position: detect aggregated image token position for a model
- load_attention_data: load correctly-conflicting example IDs and data dir
- compute_image_attention_fractions: average image-token attention per head
"""

from functools import cache
from pathlib import Path

import numpy as np
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import (
    LlavaNextProcessor,
    PaliGemmaProcessor,
    Qwen2_5_VLProcessor,
)

from scripts.analysis._patching_common import load_correct_example_ids, sort_models
from vdpo.settings import settings

# ---------------------------------------------------------------------------
# Model configs
# NOTE: HuggingFace paths must stay in sync with the VLM adapters in
# src/vdpo/vlm/{qwen,llava_next,paligemma_2}.py.
# ---------------------------------------------------------------------------

_MODEL_CONFIGS: dict[str, dict] = {
    "qwen/3B": {
        "hf_path": "Qwen/Qwen2.5-VL-3B-Instruct",
        "family": "qwen",
        "size": "3B",
    },
    "qwen/7B": {
        "hf_path": "Qwen/Qwen2.5-VL-7B-Instruct",
        "family": "qwen",
        "size": "7B",
    },
    "llava_next/7B": {
        "hf_path": "llava-hf/llava-v1.6-mistral-7b-hf",
        "family": "llava_next",
        "size": "7B",
    },
    "paligemma_2/3B": {
        "hf_path": "google/paligemma2-3b-mix-448",
        "family": "paligemma_2",
        "size": "3B",
    },
    "paligemma_2/10B": {
        "hf_path": "google/paligemma2-10b-mix-448",
        "family": "paligemma_2",
        "size": "10B",
    },
}

OUTPUT_DIR = settings.output_dir / "extract_attention_weights"

_DUMMY_QUESTION = "What color is this object?"


def _make_dummy_image() -> Image.Image:
    return Image.new("RGB", (224, 224), color=(128, 128, 128))


# ---------------------------------------------------------------------------
# 1. get_image_position
# ---------------------------------------------------------------------------


@cache
def get_image_position(model: str) -> int:
    """Return the aggregated image token position for a model.

    Loads only the processor (no GPU), tokenizes a dummy prompt, finds
    `image_token_id` positions, and computes the aggregated image position
    using searchsorted(non_image_idx, first_img) — the same logic as
    _aggregate_image_tokens in the extraction runner.

    Expected results:
        qwen/3B → 15, qwen/7B → 15
        llava_next/7B → 5
        paligemma_2/3B → 0, paligemma_2/10B → 0
    """
    cfg = _MODEL_CONFIGS[model]
    path = cfg["hf_path"]
    family = cfg["family"]
    dummy_image = _make_dummy_image()
    question = _DUMMY_QUESTION

    if family == "qwen":
        processor = Qwen2_5_VLProcessor.from_pretrained(
            path,
            use_fast=True,
            min_pixels=1280 * 28 * 28,
            max_pixels=1280 * 28 * 28,
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": dummy_image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, *_ = process_vision_info(messages)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors="pt",
        )

    elif family == "llava_next":
        processor = LlavaNextProcessor.from_pretrained(path, use_fast=True)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=text,
            images=dummy_image,
            padding=False,
            return_tensors="pt",
        )

    elif family == "paligemma_2":
        processor = PaliGemmaProcessor.from_pretrained(path, use_fast=True)
        inputs = processor(
            images=dummy_image.convert("RGB"),
            text=question,
            padding=False,
            return_tensors="pt",
        )

    else:
        raise ValueError(f"Unknown family: {family!r}")

    image_token_id = processor.image_token_id
    input_ids = inputs["input_ids"][0].numpy()
    image_mask = input_ids == image_token_id

    image_idx = np.where(image_mask)[0]
    if len(image_idx) == 0:
        raise RuntimeError(f"No image tokens found in dummy prompt for model {model!r}")

    non_image_idx = np.where(~image_mask)[0]
    first_img = image_idx[0]
    return int(np.searchsorted(non_image_idx, first_img))


# ---------------------------------------------------------------------------
# 2. load_attention_data
# ---------------------------------------------------------------------------


def load_attention_data(model: str) -> tuple[list[str], Path]:
    """Return (example_ids, data_dir) for a model.

    Only returns example IDs that have .npz files AND are in
    load_correct_example_ids() (correctly-conflicting examples).
    """
    cfg = _MODEL_CONFIGS[model]
    family = cfg["family"]
    size = cfg["size"]
    data_dir = OUTPUT_DIR / family / size

    correct_ids_by_model = load_correct_example_ids()
    correct_ids: set[str] = correct_ids_by_model.get(model, set())

    example_ids: list[str] = []
    if data_dir.exists():
        for npz_path in sorted(data_dir.glob("*.npz")):
            example_id = npz_path.stem
            if example_id in correct_ids:
                example_ids.append(example_id)

    return example_ids, data_dir


# ---------------------------------------------------------------------------
# 3. compute_image_attention_fractions
# ---------------------------------------------------------------------------


def compute_image_attention_fractions(
    model: str,
    heads: list[dict],
    image_pos: int,
) -> dict[tuple[int, int], dict[str, float]]:
    """Compute average image-token attention fractions for the given heads.

    Args:
        model: Model key, e.g. "qwen/3B".
        heads: List of {"layer": int, "head": int} dicts.
        image_pos: Aggregated image token position in the sequence
            (as returned by get_image_position).

    Returns:
        {(layer, head): {"visual": float, "prior": float, "delta": float}}
        averaged across all correctly-conflicting examples with .npz files.
        delta = visual - prior (positive means more image attention under
        visual grounding).
    """
    example_ids, data_dir = load_attention_data(model)

    # accumulators: (layer, head) → [visual_vals, prior_vals]
    visual_acc: dict[tuple[int, int], list[float]] = {}
    prior_acc: dict[tuple[int, int], list[float]] = {}
    for h in heads:
        key = (h["layer"], h["head"])
        visual_acc[key] = []
        prior_acc[key] = []

    for example_id in example_ids:
        npz_path = data_dir / f"{example_id}.npz"
        with np.load(npz_path) as data:
            for h in heads:
                layer = h["layer"]
                head = h["head"]
                key = (layer, head)

                visual_key = f"visual_layer_{layer}"
                prior_key = f"prior_layer_{layer}"

                if visual_key not in data or prior_key not in data:
                    continue

                # shape: (n_heads, new_seq_len, new_seq_len)
                visual_attn = data[visual_key]
                prior_attn = data[prior_key]

                assert image_pos < visual_attn.shape[-1], (
                    f"image_pos={image_pos} out of bounds for "
                    f"seq_len={visual_attn.shape[-1]} in {npz_path.name}"
                )

                # last token attention to image position
                visual_acc[key].append(float(visual_attn[head, -1, image_pos]))
                prior_acc[key].append(float(prior_attn[head, -1, image_pos]))

    result: dict[tuple[int, int], dict[str, float]] = {}
    for h in heads:
        key = (h["layer"], h["head"])
        v_vals = visual_acc[key]
        p_vals = prior_acc[key]
        if not v_vals:
            result[key] = {
                "visual": float("nan"),
                "prior": float("nan"),
                "delta": float("nan"),
            }
        else:
            v_mean = float(np.mean(v_vals))
            p_mean = float(np.mean(p_vals))
            result[key] = {
                "visual": v_mean,
                "prior": p_mean,
                "delta": v_mean - p_mean,
            }

    return result


# ---------------------------------------------------------------------------
# Re-export sort_models for convenience
# ---------------------------------------------------------------------------

__all__ = [
    "OUTPUT_DIR",
    "compute_image_attention_fractions",
    "get_image_position",
    "load_attention_data",
    "sort_models",
]
