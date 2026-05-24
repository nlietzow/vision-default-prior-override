import gc
import json
from pathlib import Path

import numpy as np
import torch
from bitsandbytes.functional import dequantize_4bit
from transformers import BatchFeature

from vdpo.runner.core import VMIRunnerBase
from vdpo.types.enums import GroundingMode, ImageVariant, ModelFamily
from vdpo.types.models import DatasetExampleColor
from vdpo.utils.load_model import ModelSize


class ExtractAttentionWeights(VMIRunnerBase):
    runner_name = "extract_attention_weights"

    def __init__(
        self,
        model_size: ModelSize,
        *,
        model_family: ModelFamily = ModelFamily.QWEN,
        min_layer: int = 0,
        max_layer: int | None = None,
        early_decoding_k: int = 20,
    ):
        super().__init__(
            model_size, model_family=model_family, attn_implementation="eager"
        )
        text_config = self.get_text_config()
        self.num_heads = text_config.num_attention_heads
        self.head_dim = getattr(
            text_config, "head_dim", text_config.hidden_size // self.num_heads
        )
        self.num_layers = text_config.num_hidden_layers
        self.min_layer = min_layer
        self.max_layer = max_layer if max_layer is not None else self.num_layers
        self.early_decoding_k = early_decoding_k
        self.image_token_id = self.adapter.get_image_token_id(self.processor)

    def process_example(self, example: DatasetExampleColor) -> None:
        f_meta = self._get_output_file(example, suffix=".json")
        if f_meta.exists():
            self.logger.info(f"Skipping existing output: {f_meta}")
            return

        inputs_prior = self.build_inputs(
            example=example,
            grounding_mode=GroundingMode.PRIOR,
            image_variant=ImageVariant.COUNTERFACTUAL,
        )
        inputs_visual = self.build_inputs(
            example=example,
            grounding_mode=GroundingMode.VISUAL,
            image_variant=ImageVariant.COUNTERFACTUAL,
        )

        image_mask_prior = self._get_image_token_mask(inputs_prior)
        image_mask_visual = self._get_image_token_mask(inputs_visual)

        prior_logits, prior_attns = self._forward_with_attention(inputs_prior)
        visual_logits, visual_attns = self._forward_with_attention(inputs_visual)

        next_token_prior = prior_logits.argmax().item()
        next_token_visual = visual_logits.argmax().item()
        del prior_logits, visual_logits

        arrays = {}
        saved_layers = []
        for idx in range(self.min_layer, self.max_layer):
            arrays[f"prior_layer_{idx}"] = _aggregate_image_tokens(
                prior_attns[idx], image_mask_prior
            )
            arrays[f"visual_layer_{idx}"] = _aggregate_image_tokens(
                visual_attns[idx], image_mask_visual
            )
            saved_layers.append(idx)

        del prior_attns, visual_attns

        seq_len = (
            int(arrays[f"prior_layer_{saved_layers[0]}"].shape[-1])
            if saved_layers
            else 0
        )
        num_image_tokens = int(image_mask_prior.sum())

        # nnsight passes for per-head output vectors
        prior_head_outs = self._forward_and_cache_head_outputs(inputs_prior)
        visual_head_outs = self._forward_and_cache_head_outputs(inputs_visual)

        del inputs_prior, inputs_visual

        # Difference logit lens
        early_decoding = self._compute_early_decoding(prior_head_outs, visual_head_outs)

        # Add head outputs and early decoding to arrays
        for idx in saved_layers:
            arrays[f"prior_head_out_layer_{idx}"] = prior_head_outs[idx]
            arrays[f"visual_head_out_layer_{idx}"] = visual_head_outs[idx]
            arrays[f"diff_top_ids_layer_{idx}"] = early_decoding[idx]["top_ids"]
            arrays[f"diff_top_logits_layer_{idx}"] = early_decoding[idx]["top_logits"]
            arrays[f"diff_bottom_ids_layer_{idx}"] = early_decoding[idx]["bottom_ids"]
            arrays[f"diff_bottom_logits_layer_{idx}"] = early_decoding[idx][
                "bottom_logits"
            ]

        del prior_head_outs, visual_head_outs, early_decoding

        f_npz = self._get_output_file(example, suffix=".npz")
        np.savez_compressed(f_npz, **arrays)
        del arrays

        metadata = {
            "example_id": example.example_id,
            "next_token_prior": next_token_prior,
            "next_token_text_prior": self.processor.decode(next_token_prior),
            "next_token_visual": next_token_visual,
            "next_token_text_visual": self.processor.decode(next_token_visual),
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "early_decoding_k": self.early_decoding_k,
            "seq_len": seq_len,
            "num_image_tokens": num_image_tokens,
            "saved_layers": saved_layers,
        }
        with f_meta.open("w") as f:
            json.dump(metadata, f, indent=4)

    def _get_image_token_mask(self, inputs) -> np.ndarray:
        input_ids = inputs["input_ids"][0]  # (seq_len,)
        return (input_ids == self.image_token_id).cpu().numpy()

    @torch.inference_mode()
    def _forward_with_attention(
        self, inputs
    ) -> tuple[torch.Tensor, dict[int, np.ndarray]]:
        outputs = self.model(**inputs, output_attentions=True)
        logits = outputs.logits[0, -1, :].cpu().clone()

        attentions: dict[int, np.ndarray] = {}
        for idx in range(self.min_layer, self.max_layer):
            attn = outputs.attentions[idx][0]  # (num_heads, seq_len, seq_len)
            attentions[idx] = attn.cpu().to(torch.float16).numpy()

        del outputs
        return logits, attentions

    @torch.no_grad()
    def _forward_and_cache_head_outputs(
        self, inputs: BatchFeature
    ) -> dict[int, np.ndarray]:
        """Capture per-head o_proj input for the last token via nnsight."""
        activations: dict[int, torch.Tensor] = {}

        with self.nnsight.trace() as tracer:  # noqa: SIM117
            with tracer.invoke(**inputs):
                for idx in range(self.min_layer, self.max_layer):
                    layer = self.get_language_layers()[idx]
                    activations[idx] = self.adapter.get_o_proj_input_last_token(
                        layer
                    ).save()

        result: dict[int, np.ndarray] = {}
        for idx, act in activations.items():
            # act is (hidden_size,) → reshape to (num_heads, head_dim)
            arr = act.cpu().to(torch.float16).numpy()
            result[idx] = arr.reshape(self.num_heads, self.head_dim)

        del activations
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    @torch.no_grad()
    def _compute_early_decoding(
        self,
        prior_head_outs: dict[int, np.ndarray],
        visual_head_outs: dict[int, np.ndarray],
    ) -> dict[int, dict[str, np.ndarray]]:
        """Compute difference logit lens for each layer's attention heads.

        For each head: project (visual - prior) contribution through W_O and W_U
        to get logit-space differences, then save top-k and bottom-k tokens.
        """
        w_u = _get_weight(
            self.adapter.get_lm_head_module(self.model)
        )  # (vocab, d_model)
        k = self.early_decoding_k
        result: dict[int, dict[str, np.ndarray]] = {}

        for idx in range(self.min_layer, self.max_layer):
            layer = self.adapter.get_language_layer_modules(self.model)[idx]
            w_o = _get_weight(layer.self_attn.o_proj)  # (d_model, d_model)
            # W_O.T reshaped: (num_heads, head_dim, d_model)
            w_o_slices = w_o.T.reshape(self.num_heads, self.head_dim, -1)

            prior_heads = torch.from_numpy(prior_head_outs[idx]).float()
            visual_heads = torch.from_numpy(visual_head_outs[idx]).float()

            # Per-head contribution through W_O: (num_heads, d_model)
            prior_contrib = torch.einsum("nh,nhd->nd", prior_heads, w_o_slices)
            visual_contrib = torch.einsum("nh,nhd->nd", visual_heads, w_o_slices)
            diff = visual_contrib - prior_contrib  # (num_heads, d_model)

            # Project through unembedding: (num_heads, vocab_size)
            logits = diff @ w_u.T

            top_logits, top_ids = logits.topk(k, dim=-1)
            bottom_logits, bottom_ids = logits.topk(k, dim=-1, largest=False)

            result[idx] = {
                "top_ids": top_ids.numpy().astype(np.int32),
                "top_logits": top_logits.to(torch.float16).numpy(),
                "bottom_ids": bottom_ids.numpy().astype(np.int32),
                "bottom_logits": bottom_logits.to(torch.float16).numpy(),
            }

            del w_o, w_o_slices, prior_heads, visual_heads
            del prior_contrib, visual_contrib, diff, logits

        del w_u
        return result

    def _get_output_file(self, example: DatasetExampleColor, suffix: str) -> Path:
        f_out = (self.get_output_dir() / example.example_id).with_suffix(suffix)
        f_out.parent.mkdir(parents=True, exist_ok=True)
        return f_out


def _get_weight(linear: torch.nn.Module) -> torch.Tensor:
    """Extract the weight matrix from a linear layer, dequantizing if 4-bit."""
    w = linear.weight
    if hasattr(w, "quant_state"):
        return dequantize_4bit(w.data, w.quant_state).cpu().float()
    return w.detach().cpu().float()


def _aggregate_image_tokens(attn: np.ndarray, image_mask: np.ndarray) -> np.ndarray:
    """Aggregate all image token positions into a single position by summing.

    Args:
        attn: (num_heads, seq_len, seq_len) attention weights.
        image_mask: (seq_len,) boolean mask, True for image pad tokens.

    Returns:
        (num_heads, new_seq_len, new_seq_len) where new_seq_len = non_image + 1.
    """
    image_idx = np.where(image_mask)[0]
    if len(image_idx) == 0:
        return attn

    non_image_idx = np.where(~image_mask)[0]
    first_img = image_idx[0]
    img_new_pos = int(np.searchsorted(non_image_idx, first_img))

    # Aggregate columns (key dim): sum attention TO all image tokens
    img_col_sum = attn[:, :, image_idx].sum(axis=2, keepdims=True)
    non_img_cols = attn[:, :, non_image_idx]
    cols_agg = np.concatenate(
        [
            non_img_cols[:, :, :img_new_pos],
            img_col_sum,
            non_img_cols[:, :, img_new_pos:],
        ],
        axis=2,
    )

    # Aggregate rows (query dim): sum attention FROM all image tokens
    img_row_sum = cols_agg[:, image_idx, :].sum(axis=1, keepdims=True)
    non_img_rows = cols_agg[:, non_image_idx, :]
    return np.concatenate(
        [
            non_img_rows[:, :img_new_pos, :],
            img_row_sum,
            non_img_rows[:, img_new_pos:, :],
        ],
        axis=1,
    )
