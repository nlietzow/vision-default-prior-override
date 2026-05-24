import gc
import json
from pathlib import Path

import torch
from transformers import BatchFeature

from vdpo.runner.core import VMIRunnerBase
from vdpo.types.contrast import PRIOR_CIRCUIT_CONTRAST, ContrastSpec
from vdpo.types.enums import ModelFamily
from vdpo.types.models import DatasetExampleColor
from vdpo.utils.load_model import ModelSize


class PatchingLastTokenAttnHeads(VMIRunnerBase):
    runner_name = "patching_last_token_attn_heads"

    def __init__(
        self,
        model_size: ModelSize,
        *,
        model_family: ModelFamily = ModelFamily.QWEN,
        contrast: ContrastSpec = PRIOR_CIRCUIT_CONTRAST,
        min_layer: int = 0,
        max_layer: int | None = None,
    ):
        super().__init__(model_size, model_family=model_family)
        self.contrast = contrast
        self.min_layer = min_layer
        self.max_layer = max_layer
        text_config = self.get_text_config()
        self.num_heads = text_config.num_attention_heads
        self.head_dim = getattr(
            text_config, "head_dim", text_config.hidden_size // self.num_heads
        )

    # noinspection DuplicatedCode
    def process_example(
        self,
        example: DatasetExampleColor,
    ) -> None:
        f_out = self._get_output_file(example=example)
        if f_out.exists():
            self.logger.info(f"Skipping existing output: {f_out}")
            return

        src_grounding, src_image = self.contrast.source
        tgt_grounding, tgt_image = self.contrast.target

        inputs_source = self.build_inputs(
            example=example,
            grounding_mode=src_grounding,
            image_variant=src_image,
        )
        inputs_target = self.build_inputs(
            example=example,
            grounding_mode=tgt_grounding,
            image_variant=tgt_image,
        )
        if inputs_source["input_ids"].shape != inputs_target["input_ids"].shape:
            raise ValueError(
                f"Input shapes do not match: "
                f"{inputs_source['input_ids'].shape} vs "
                f"{inputs_target['input_ids'].shape}"
            )

        outputs_source, activations_source = self._forward_and_cache(inputs_source)
        outputs_target, activations_target = self._forward_and_cache(inputs_target)

        next_token_idx_source = outputs_source.argmax().item()
        next_token_idx_target = outputs_target.argmax().item()
        source_token_logit_source_run = outputs_source[next_token_idx_source].item()
        target_token_logit_source_run = outputs_source[next_token_idx_target].item()
        source_token_logit_target_run = outputs_target[next_token_idx_source].item()
        target_token_logit_target_run = outputs_target[next_token_idx_target].item()

        del outputs_source, outputs_target

        results_source_to_target: list[dict] = []
        results_target_to_source: list[dict] = []
        if next_token_idx_source != next_token_idx_target:
            results_source_to_target = self._run_patching(
                inputs=inputs_target,
                activations=activations_source,
                next_token_idx_source=next_token_idx_source,
                next_token_idx_target=next_token_idx_target,
            )
            del activations_source
            results_target_to_source = self._run_patching(
                inputs=inputs_source,
                activations=activations_target,
                next_token_idx_source=next_token_idx_source,
                next_token_idx_target=next_token_idx_target,
            )
            del activations_target

        payload = self._build_payload(
            example_id=example.example_id,
            next_token_idx_source=next_token_idx_source,
            next_token_idx_target=next_token_idx_target,
            source_token_logit_source_run=source_token_logit_source_run,
            target_token_logit_source_run=target_token_logit_source_run,
            source_token_logit_target_run=source_token_logit_target_run,
            target_token_logit_target_run=target_token_logit_target_run,
            results_source_to_target=results_source_to_target,
            results_target_to_source=results_target_to_source,
        )
        with f_out.open("w") as f:
            json.dump(payload, f, indent=4)

    def _build_payload(
        self,
        *,
        example_id: str,
        next_token_idx_source: int,
        next_token_idx_target: int,
        source_token_logit_source_run: float,
        target_token_logit_source_run: float,
        source_token_logit_target_run: float,
        target_token_logit_target_run: float,
        results_source_to_target: list[dict],
        results_target_to_source: list[dict],
    ) -> dict:
        """Serialize results. Use legacy prior/visual field names for the
        prior-circuit contrast to keep existing analysis scripts working."""
        if self.contrast.name == "prior_circuit":
            return {
                "example_id": example_id,
                "next_token_idx_prior": next_token_idx_source,
                "next_token_idx_visual": next_token_idx_target,
                "prior_token_logit_prior_run": source_token_logit_source_run,
                "visual_token_logit_prior_run": target_token_logit_source_run,
                "prior_token_logit_visual_run": source_token_logit_target_run,
                "visual_token_logit_visual_run": target_token_logit_target_run,
                "results_p2v": results_source_to_target,
                "results_v2p": results_target_to_source,
            }
        return {
            "example_id": example_id,
            "contrast": self.contrast.name,
            "next_token_idx_source": next_token_idx_source,
            "next_token_idx_target": next_token_idx_target,
            "source_token_logit_source_run": source_token_logit_source_run,
            "target_token_logit_source_run": target_token_logit_source_run,
            "source_token_logit_target_run": source_token_logit_target_run,
            "target_token_logit_target_run": target_token_logit_target_run,
            "results_source_to_target": results_source_to_target,
            "results_target_to_source": results_target_to_source,
        }

    def _run_patching(
        self,
        inputs: BatchFeature,
        activations: dict[int, torch.Tensor],
        next_token_idx_source: int,
        next_token_idx_target: int,
    ) -> list[dict]:
        results = []
        for layer_idx, activation in activations.items():
            activation = activation.to(self.nnsight.device)
            layer_results = self._forward_with_patching(
                inputs=inputs,
                layer_idx_to_patch=layer_idx,
                activations_to_patch=activation,
                next_token_idx_source=next_token_idx_source,
                next_token_idx_target=next_token_idx_target,
            )
            del activation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            results.extend(layer_results)
        return results

    @torch.no_grad()
    def _forward_and_cache(
        self, inputs: BatchFeature
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        activations: dict[int, torch.Tensor] = {}
        layers = self.get_language_layers()
        max_layer = self.max_layer if self.max_layer is not None else len(layers)

        with self.nnsight.trace() as tracer:  # noqa: SIM117
            with tracer.invoke(**inputs):
                for idx, layer in enumerate(layers):
                    if self.min_layer <= idx < max_layer:
                        activations[idx] = self.adapter.get_o_proj_input_last_token(
                            layer
                        ).save()
                outputs = self.adapter.get_lm_head_output_saved(self.nnsight)

        return outputs.cpu().clone(), {
            idx: act.cpu().clone() for idx, act in activations.items()
        }

    @torch.no_grad()
    def _forward_with_patching(
        self,
        inputs: BatchFeature,
        layer_idx_to_patch: int,
        activations_to_patch: torch.Tensor,
        next_token_idx_source: int,
        next_token_idx_target: int,
    ) -> list[dict]:
        results = []
        for head_idx in range(self.num_heads):
            start = head_idx * self.head_dim
            end = start + self.head_dim

            with self.nnsight.trace() as tracer:  # noqa: SIM117
                with tracer.invoke(**inputs):
                    layer = self.get_language_layers()[layer_idx_to_patch]
                    self.adapter.set_o_proj_input_slice(
                        layer, start, end, activations_to_patch[start:end]
                    )
                    outputs = self.adapter.get_lm_head_output_saved(self.nnsight)

            results.append(
                self._layer_result(
                    layer_idx_to_patch=layer_idx_to_patch,
                    head_idx=head_idx,
                    logit_source=outputs[next_token_idx_source].item(),
                    logit_target=outputs[next_token_idx_target].item(),
                )
            )
            del outputs, tracer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results

    def _layer_result(
        self,
        *,
        layer_idx_to_patch: int,
        head_idx: int,
        logit_source: float,
        logit_target: float,
    ) -> dict:
        if self.contrast.name == "prior_circuit":
            return {
                "layer_idx": layer_idx_to_patch,
                "head_idx": head_idx,
                "logit_prior": logit_source,
                "logit_visual": logit_target,
            }
        return {
            "layer_idx": layer_idx_to_patch,
            "head_idx": head_idx,
            "logit_source": logit_source,
            "logit_target": logit_target,
        }

    def _get_output_file(self, example: DatasetExampleColor) -> Path:
        contrast_subdir = (
            None if self.contrast.name == "prior_circuit" else self.contrast.name
        )
        out_dir = self.get_output_dir(contrast_subdir=contrast_subdir)
        f_out = (out_dir / example.example_id).with_suffix(".json")
        f_out.parent.mkdir(parents=True, exist_ok=True)
        return f_out
