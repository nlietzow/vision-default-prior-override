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


class PatchingLastTokenMLP(VMIRunnerBase):
    runner_name = "patching_last_token_mlp"

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
        s_logit_s = outputs_source[next_token_idx_source].item()
        t_logit_s = outputs_source[next_token_idx_target].item()
        s_logit_t = outputs_target[next_token_idx_source].item()
        t_logit_t = outputs_target[next_token_idx_target].item()

        del outputs_source, outputs_target

        results_s2t: list[dict] = []
        results_t2s: list[dict] = []
        if next_token_idx_source != next_token_idx_target:
            results_s2t = self._run_patching(
                inputs=inputs_target,
                activations=activations_source,
                next_token_idx_source=next_token_idx_source,
                next_token_idx_target=next_token_idx_target,
            )
            del activations_source
            results_t2s = self._run_patching(
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
            s_logit_s=s_logit_s,
            t_logit_s=t_logit_s,
            s_logit_t=s_logit_t,
            t_logit_t=t_logit_t,
            results_s2t=results_s2t,
            results_t2s=results_t2s,
        )
        with f_out.open("w") as f:
            json.dump(payload, f, indent=4)

    def _build_payload(
        self,
        *,
        example_id: str,
        next_token_idx_source: int,
        next_token_idx_target: int,
        s_logit_s: float,
        t_logit_s: float,
        s_logit_t: float,
        t_logit_t: float,
        results_s2t: list[dict],
        results_t2s: list[dict],
    ) -> dict:
        if self.contrast.name == "prior_circuit":
            return {
                "example_id": example_id,
                "next_token_idx_prior": next_token_idx_source,
                "next_token_idx_visual": next_token_idx_target,
                "prior_token_logit_prior_run": s_logit_s,
                "visual_token_logit_prior_run": t_logit_s,
                "prior_token_logit_visual_run": s_logit_t,
                "visual_token_logit_visual_run": t_logit_t,
                "results_p2v": results_s2t,
                "results_v2p": results_t2s,
            }
        return {
            "example_id": example_id,
            "contrast": self.contrast.name,
            "next_token_idx_source": next_token_idx_source,
            "next_token_idx_target": next_token_idx_target,
            "source_token_logit_source_run": s_logit_s,
            "target_token_logit_source_run": t_logit_s,
            "source_token_logit_target_run": s_logit_t,
            "target_token_logit_target_run": t_logit_t,
            "results_source_to_target": results_s2t,
            "results_target_to_source": results_t2s,
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
            result = self._forward_with_patching(
                inputs=inputs,
                layer_idx_to_patch=layer_idx,
                activations_to_patch=activation,
                next_token_idx_source=next_token_idx_source,
                next_token_idx_target=next_token_idx_target,
            )
            del activation
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            results.append(result)
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
                        activations[idx] = self.adapter.get_mlp_output_last_token(
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
    ) -> dict:
        with self.nnsight.trace() as tracer:  # noqa: SIM117
            with tracer.invoke(**inputs):
                layer = self.get_language_layers()[layer_idx_to_patch]
                self.adapter.set_mlp_output_last_token(layer, activations_to_patch)
                outputs = self.adapter.get_lm_head_output_saved(self.nnsight)

        if self.contrast.name == "prior_circuit":
            result = {
                "layer_idx_to_patch": layer_idx_to_patch,
                "logit_prior": outputs[next_token_idx_source].item(),
                "logit_visual": outputs[next_token_idx_target].item(),
            }
        else:
            result = {
                "layer_idx_to_patch": layer_idx_to_patch,
                "logit_source": outputs[next_token_idx_source].item(),
                "logit_target": outputs[next_token_idx_target].item(),
            }
        del outputs, tracer
        return result

    def _get_output_file(self, example: DatasetExampleColor) -> Path:
        contrast_subdir = (
            None if self.contrast.name == "prior_circuit" else self.contrast.name
        )
        out_dir = self.get_output_dir(contrast_subdir=contrast_subdir)
        f_out = (out_dir / example.example_id).with_suffix(".json")
        f_out.parent.mkdir(parents=True, exist_ok=True)
        return f_out
