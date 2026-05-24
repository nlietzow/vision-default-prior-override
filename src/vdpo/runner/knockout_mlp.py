import gc
import json
from pathlib import Path

import torch
from transformers import BatchFeature

from vdpo.runner.core import VMIRunnerBase
from vdpo.types.contrast import PRIOR_CIRCUIT_CONTRAST, ContrastSpec
from vdpo.types.enums import GroundingMode, ImageVariant, ModelFamily
from vdpo.types.models import DatasetExampleColor
from vdpo.utils.knockout_targets import load_knockout_targets
from vdpo.utils.load_model import ModelSize


class KnockoutMLP(VMIRunnerBase):
    runner_name = "knockout_mlp"

    def __init__(
        self,
        model_size: ModelSize,
        *,
        model_family: ModelFamily = ModelFamily.QWEN,
        contrast: ContrastSpec = PRIOR_CIRCUIT_CONTRAST,
    ):
        super().__init__(model_size, model_family=model_family)
        text_config = self.get_text_config()
        self._zeros = torch.zeros(text_config.hidden_size, device=self.model.device)

        self.contrast = contrast
        targets = load_knockout_targets(
            model_family.value, model_size.value, "mlp", contrast=contrast.name
        )
        self.promoting = sorted(targets["promoting"], key=lambda t: t["layer_idx"])
        self.suppressing = sorted(targets["suppressing"], key=lambda t: t["layer_idx"])

        self.knockout_configs: list[tuple[str, list[dict]]] = []
        for t in self.promoting + self.suppressing:
            kid = f"ind_L{t['layer_idx']}"
            self.knockout_configs.append((kid, [t]))
        self.knockout_configs.append(("grp_promoting", self.promoting))
        self.knockout_configs.append(("grp_suppressing", self.suppressing))

    def process_example(self, example: DatasetExampleColor) -> None:
        inputs_visual = self.build_inputs(
            example=example,
            grounding_mode=GroundingMode.VISUAL,
            image_variant=ImageVariant.COUNTERFACTUAL,
        )
        inputs_prior = self.build_inputs(
            example=example,
            grounding_mode=GroundingMode.PRIOR,
            image_variant=ImageVariant.COUNTERFACTUAL,
        )

        baseline_visual = self._forward_clean(inputs_visual)
        baseline_prior = self._forward_clean(inputs_prior)
        next_token_idx_visual = baseline_visual.argmax().item()
        next_token_idx_prior = baseline_prior.argmax().item()
        del baseline_visual, baseline_prior

        if next_token_idx_visual == next_token_idx_prior:
            self.logger.info(f"Skipping {example.example_id}: predictions agree")
            return

        for knockout_id, targets in self.knockout_configs:
            f_out = self._get_output_file(example, knockout_id)
            if f_out.exists():
                self.logger.info(f"Skipping existing: {f_out.name}")
                continue

            out_visual = self._forward_with_knockout(inputs_visual, targets)
            out_prior = self._forward_with_knockout(inputs_prior, targets)

            result = {
                "example_id": example.example_id,
                "knockout_mode": "group"
                if knockout_id.startswith("grp")
                else "individual",
                "knockout_id": knockout_id,
                "targets": targets,
                "next_token_idx_visual": next_token_idx_visual,
                "next_token_idx_prior": next_token_idx_prior,
                "visual_grounding": {
                    "logit_prior": out_visual[next_token_idx_prior].item(),
                    "logit_visual": out_visual[next_token_idx_visual].item(),
                    "predicted": "visual"
                    if out_visual[next_token_idx_visual]
                    > out_visual[next_token_idx_prior]
                    else "prior",
                },
                "prior_grounding": {
                    "logit_prior": out_prior[next_token_idx_prior].item(),
                    "logit_visual": out_prior[next_token_idx_visual].item(),
                    "predicted": "visual"
                    if out_prior[next_token_idx_visual]
                    > out_prior[next_token_idx_prior]
                    else "prior",
                },
            }

            with f_out.open("w") as f:
                json.dump(result, f, indent=4)

            del out_visual, out_prior
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @torch.no_grad()
    def _forward_clean(self, inputs: BatchFeature) -> torch.Tensor:
        with self.nnsight.trace() as tracer:  # noqa: SIM117
            with tracer.invoke(**inputs):
                outputs = self.adapter.get_lm_head_output_saved(self.nnsight)
        return outputs.cpu().clone()

    @torch.no_grad()
    def _forward_with_knockout(
        self, inputs: BatchFeature, targets: list[dict]
    ) -> torch.Tensor:
        sorted_targets = sorted(targets, key=lambda t: t["layer_idx"])
        with self.nnsight.trace() as tracer:  # noqa: SIM117
            with tracer.invoke(**inputs):
                layers = self.get_language_layers()
                for target in sorted_targets:
                    layer = layers[target["layer_idx"]]
                    self.adapter.set_mlp_output_last_token(layer, self._zeros)
                outputs = self.adapter.get_lm_head_output_saved(self.nnsight)
        return outputs.cpu().clone()

    def _get_output_file(self, example: DatasetExampleColor, knockout_id: str) -> Path:
        contrast_subdir = (
            None if self.contrast.name == "prior_circuit" else self.contrast.name
        )
        out_dir = self.get_output_dir(contrast_subdir=contrast_subdir)
        f_out = (out_dir / f"{example.example_id}__{knockout_id}").with_suffix(".json")
        f_out.parent.mkdir(parents=True, exist_ok=True)
        return f_out
