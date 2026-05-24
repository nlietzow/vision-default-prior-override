import json
from pathlib import Path

import torch

from vdpo.runner.core import VMIRunnerBase
from vdpo.types.enums import GroundingMode, ImageVariant
from vdpo.types.models import DatasetExampleColor


class InferenceRunner(VMIRunnerBase):
    runner_name = "inference"

    @torch.no_grad()
    def process_example(
        self,
        example: DatasetExampleColor,
        *,
        grounding_mode: GroundingMode = GroundingMode.VISUAL,
        image_variant: ImageVariant = ImageVariant.ORIGINAL,
    ) -> None:
        f_out = self.get_output_file(
            example=example,
            grounding_mode=grounding_mode,
            image_variant=image_variant,
        )
        if f_out.exists():
            self.logger.info(f"Skipping existing output: {f_out}")
            return None

        inputs = self.build_inputs(
            example=example,
            grounding_mode=grounding_mode,
            image_variant=image_variant,
        )
        outputs = self.model(**inputs).logits[0, -1, :]

        next_token_idx = outputs.argmax().item()
        next_token_logit = outputs[next_token_idx].item()
        next_token_text = self.processor.decode(next_token_idx)

        incorrect_token_ids, correct_token_ids = self.get_answer_token_ids(example)
        correct_token_ids_list = list(correct_token_ids)
        incorrect_token_ids_list = list(incorrect_token_ids)

        max_correct = outputs[correct_token_ids_list].max().item()
        max_incorrect = outputs[incorrect_token_ids_list].max().item()

        results = {
            "model_size": self.model_size.value,
            "example_id": example.example_id,
            "grounding_mode": grounding_mode.value,
            "image_variant": image_variant.value,
            "next_token_id": next_token_idx,
            "next_token_text": next_token_text,
            "next_token_logit": next_token_logit,
            "correct_token_ids": correct_token_ids_list,
            "correct_max_logit": max_correct,
            "incorrect_token_id": incorrect_token_ids_list,
            "incorrect_max_logit": max_incorrect,
        }
        with Path.open(f_out, "w") as f:
            json.dump(results, f, indent=4)

        del inputs, outputs
        return None

    def get_output_file(
        self,
        example: DatasetExampleColor,
        grounding_mode: GroundingMode,
        image_variant: ImageVariant,
    ):
        f_out = (
            self.get_output_dir()
            / example.example_id
            / f"{grounding_mode.value}_{image_variant.value}.json"
        )
        f_out.parent.mkdir(parents=True, exist_ok=True)
        return f_out
