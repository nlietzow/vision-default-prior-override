from abc import ABC, abstractmethod

from nnsight import NNsight
from PIL import Image

from vdpo.settings import settings
from vdpo.types.enums import GroundingMode, ImageVariant, ModelFamily
from vdpo.types.models import DatasetExampleColor
from vdpo.utils.inflect_engine import get_inflect_engine
from vdpo.utils.load_model import ModelSize, load_vlm
from vdpo.utils.logger import setup_logger


class VMIRunnerBase(ABC):
    @property
    @abstractmethod
    def runner_name(self):
        pass

    def __init__(
        self,
        model_size: ModelSize,
        *,
        model_family: ModelFamily = ModelFamily.QWEN,
        **load_kwargs,
    ):
        self._model = load_vlm(
            model_size=model_size, model_family=model_family, **load_kwargs
        )
        self.nnsight = NNsight(self.model)
        self.adapter = self._model.adapter
        self.logger = setup_logger("vmi_runner")

    @property
    def model(self):
        return self._model.model

    @property
    def processor(self):
        return self._model.processor

    @property
    def model_size(self):
        return self._model.model_size

    @property
    def model_family(self):
        return self._model.model_family

    @abstractmethod
    def process_example(self, example: DatasetExampleColor):
        pass

    def build_inputs(
        self,
        example: DatasetExampleColor,
        grounding_mode: GroundingMode,
        image_variant: ImageVariant,
    ):
        image = self._get_image(example, image_variant)
        question = self._get_question(example, grounding_mode)
        return self.adapter.build_inputs(
            processor=self.processor,
            image=image,
            question=question,
            device=self.model.device,
        )

    def get_answer_token_ids(self, example: DatasetExampleColor):
        token_ids: list[set[int]] = []

        for ans in (example.incorrect_answer, *example.correct_answer):
            color = ans.strip()
            variants = {color.capitalize(), color.lower()}
            ids: set[int] = set()
            for variant in variants:
                first_token_id = (
                    self.processor.tokenizer(
                        text=variant,
                        add_special_tokens=False,
                        return_tensors="pt",
                    )
                    .input_ids[0, 0]
                    .item()
                )
                ids.add(first_token_id)
            token_ids.append(ids)

        incorrect_token_ids = token_ids.pop(0)
        correct_token_ids: set[int] = set()
        for ids in token_ids:
            correct_token_ids.update(ids)
        return incorrect_token_ids, correct_token_ids

    def get_language_layers(self):
        return self.adapter.get_language_layers(self.nnsight)

    def get_text_config(self):
        return self.adapter.get_text_config(self.model)

    @staticmethod
    def _get_question(example, grounding_mode: GroundingMode) -> str:
        object_name = example.object.lower()
        suffix = "Respond with the color name only."

        if grounding_mode == GroundingMode.VISUAL:
            return f"What color is this {object_name} here? {suffix}"
        if grounding_mode == GroundingMode.PRIOR:
            a_object_name = get_inflect_engine().a(object_name)
            return f"What color is {a_object_name} usually? {suffix}"

        raise ValueError(f"Unknown grounding mode: {grounding_mode}")

    @staticmethod
    def _get_image(example, image_variant: ImageVariant) -> Image.Image:
        if image_variant == ImageVariant.ORIGINAL:
            return example.original_image.pil_image
        if image_variant == ImageVariant.COUNTERFACTUAL:
            return example.counterfact_image.pil_image

        raise ValueError(f"Unknown image variant: {image_variant}")

    def get_output_dir(self, contrast_subdir: str | None = None):
        parts = [settings.output_dir, self.runner_name]
        if contrast_subdir is not None:
            parts.append(contrast_subdir)
        parts.extend([self.model_family.value, self.model_size.value])
        path = parts[0]
        for p in parts[1:]:
            path = path / p
        path.mkdir(parents=True, exist_ok=True)
        return path
