import torch
from nnsight import NNsight
from PIL import Image
from transformers import (
    BatchFeature,
    BitsAndBytesConfig,
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    PreTrainedModel,
    ProcessorMixin,
)

from vdpo.types.enums import ModelFamily, ModelSize
from vdpo.vlm.base import VLMAdapter

MODEL_PATH_TEMPLATE = "google/paligemma2-{size}-mix-448"

# PaliGemma sizes don't match ModelSize values directly
_SIZE_MAP: dict[ModelSize, str] = {
    ModelSize.SMALL_3B: "3b",
    ModelSize.MEDIUM_10B: "10b",
}


class PaliGemma2Adapter(VLMAdapter):
    model_family = ModelFamily.PALIGEMMA_2

    def load_model(
        self,
        model_size: ModelSize,
        *,
        load_in_4bit: bool,
        device_map: str,
        attn_implementation: str | None,
    ) -> tuple[PreTrainedModel, ProcessorMixin]:
        if model_size not in _SIZE_MAP:
            raise ValueError(
                f"PaliGemma 2 does not support size {model_size.value}. "
                f"Supported: {', '.join(s.value for s in _SIZE_MAP)}"
            )
        model_path = MODEL_PATH_TEMPLATE.format(size=_SIZE_MAP[model_size])

        quantization_config = None
        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )

        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            dtype = None

        extra_kwargs = {}
        if attn_implementation is not None:
            extra_kwargs["attn_implementation"] = attn_implementation

        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device_map,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            **extra_kwargs,
        ).eval()

        for param in model.parameters():
            param.requires_grad = False

        processor = PaliGemmaProcessor.from_pretrained(model_path, use_fast=True)

        return model, processor

    def build_inputs(
        self,
        processor: ProcessorMixin,
        image: Image.Image,
        question: str,
        device,
    ) -> BatchFeature:
        # PaliGemma uses prefix LM — no chat template, just pass text directly
        # Ensure RGB — some dataset images are RGBA/grayscale
        image = image.convert("RGB")
        return processor(
            images=image,
            text=question,
            padding=False,
            return_tensors="pt",
        ).to(device)

    def get_language_layers(self, nnsight: NNsight):
        # PaliGemmaForConditionalGeneration.model = PaliGemmaModel
        # PaliGemmaModel.language_model = Gemma2Model (has .layers directly)
        return nnsight.model.language_model.layers

    def get_lm_head_output_saved(self, nnsight: NNsight):
        return nnsight.lm_head.output[0, -1, :].save()

    def get_text_config(self, model: PreTrainedModel):
        return model.model.language_model.config

    def get_layer_output_last_token(self, layer):
        # layer.output[0] is 2D (seq, hidden) — no batch dim
        return layer.output[0][-1, :]

    def set_layer_output_last_token(self, layer, value):
        layer.output[0][-1, :] = value

    def get_o_proj_input_last_token(self, layer):
        # o_proj.input[0] is 2D (seq, hidden) — no batch dim
        return layer.self_attn.o_proj.input[0][-1, :]

    def set_o_proj_input_slice(self, layer, start: int, end: int, value):
        layer.self_attn.o_proj.input[0][-1, start:end] = value

    def set_mlp_output_last_token(self, layer, value):
        # mlp.output is 3D (batch, seq, hidden) — MLP returns a single tensor,
        # unlike layer.output which is also a single tensor but accessed via [0]
        # (which already strips the batch dim). Same pattern as Qwen.
        layer.mlp.output[0, -1, :] = value

    def get_mlp_output_last_token(self, layer):
        # mlp.output is 3D (batch, seq, hidden) — same as Qwen
        return layer.mlp.output[0, -1, :]

    def get_lm_head_module(self, model: PreTrainedModel):
        return model.lm_head

    def get_language_layer_modules(self, model: PreTrainedModel):
        return model.model.language_model.layers

    def get_image_token_id(self, processor: ProcessorMixin) -> int:
        return processor.image_token_id
