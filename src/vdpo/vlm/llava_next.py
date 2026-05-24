import torch
from nnsight import NNsight
from PIL import Image
from transformers import (
    BatchFeature,
    BitsAndBytesConfig,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    PreTrainedModel,
    ProcessorMixin,
)

from vdpo.types.enums import ModelFamily, ModelSize
from vdpo.vlm.base import VLMAdapter

MODEL_PATH = "llava-hf/llava-v1.6-mistral-7b-hf"


class LlavaNextAdapter(VLMAdapter):
    model_family = ModelFamily.LLAVA_NEXT

    def load_model(
        self,
        model_size: ModelSize,
        *,
        load_in_4bit: bool,
        device_map: str,
        attn_implementation: str | None,
    ) -> tuple[PreTrainedModel, ProcessorMixin]:
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

        model = LlavaNextForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=dtype,
            device_map=device_map,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            **extra_kwargs,
        ).eval()

        for param in model.parameters():
            param.requires_grad = False

        processor = LlavaNextProcessor.from_pretrained(MODEL_PATH, use_fast=True)

        return model, processor

    def build_inputs(
        self,
        processor: ProcessorMixin,
        image: Image.Image,
        question: str,
        device,
    ) -> BatchFeature:
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

        return processor(
            text=text,
            images=image,
            padding=False,
            return_tensors="pt",
        ).to(device)

    def get_language_layers(self, nnsight: NNsight):
        # LlavaNextForConditionalGeneration.model = LlavaNextModel
        # LlavaNextModel.language_model = MistralModel (has .layers)
        return nnsight.model.language_model.layers

    def get_lm_head_output_saved(self, nnsight: NNsight):
        return nnsight.lm_head.output[0, -1, :].save()

    def get_text_config(self, model: PreTrainedModel):
        return model.model.language_model.config

    def get_layer_output_last_token(self, layer):
        # Mistral layer.output[0] is 2D (seq, hidden) — no batch dim
        return layer.output[0][-1, :]

    def set_layer_output_last_token(self, layer, value):
        layer.output[0][-1, :] = value

    def get_o_proj_input_last_token(self, layer):
        # Mistral o_proj.input[0] is 2D (seq, hidden) — same as Qwen
        return layer.self_attn.o_proj.input[0][-1, :]

    def set_o_proj_input_slice(self, layer, start: int, end: int, value):
        layer.self_attn.o_proj.input[0][-1, start:end] = value

    def set_mlp_output_last_token(self, layer, value):
        # Mistral mlp.output is 3D (batch, seq, hidden)
        layer.mlp.output[0, -1, :] = value

    def get_mlp_output_last_token(self, layer):
        return layer.mlp.output[0, -1, :]

    def get_lm_head_module(self, model: PreTrainedModel):
        return model.lm_head

    def get_language_layer_modules(self, model: PreTrainedModel):
        return model.model.language_model.layers

    def get_image_token_id(self, processor: ProcessorMixin) -> int:
        return processor.image_token_id
