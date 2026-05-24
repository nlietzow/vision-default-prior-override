import torch
from nnsight import NNsight
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import (
    BatchFeature,
    BitsAndBytesConfig,
    PreTrainedModel,
    ProcessorMixin,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
)

from vdpo.types.enums import ModelFamily, ModelSize
from vdpo.vlm.base import VLMAdapter

MODEL_PATH_TEMPLATE = "Qwen/Qwen2.5-VL-{size}-Instruct"


class QwenAdapter(VLMAdapter):
    model_family = ModelFamily.QWEN

    def load_model(
        self,
        model_size: ModelSize,
        *,
        load_in_4bit: bool,
        device_map: str,
        attn_implementation: str | None,
    ) -> tuple[PreTrainedModel, ProcessorMixin]:
        model_path = MODEL_PATH_TEMPLATE.format(size=model_size.value)

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

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=device_map,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            **extra_kwargs,
        ).eval()

        for param in model.parameters():
            param.requires_grad = False

        processor = Qwen2_5_VLProcessor.from_pretrained(
            model_path,
            use_fast=True,
            min_pixels=1280 * 28 * 28,
            max_pixels=1280 * 28 * 28,
        )

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
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, *_ = process_vision_info(messages)

        return processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors="pt",
        ).to(device)

    def get_language_layers(self, nnsight: NNsight):
        return nnsight.model.language_model.layers

    def get_lm_head_output_saved(self, nnsight: NNsight):
        return nnsight.lm_head.output[0, -1, :].save()

    def get_text_config(self, model: PreTrainedModel):
        return model.model.language_model.config

    def get_layer_output_last_token(self, layer):
        # layer.output[0] is 3D (batch, seq, hidden) — Qwen
        return layer.output[0][0, -1, :]

    def set_layer_output_last_token(self, layer, value):
        layer.output[0][0, -1, :] = value

    def get_o_proj_input_last_token(self, layer):
        # o_proj.input[0] is 2D (seq, hidden) — no batch dim in Qwen
        return layer.self_attn.o_proj.input[0][-1, :]

    def set_o_proj_input_slice(self, layer, start: int, end: int, value):
        layer.self_attn.o_proj.input[0][-1, start:end] = value

    def set_mlp_output_last_token(self, layer, value):
        layer.mlp.output[0, -1, :] = value

    def get_mlp_output_last_token(self, layer):
        # layer.mlp.output is 3D (batch, seq, hidden) — Qwen
        return layer.mlp.output[0, -1, :]

    def get_lm_head_module(self, model: PreTrainedModel):
        return model.lm_head

    def get_language_layer_modules(self, model: PreTrainedModel):
        return model.model.language_model.layers

    def get_image_token_id(self, processor: ProcessorMixin) -> int:
        return processor.image_token_id
