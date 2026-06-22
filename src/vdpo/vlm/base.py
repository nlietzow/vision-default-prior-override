from abc import ABC, abstractmethod

from nnsight import NNsight
from PIL import Image
from transformers import BatchFeature, PreTrainedModel, ProcessorMixin

from vdpo.types.enums import ModelFamily, ModelSize


class VLMAdapter(ABC):
    model_family: ModelFamily

    @abstractmethod
    def load_model(
        self,
        model_size: ModelSize,
        *,
        load_in_4bit: bool,
        device_map: str,
        attn_implementation: str | None,
    ) -> tuple[PreTrainedModel, ProcessorMixin]:
        """Load the model and processor. Returns (model, processor)."""

    @abstractmethod
    def build_inputs(
        self,
        processor: ProcessorMixin,
        image: Image.Image,
        question: str,
        device,
    ) -> BatchFeature:
        """Build model inputs from an image and question string."""

    @abstractmethod
    def get_language_layers(self, nnsight: NNsight):
        """Return the language model layer list (nnsight proxy)."""

    @abstractmethod
    def get_text_config(self, model: PreTrainedModel):
        """Return the text config with num_attention_heads, hidden_size, etc."""

    @abstractmethod
    def get_layer_output_last_token(self, layer):
        """Return layer residual stream output for the last token."""

    @abstractmethod
    def get_o_proj_input_last_token(self, layer):
        """Return o_proj.input for the last token (full hidden dim)."""

    @abstractmethod
    def set_layer_output_last_token(self, layer, value):
        """Patch the layer residual stream output at the last token."""

    @abstractmethod
    def set_o_proj_input_slice(self, layer, start: int, end: int, value):
        """Patch a slice of o_proj.input at the last token (for head patching)."""

    @abstractmethod
    def set_mlp_output_last_token(self, layer, value):
        """Patch the MLP output at the last token."""

    @abstractmethod
    def get_mlp_output_last_token(self, layer):
        """Return MLP output for the last token."""

    @abstractmethod
    def get_lm_head_output_saved(self, nnsight: NNsight):
        """Return lm_head.output[0, -1, :].save() inside a trace context."""

    @abstractmethod
    def get_lm_head_module(self, model: PreTrainedModel):
        """Return the raw lm_head nn.Module (for weight extraction)."""

    @abstractmethod
    def get_language_layer_modules(self, model: PreTrainedModel):
        """Return the raw language model layer list (for weight extraction)."""

    @abstractmethod
    def get_image_token_id(self, processor: ProcessorMixin) -> int:
        """Return the image placeholder token ID used by this model."""

    def get_o_proj_module(self, layer):
        """Return the attention output-projection module for a layer.

        Used by tutorial-style forward-hook interventions. All current
        families expose it at ``layer.self_attn.o_proj``; override if a
        future family differs.
        """
        return layer.self_attn.o_proj
