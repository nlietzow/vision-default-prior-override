from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin

    from vdpo.types.enums import ModelFamily, ModelSize
    from vdpo.vlm.base import VLMAdapter


class ModelWithProcessor(NamedTuple):
    model: PreTrainedModel
    processor: ProcessorMixin
    model_size: ModelSize
    model_family: ModelFamily
    adapter: VLMAdapter
