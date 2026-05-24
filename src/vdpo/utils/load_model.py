from vdpo.settings import settings
from vdpo.types.core import ModelWithProcessor
from vdpo.types.enums import ModelFamily, ModelSize
from vdpo.utils.logger import logger
from vdpo.vlm.base import VLMAdapter
from vdpo.vlm.llava_next import LlavaNextAdapter
from vdpo.vlm.paligemma_2 import PaliGemma2Adapter
from vdpo.vlm.qwen import QwenAdapter

_ADAPTERS: dict[ModelFamily, VLMAdapter] = {
    ModelFamily.QWEN: QwenAdapter(),
    ModelFamily.LLAVA_NEXT: LlavaNextAdapter(),
    ModelFamily.PALIGEMMA_2: PaliGemma2Adapter(),
}


def get_adapter(model_family: ModelFamily) -> VLMAdapter:
    return _ADAPTERS[model_family]


def load_vlm(
    model_size: ModelSize,
    *,
    model_family: ModelFamily = ModelFamily.QWEN,
    load_in_4bit: bool | None = None,
    device_map: str = "auto",
    attn_implementation: str | None = None,
) -> ModelWithProcessor:
    """Load a VLM model and its processor via the appropriate adapter."""
    if load_in_4bit is None:
        load_in_4bit = settings.load_in_4bit

    adapter = get_adapter(model_family)
    model, processor = adapter.load_model(
        model_size,
        load_in_4bit=load_in_4bit,
        device_map=device_map,
        attn_implementation=attn_implementation,
    )

    logger.info(
        f"Loaded {model_family.value} model (size={model_size.value}) "
        f"with 4-bit: {load_in_4bit}"
    )

    return ModelWithProcessor(
        model=model,
        processor=processor,
        model_size=model_size,
        model_family=model_family,
        adapter=adapter,
    )
