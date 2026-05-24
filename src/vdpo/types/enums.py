from enum import StrEnum


class ModelSize(StrEnum):
    SMALL_3B = "3B"
    MEDIUM_7B = "7B"
    MEDIUM_10B = "10B"
    LARGE_32B = "32B"
    X_LARGE_72B = "72B"


class DatasetSplitName(StrEnum):
    COLOR = "color"
    SIZE = "size"


class GroundingMode(StrEnum):
    VISUAL = "visual"
    PRIOR = "prior"


class ImageVariant(StrEnum):
    ORIGINAL = "original"
    COUNTERFACTUAL = "counterfactual"


class ModelFamily(StrEnum):
    QWEN = "qwen"
    LLAVA_NEXT = "llava_next"
    PALIGEMMA_2 = "paligemma_2"
