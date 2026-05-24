import contextlib

import datasets

from vdpo.settings import settings
from vdpo.types.enums import DatasetSplitName
from vdpo.types.models import (
    DatasetExample,
    DatasetExampleColor,
    DatasetExampleSize,
)
from vdpo.utils.logger import logger


def load_color_examples(
    samples: int = -1,
) -> list[DatasetExampleColor]:
    examples = []
    for example in load_examples(DatasetSplitName.COLOR, samples=samples):
        if isinstance(example, DatasetExampleColor):
            examples.append(example)
            continue
        raise ValueError(f"Expected DatasetExampleColor, got {type(example)}")
    return examples


def load_examples(
    split: DatasetSplitName,
    *,
    samples: int = -1,
) -> list[DatasetExample]:
    """
    Load a dataset from the Hugging Face datasets library and convert it to a pandas DataFrame.
    """

    def generator():
        if split == DatasetSplitName.COLOR:
            model_class = DatasetExampleColor
        elif split == DatasetSplitName.SIZE:
            model_class = DatasetExampleSize
        else:
            raise ValueError(f"Unknown split '{split}'")

        df = datasets.load_dataset(settings.dataset_path)[split].to_pandas()
        for _, row in df.iterrows():
            with contextlib.suppress(ValueError):
                yield model_class.model_validate(row.to_dict())

    examples = sorted(generator(), key=lambda x: x.example_id)

    if samples > 0:
        examples = examples[:samples]

    logger.info(
        f"Loaded {len(examples)} examples from dataset '{settings.dataset_path}' split '{split}'"
    )

    return examples


if __name__ == "__main__":
    load_examples(DatasetSplitName.COLOR, samples=-1)
    load_examples(DatasetSplitName.SIZE, samples=-1)
