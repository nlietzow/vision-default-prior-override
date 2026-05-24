import argparse
import gc

import torch
from tqdm.auto import tqdm

from vdpo.runner.extract_attention_weights import ExtractAttentionWeights
from vdpo.types.enums import ModelFamily, ModelSize
from vdpo.utils.load_examples import load_color_examples


# noinspection DuplicatedCode
def main(model_size: ModelSize, model_family: ModelFamily):
    runner = ExtractAttentionWeights(model_size=model_size, model_family=model_family)
    examples = load_color_examples()
    for example in tqdm(examples, desc="Processing examples", unit="example"):
        runner.process_example(example)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-size",
        type=ModelSize,
        required=True,
        help="Size of the model to use for attention weight extraction.",
    )
    parser.add_argument(
        "--model-family",
        type=ModelFamily,
        default=ModelFamily.QWEN,
        choices=list(ModelFamily),
        help="Model family to use (default: qwen).",
    )
    args = parser.parse_args()
    main(model_size=args.model_size, model_family=args.model_family)
