import argparse
import gc
from itertools import product

import torch
from tqdm.auto import tqdm

from vdpo.runner.inference import InferenceRunner
from vdpo.types.enums import GroundingMode, ImageVariant, ModelFamily, ModelSize
from vdpo.utils.load_examples import load_color_examples


def main(model_size: ModelSize, model_family: ModelFamily):
    inference_model = InferenceRunner(model_size=model_size, model_family=model_family)
    examples = load_color_examples()
    for example in tqdm(examples, desc="Processing examples", unit="example"):
        for grounding_mode, image_variant in product(GroundingMode, ImageVariant):
            inference_model.process_example(
                example,
                grounding_mode=GroundingMode(grounding_mode),
                image_variant=ImageVariant(image_variant),
            )

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
        choices=list(ModelSize),
        help="Size of the model to use for inference.",
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
