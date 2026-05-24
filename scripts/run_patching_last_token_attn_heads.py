import argparse
import gc

import torch
from tqdm.auto import tqdm

from vdpo.runner.patching_last_token_attn_heads import PatchingLastTokenAttnHeads
from vdpo.types.contrast import ContrastSpec
from vdpo.types.enums import ModelFamily, ModelSize
from vdpo.utils.load_examples import load_color_examples


# noinspection DuplicatedCode
def main(model_size: ModelSize, model_family: ModelFamily, contrast: ContrastSpec):
    patching_model = PatchingLastTokenAttnHeads(
        model_size=model_size,
        model_family=model_family,
        contrast=contrast,
    )
    examples = load_color_examples()
    for example in tqdm(examples, desc="Processing examples", unit="example"):
        patching_model.process_example(example)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", type=ModelSize, required=True)
    parser.add_argument(
        "--model-family",
        type=ModelFamily,
        default=ModelFamily.QWEN,
        choices=list(ModelFamily),
    )
    parser.add_argument(
        "--contrast",
        type=str,
        default="prior_circuit",
        choices=["prior_circuit", "visual_circuit"],
        help="Which patching contrast to run.",
    )
    args = parser.parse_args()
    main(
        model_size=args.model_size,
        model_family=args.model_family,
        contrast=ContrastSpec.from_name(args.contrast),
    )
