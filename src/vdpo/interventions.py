"""Lightweight, hook-based interventions for tutorials and quick experiments.

`head_knockout` reproduces the last-token attention-head ablation used by
`KnockoutAttnHeads`, but as a plain forward-hook context manager so it also
works inside `model.generate()` without nnsight.
"""

from collections import defaultdict
from contextlib import contextmanager


@contextmanager
def head_knockout(model, adapter, heads):
    """Zero the o_proj-input contribution of attention heads at the last token.

    Args:
        model: the loaded VLM (a transformers ``PreTrainedModel``).
        adapter: the ``VLMAdapter`` for this model family.
        heads: iterable of ``(layer_idx, head_idx)`` tuples to ablate.

    For a single prompt forward the last position is the final prompt token, so
    this matches the slice ``KnockoutAttnHeads`` zeros. During generation with a
    KV cache each step's o_proj input holds only the new token, so the last
    position is the token being predicted and the ablation persists across the
    whole generation. Zeroing targets the last position of every row, so for a
    batch larger than 1 the final token of each sequence is ablated; the tutorial
    runs at batch size 1. All hooks are removed on exit.
    """
    text_config = adapter.get_text_config(model)
    num_heads = text_config.num_attention_heads
    head_dim = getattr(text_config, "head_dim", text_config.hidden_size // num_heads)

    heads_by_layer: dict[int, list[int]] = defaultdict(list)
    for layer_idx, head_idx in heads:
        heads_by_layer[layer_idx].append(head_idx)

    layers = adapter.get_language_layer_modules(model)

    def make_pre_hook(head_idxs):
        def pre_hook(module, args, kwargs):
            hidden = args[0].clone()
            for head_idx in head_idxs:
                start = head_idx * head_dim
                end = start + head_dim
                hidden[..., -1, start:end] = 0
            return (hidden, *args[1:]), kwargs

        return pre_hook

    handles = []
    try:
        for layer_idx, head_idxs in heads_by_layer.items():
            o_proj = adapter.get_o_proj_module(layers[layer_idx])
            handles.append(
                o_proj.register_forward_pre_hook(
                    make_pre_hook(head_idxs), with_kwargs=True
                )
            )
        yield
    finally:
        for handle in handles:
            handle.remove()
