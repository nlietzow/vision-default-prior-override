from types import SimpleNamespace

import torch
import torch.nn as nn

from vdpo.interventions import head_knockout
from vdpo.vlm.qwen import QwenAdapter


def test_get_o_proj_module_returns_self_attn_o_proj():
    layer = SimpleNamespace(self_attn=SimpleNamespace(o_proj="OPROJ"))
    # get_o_proj_module is concrete on the base, so any family inherits it
    assert QwenAdapter().get_o_proj_module(layer) == "OPROJ"


class _RecordingLinear(nn.Linear):
    def forward(self, x):
        self.last_input = x.detach().clone()
        return super().forward(x)


class _Attn(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.o_proj = _RecordingLinear(hidden, hidden, bias=False)


class _Layer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.self_attn = _Attn(hidden)


class _Model(nn.Module):
    def __init__(self, n_layers, hidden):
        super().__init__()
        self.layers = nn.ModuleList(_Layer(hidden) for _ in range(n_layers))


class _Config:
    def __init__(self, num_heads, head_dim):
        self.num_attention_heads = num_heads
        self.head_dim = head_dim
        self.hidden_size = num_heads * head_dim


class _Adapter:
    def __init__(self, config):
        self._config = config

    def get_text_config(self, model):
        return self._config

    def get_language_layer_modules(self, model):
        return model.layers

    def get_o_proj_module(self, layer):
        return layer.self_attn.o_proj


def _build():
    num_heads, head_dim = 4, 2
    hidden = num_heads * head_dim  # 8
    model = _Model(n_layers=3, hidden=hidden)
    adapter = _Adapter(_Config(num_heads, head_dim))
    return model, adapter, hidden


def test_head_knockout_zeros_target_head_slice_at_last_token_only():
    model, adapter, hidden = _build()
    o_proj = model.layers[1].self_attn.o_proj
    x = torch.arange(2 * hidden, dtype=torch.float32).reshape(1, 2, hidden) + 1.0

    # knock out head index 1 (slice [2:4]) in layer 1
    with head_knockout(model, adapter, [(1, 1)]):
        o_proj(x)

    seen = o_proj.last_input
    assert torch.all(seen[0, -1, 2:4] == 0)  # target head, last token: zeroed
    assert torch.all(seen[0, -1, 0:2] == x[0, -1, 0:2])  # other head untouched
    assert torch.all(seen[0, -1, 4:8] == x[0, -1, 4:8])  # other heads untouched
    assert torch.all(seen[0, 0, :] == x[0, 0, :])  # earlier token untouched
    # the input the caller still holds is not mutated in place
    assert torch.all(x[0, -1, 2:4] != 0)


def test_head_knockout_only_hooks_listed_layers():
    model, adapter, hidden = _build()
    untouched = model.layers[0].self_attn.o_proj
    x = torch.ones(1, 2, hidden)
    with head_knockout(model, adapter, [(1, 0)]):
        untouched(x)
    assert torch.all(untouched.last_input == x)


def test_head_knockout_removes_all_hooks_on_exit():
    model, adapter, _ = _build()
    with head_knockout(model, adapter, [(0, 0), (1, 1)]):
        # hooks are registered on exactly the listed layers while active
        assert len(model.layers[0].self_attn.o_proj._forward_pre_hooks) == 1
        assert len(model.layers[1].self_attn.o_proj._forward_pre_hooks) == 1
        assert len(model.layers[2].self_attn.o_proj._forward_pre_hooks) == 0
    for layer in model.layers:
        assert len(layer.self_attn.o_proj._forward_pre_hooks) == 0
