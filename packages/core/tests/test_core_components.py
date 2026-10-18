import numpy as np
import pytest
import torch
import torch.nn as nn
from diffusion_omni.core.codecs import TextCodec
from diffusion_omni.core.config import (
    SamplingConfig,
    apply_overrides,
    from_dict,
    load_config,
)
from diffusion_omni.core.nn.adapters import ImagePatchAdapter, VideoPatchAdapter
from diffusion_omni.core.nn.lora import apply_lora, lora_disabled, merged_state_dict
from diffusion_omni.core.registry import Registry
from diffusion_omni.core.utils import to_grid, write_png


def test_registry_build_context_and_dotted_paths():
    reg = Registry("thing")

    @reg.register("pair")
    class Pair:
        def __init__(self, a, b=2):
            self.a, self.b = a, b

    p = reg.build({"type": "pair", "a": 1}, b=5, unrelated=0)
    assert (p.a, p.b) == (1, 5)
    assert reg.build({"type": "pair", "a": 1, "b": 3}, b=5).b == 3  # spec wins over context
    assert isinstance(reg.build({"type": "torch.nn:Identity"}), nn.Identity)
    with pytest.raises(KeyError, match="available"):
        reg.get("missing")


def test_config_overrides_and_floats(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("a: {lr: 1e-4}\n")
    cfg = load_config(p, ["a.lr=3e-4", "b.c=[1, 2]", "d=null"])
    assert cfg == {"a": {"lr": 3e-4}, "b": {"c": [1, 2]}, "d": None}
    assert apply_overrides({}, ["x.y=hello"]) == {"x": {"y": "hello"}}
    with pytest.raises(KeyError, match="unknown keys"):
        from_dict(SamplingConfig, {"not_a_field": 1})


@pytest.mark.parametrize("adapter,shape", [
    (ImagePatchAdapter(16, (3, 8, 12), patch=2), (2, 3, 8, 12)),
    (VideoPatchAdapter(16, (3, 4, 8, 8), patch=4, temporal_patch=2), (2, 3, 4, 8, 8)),
])
def test_patchify_roundtrip(adapter, shape):
    # With identity projections, unembed(embed(x)) must be x.
    n = adapter.proj.in_features
    adapter.proj = nn.Identity()
    adapter.head = nn.Identity()
    adapter.pos = torch.zeros(1, 1, n)
    x = torch.randn(shape)
    torch.testing.assert_close(adapter.unembed(adapter.embed(x), x), x)


def test_lora_is_noop_at_init_and_merges_exactly():
    torch.manual_seed(0)
    model = nn.Module()
    model.qkv, model.fc2 = nn.Linear(8, 8), nn.Linear(8, 4)
    model.forward = lambda x: model.fc2(model.qkv(x).relu())
    x = torch.randn(5, 8)
    before = model(x)
    assert apply_lora(model, rank=2, targets=("qkv", "fc2")) == 2
    torch.testing.assert_close(model(x), before)
    nn.init.normal_(model.qkv.lora_b.weight)
    after = model(x)
    assert not torch.allclose(after, before)
    with lora_disabled(model):
        torch.testing.assert_close(model(x), before)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"qkv.lora_a.weight", "qkv.lora_b.weight", "fc2.lora_a.weight", "fc2.lora_b.weight"}
    plain = nn.Module()
    plain.qkv, plain.fc2 = nn.Linear(8, 8), nn.Linear(8, 4)
    plain.load_state_dict(merged_state_dict(model))
    torch.testing.assert_close(plain.fc2(plain.qkv(x).relu()), after)


def test_text_codec_roundtrip():
    codec = TextCodec((12,))
    x = codec.encode(["héllo", "a much longer line than fits"])
    assert x.shape == (2, 12)
    assert codec.decode(x) == ["héllo", "a much longe"]


def test_png_writer(tmp_path):
    img = to_grid(np.random.rand(3, 3, 2, 4, 4))  # 3 RGB videos, 2 frames
    write_png(tmp_path / "x.png", img, scale=2)
    assert (tmp_path / "x.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
