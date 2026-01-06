import torch

from nested_learning.cms import CMS
from nested_learning.hope.block import HOPEAttentionBlock, HOPEAttentionBlockConfig
from nested_learning.levels import LevelSpec


def test_cms_forward_preserves_shape() -> None:
    cms = CMS(
        dim=16,
        levels=[LevelSpec(name="fast", update_period=2), LevelSpec(name="slow", update_period=4)],
    )
    x = torch.randn(2, 9, 16)
    out, inputs, outputs = cms(x, return_intermediates=True)
    assert out.shape == x.shape
    assert set(inputs.keys()) == {"fast", "slow"}
    assert set(outputs.keys()) == {"fast", "slow"}


def test_cms_updates_respect_update_period_tokens() -> None:
    cfg = HOPEAttentionBlockConfig(
        dim=16,
        heads=4,
        cms_levels=[
            LevelSpec(name="fast", update_period=2),
            LevelSpec(name="slow", update_period=4),
        ],
        optimizer_configs={},
    )
    block = HOPEAttentionBlock(cfg)
    x = torch.randn(1, 9, 16)
    teach = torch.randn_like(x)
    _ = block(x, teach_signal=teach)
    stats = block.pop_update_stats()
    assert stats["cms.fast"]["gate_hit"] == 5.0
    assert stats["cms.fast"]["chunk_tokens"] == 9.0
    assert stats["cms.slow"]["gate_hit"] == 3.0
    assert stats["cms.slow"]["chunk_tokens"] == 9.0


def test_cms_updates_skip_when_no_signal() -> None:
    cfg = HOPEAttentionBlockConfig(
        dim=16,
        heads=4,
        cms_levels=[LevelSpec(name="fast", update_period=2)],
        optimizer_configs={},
    )
    block = HOPEAttentionBlock(cfg)
    x = torch.randn(1, 8, 16)
    teach = torch.zeros_like(x)
    _ = block(x, teach_signal=teach)
    stats = block.pop_update_stats()
    assert stats == {}
