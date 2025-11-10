import torch

from nested_learning.hope.block import HOPEBlock, HOPEBlockConfig
from nested_learning.levels import LevelSpec


def make_block() -> HOPEBlock:
    config = HOPEBlockConfig(
        dim=32,
        heads=4,
        titan_level=LevelSpec(name="titan", update_period=2),
        cms_levels=[LevelSpec(name="fast", update_period=1)],
    )
    return HOPEBlock(config)


def test_hope_block_forward() -> None:
    block = make_block()
    tokens = torch.randn(2, 8, 32)
    out = block(tokens)
    assert out.shape == tokens.shape


def test_hope_block_self_mod() -> None:
    block = make_block()
    tokens = torch.randn(2, 8, 32)
    teach = torch.randn_like(tokens)
    out = block(tokens, teach_signal=teach)
    assert out.shape == tokens.shape
