import torch

from nested_learning.fast_state import build_block_fast_state
from nested_learning.hope.block import HOPESelfModBlock, HOPESelfModBlockConfig
from nested_learning.levels import LevelSpec


def test_selfmod_updates_without_teach_signal() -> None:
    cfg = HOPESelfModBlockConfig(
        dim=8,
        cms_levels=[LevelSpec(name="fast", update_period=1)],
        optimizer_configs={},
        selfmod_online_updates=True,
    )
    block = HOPESelfModBlock(cfg)
    fast_state = build_block_fast_state(
        titan_module=None,
        cms_blocks=block.cms.blocks,
        selfmod_module=block.selfmod,
        specs=cfg.cms_levels,
        optimizer_configs={},
        default_lr=cfg.self_mod_lr,
    )
    assert fast_state.selfmod_state is not None
    x = torch.randn(1, 4, 8)
    before = fast_state.selfmod_state.memory.w1.clone()
    _ = block(x, fast_state=fast_state)
    after = fast_state.selfmod_state.memory.w1
    assert not torch.allclose(before, after)
