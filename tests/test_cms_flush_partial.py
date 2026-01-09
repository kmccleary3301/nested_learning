import torch

from nested_learning.fast_state import build_block_fast_state
from nested_learning.hope.block import HOPEAttentionBlock, HOPEAttentionBlockConfig
from nested_learning.levels import LevelSpec


def _run_block(*, flush_partial: bool, use_fast_state: bool) -> dict[str, float]:
    torch.manual_seed(0)
    cfg = HOPEAttentionBlockConfig(
        dim=8,
        heads=1,
        cms_levels=(LevelSpec(name="fast", update_period=4),),
        cms_flush_partial_at_end=flush_partial,
        cms_online_updates=True,
        cms_chunk_reduction="sum",
    )
    block = HOPEAttentionBlock(cfg)
    x = torch.randn(1, 6, 8)
    teach = torch.ones(1, 6, 8)
    fast_state = None
    if use_fast_state:
        fast_state = build_block_fast_state(
            titan_module=None,
            cms_blocks=dict(block.cms.blocks.items()),
            specs=cfg.cms_levels,
            optimizer_configs=cfg.optimizer_configs,
            default_lr=cfg.self_mod_lr,
        )
    _out = block(x, teach_signal=teach, fast_state=fast_state)
    stats = block.pop_update_stats()
    return stats["cms.fast"]


def test_cms_flush_partial_disabled_leaves_remainder_unupdated() -> None:
    for use_fast_state in (False, True):
        payload = _run_block(flush_partial=False, use_fast_state=use_fast_state)
        assert payload["gate_hit"] == 1.0
        assert payload["chunk_tokens"] == 4.0


def test_cms_flush_partial_enabled_updates_final_remainder() -> None:
    for use_fast_state in (False, True):
        payload = _run_block(flush_partial=True, use_fast_state=use_fast_state)
        assert payload["gate_hit"] == 2.0
        assert payload["chunk_tokens"] == 6.0

