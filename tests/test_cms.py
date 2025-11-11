import torch

from nested_learning.cms import CMS
from nested_learning.levels import LevelSpec


def test_cms_forward_and_update() -> None:
    cms = CMS(
        dim=16,
        levels=[LevelSpec(name="fast", update_period=1), LevelSpec(name="slow", update_period=2)],
    )
    x = torch.randn(2, 4, 16)
    out, inputs, outputs = cms(x, return_intermediates=True)
    assert out.shape == x.shape
    chunks = cms.accumulate_chunks(inputs=inputs, outputs=outputs)
    assert "fast" in chunks
    assert "slow" not in chunks


def test_cms_chunk_accumulation_respects_period() -> None:
    cms = CMS(
        dim=8,
        levels=[LevelSpec(name="fast", update_period=1), LevelSpec(name="slow", update_period=3)],
    )
    ready_counts = {"fast": 0, "slow": 0}
    for step in range(5):
        x = torch.randn(2, 3, 8)
        _, inputs, outputs = cms(x, return_intermediates=True)
        chunks = cms.accumulate_chunks(inputs=inputs, outputs=outputs)
        for level in chunks:
            ready_counts[level] += 1
        if step < 2:
            assert "slow" not in chunks
    assert ready_counts["fast"] == 5
    assert ready_counts["slow"] == 1
