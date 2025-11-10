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
    magnitudes = cms.maybe_update(inputs=inputs, outputs=outputs)
    assert isinstance(magnitudes, dict)
