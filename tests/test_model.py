import torch

from nested_learning.levels import LevelSpec
from nested_learning.model import HOPEModel, ModelConfig


def test_model_forward() -> None:
    config = ModelConfig(
        vocab_size=100,
        dim=32,
        num_layers=1,
        heads=4,
        titan_level=LevelSpec(name="titan", update_period=2),
        cms_levels=[LevelSpec(name="fast", update_period=1)],
    )
    model = HOPEModel(config)
    tokens = torch.randint(0, 100, (2, 10))
    logits = model(tokens)
    assert logits.shape == (2, 10, 100)
