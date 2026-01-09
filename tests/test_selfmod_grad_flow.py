import torch
import torch.nn.functional as F

from nested_learning.levels import LevelSpec
from nested_learning.model import HOPEModel, ModelConfig


def test_hope_selfmod_forward_allows_outer_gradients() -> None:
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=32,
        dim=16,
        num_layers=1,
        heads=4,
        titan_level=LevelSpec(name="titan", update_period=1),
        cms_levels=(),
        block_variant="hope_selfmod",
    )
    model = HOPEModel(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, 6))
    logits = model(tokens)
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, cfg.vocab_size),
        tokens[:, 1:].reshape(-1),
    )
    loss.backward()
    block = model.blocks[0]
    selfmod = getattr(block, "selfmod", None)
    assert selfmod is not None
    grad = selfmod.m_memory.w1.weight.grad
    assert grad is not None
    assert grad.abs().sum().item() > 0.0

