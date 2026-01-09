import torch
import torch.nn.functional as F

from nested_learning.levels import LevelSpec
from nested_learning.model import HOPEModel, ModelConfig


def test_hope_selfmod_fast_state_preserves_meta_forward_at_init() -> None:
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=64,
        dim=16,
        num_layers=1,
        heads=2,
        titan_level=LevelSpec(name="titan", update_period=1),
        cms_levels=(),
        block_variant="hope_selfmod",
    )
    model = HOPEModel(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (1, 8))
    fast_state = model.init_fast_state()
    with torch.no_grad():
        logits_meta = model(tokens)
        logits_fast = model(tokens, fast_state=fast_state)
    assert torch.allclose(logits_meta, logits_fast, atol=1e-6)


def test_hope_selfmod_fast_state_preserves_outer_grads_for_meta_memory_init() -> None:
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=64,
        dim=16,
        num_layers=1,
        heads=2,
        titan_level=LevelSpec(name="titan", update_period=1),
        cms_levels=(),
        block_variant="hope_selfmod",
    )
    model = HOPEModel(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (1, 8))
    fast_state = model.init_fast_state()
    logits = model(tokens, fast_state=fast_state)
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        tokens[:, 1:].reshape(-1),
    )
    loss.backward()
    block = model.blocks[0]
    selfmod = getattr(block, "selfmod", None)
    assert selfmod is not None
    grad = selfmod.m_memory.w1.weight.grad
    assert grad is not None
    assert grad.abs().sum().item() > 0.0

