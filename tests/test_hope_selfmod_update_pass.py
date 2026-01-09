import torch

from nested_learning.levels import LevelSpec
from nested_learning.model import HOPEModel, ModelConfig
from nested_learning.training import compute_teach_signal


def test_hope_selfmod_updates_module_params_only_in_update_pass() -> None:
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=32,
        dim=16,
        num_layers=1,
        heads=4,
        titan_level=LevelSpec(name="titan", update_period=1),
        cms_levels=(),
        block_variant="hope_selfmod",
        self_mod_lr=1.0,
    )
    model = HOPEModel(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (1, 8))
    before = model.blocks[0].selfmod.m_memory.w2.weight.detach().clone()

    _ = model(tokens)
    after_forward = model.blocks[0].selfmod.m_memory.w2.weight.detach().clone()
    assert torch.allclose(before, after_forward, atol=1e-6, rtol=1e-6)

    with torch.no_grad():
        logits = model(tokens)
        teach = compute_teach_signal(model, logits, tokens)
        _ = model(tokens, teach_signal=teach)
    after_update = model.blocks[0].selfmod.m_memory.w2.weight.detach().clone()
    assert not torch.allclose(after_forward, after_update)

