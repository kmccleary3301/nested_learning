import torch

from nested_learning.levels import LevelSpec
from nested_learning.model import HOPEModel, ModelConfig
from nested_learning.training import compute_teach_signal


def test_hope_selfmod_variant_updates_selfmod_state_in_fast_mode() -> None:
    cfg = ModelConfig(
        vocab_size=32,
        dim=16,
        num_layers=1,
        heads=4,
        titan_level=LevelSpec(name="titan", update_period=1),
        cms_levels=[LevelSpec(name="cms_fast", update_period=2)],
        block_variant="hope_selfmod",
    )
    model = HOPEModel(cfg)
    state = model.init_fast_state()
    assert state.blocks[0].selfmod_state is not None
    before = state.blocks[0].selfmod_state.memory.w2.detach().clone()

    tokens = torch.randint(0, cfg.vocab_size, (1, 6))
    with torch.no_grad():
        logits = model(tokens, fast_state=state)
        teach = compute_teach_signal(model, logits, tokens)
        _ = model(tokens, teach_signal=teach, fast_state=state)

    after = state.blocks[0].selfmod_state.memory.w2.detach().clone()
    assert not torch.allclose(before.unsqueeze(0), after)
