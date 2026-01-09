import torch

from nested_learning.levels import LevelSpec
from nested_learning.model import HOPEModel, ModelConfig
from nested_learning.training import compute_teach_signal


def _cms_delta_l1(state, level_name: str) -> float:
    params = state.blocks[0].cms_params[level_name]
    return float(sum(delta.abs().sum().item() for delta in params.values()))


def test_e2e_update_paths_and_surprise_gate() -> None:
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=32,
        dim=16,
        num_layers=1,
        heads=4,
        titan_level=LevelSpec(name="titan", update_period=1),
        cms_levels=(LevelSpec(name="cms_fast", update_period=2),),
        block_variant="hope_selfmod",
    )
    model = HOPEModel(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (1, 8))

    # Baseline: both selfmod and CMS should update in fast-state mode.
    state = model.init_fast_state()
    assert state.blocks[0].selfmod_state is not None
    cms_before = _cms_delta_l1(state, "cms_fast")
    assert cms_before == 0.0
    selfmod_before = state.blocks[0].selfmod_state.memory.w2.detach().clone()
    with torch.no_grad():
        logits_before = model(tokens, fast_state=state)
        teach = compute_teach_signal(model, logits_before, tokens)
        _ = model(tokens, teach_signal=teach, fast_state=state)
        logits_after = model(tokens, fast_state=state)
    cms_after = _cms_delta_l1(state, "cms_fast")
    selfmod_after = state.blocks[0].selfmod_state.memory.w2.detach().clone()
    assert cms_after > 0.0
    assert not torch.allclose(selfmod_before, selfmod_after)
    assert not torch.allclose(logits_before, logits_after)

    # Surprise gate: CMS updates should be blocked when threshold exceeds the computed surprise.
    gated_state = model.init_fast_state()
    with torch.no_grad():
        gated_logits = model(tokens, fast_state=gated_state)
        gated_teach = compute_teach_signal(model, gated_logits, tokens)
    surprise = float(gated_teach.norm(dim=-1).mean().item())
    model.set_surprise_threshold(surprise + 1.0)
    try:
        with torch.no_grad():
            _ = model(tokens, teach_signal=gated_teach, fast_state=gated_state)
        assert _cms_delta_l1(gated_state, "cms_fast") == 0.0
    finally:
        model.set_surprise_threshold(None)

