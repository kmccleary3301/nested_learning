import torch

from nested_learning.levels import LevelSpec
from nested_learning.memorize import MemorizeConfig, memorize_tokens, snapshot_state_dict
from nested_learning.model import HOPEModel, ModelConfig


def _tiny_model() -> HOPEModel:
    titan = LevelSpec(name="titan", update_period=2, optimizer_key="titan_opt")
    cms = [
        LevelSpec(name="cms_fast", update_period=1, optimizer_key="cms_opt"),
        LevelSpec(name="cms_mid", update_period=2, optimizer_key="cms_opt"),
    ]
    cfg = ModelConfig(
        vocab_size=32,
        dim=16,
        num_layers=1,
        heads=4,
        titan_level=titan,
        cms_levels=cms,
        optimizers=None,
        teach_scale=0.1,
    )
    return HOPEModel(cfg)


def _tiny_model_update_every_call() -> HOPEModel:
    titan = LevelSpec(name="titan", update_period=1, optimizer_key="titan_opt")
    cms = [
        LevelSpec(name="cms_fast", update_period=1, optimizer_key="cms_opt"),
        LevelSpec(name="cms_mid", update_period=1, optimizer_key="cms_opt"),
    ]
    cfg = ModelConfig(
        vocab_size=32,
        dim=16,
        num_layers=1,
        heads=4,
        titan_level=titan,
        cms_levels=cms,
        optimizers=None,
        teach_scale=0.1,
    )
    return HOPEModel(cfg)


def _tiny_model_with_self_mod_lr(lr: float) -> HOPEModel:
    titan = LevelSpec(name="titan", update_period=1, optimizer_key="titan_opt")
    cfg = ModelConfig(
        vocab_size=32,
        dim=16,
        num_layers=1,
        heads=4,
        titan_level=titan,
        cms_levels=(),
        optimizers=None,
        teach_scale=0.1,
        self_mod_lr=lr,
    )
    return HOPEModel(cfg)


def _fast_titan_delta_norm(fast_state, before: dict[str, torch.Tensor]) -> float:
    block_state = fast_state.blocks[0]
    if block_state.titan_params is None:
        return 0.0
    total = 0.0
    for name, value in block_state.titan_params.items():
        total += (value.cpu() - before[name]).norm().item()
    return total


def test_memorize_fast_state_does_not_mutate_meta_params() -> None:
    torch.manual_seed(0)
    model = _tiny_model()
    tokens = torch.randint(0, model.config.vocab_size, (1, 8))
    baseline = snapshot_state_dict(model)
    fast_state = model.init_fast_state()
    cfg = MemorizeConfig(enabled=True, steps=2, use_fast_state=True)
    memorize_tokens(model, tokens, cfg, fast_state=fast_state)
    for name, param in model.state_dict().items():
        assert torch.allclose(baseline[name], param.cpu(), atol=1e-6)


def test_memorize_fast_state_changes_outputs_and_resets() -> None:
    torch.manual_seed(0)
    model = _tiny_model()
    tokens = torch.randint(0, model.config.vocab_size, (1, 8))
    fast_state = model.init_fast_state()
    with torch.no_grad():
        logits_before = model(tokens, fast_state=fast_state).detach().clone()

    cfg = MemorizeConfig(enabled=True, steps=1, use_fast_state=True)
    memorize_tokens(model, tokens, cfg, fast_state=fast_state)
    with torch.no_grad():
        logits_after = model(tokens, fast_state=fast_state).detach().clone()

    assert not torch.allclose(logits_before, logits_after)

    reset_state = model.init_fast_state()
    with torch.no_grad():
        logits_reset = model(tokens, fast_state=reset_state).detach().clone()
    assert torch.allclose(logits_before, logits_reset, atol=1e-6)


def test_memorize_respects_surprise_threshold() -> None:
    model = _tiny_model()
    tokens = torch.randint(0, model.config.vocab_size, (1, 8))
    fast_state = model.init_fast_state()
    block_state = fast_state.blocks[0]
    titan_before = {k: v.cpu().clone() for k, v in block_state.titan_params.items()}  # type: ignore[union-attr]
    cfg = MemorizeConfig(enabled=True, steps=1, surprise_threshold=1e6, use_fast_state=True)
    memorize_tokens(model, tokens, cfg, fast_state=fast_state)
    assert _fast_titan_delta_norm(fast_state, titan_before) == 0.0


def test_memorize_paths_filter_blocks_updates() -> None:
    model = _tiny_model()
    tokens = torch.randint(0, model.config.vocab_size, (1, 8))
    fast_state = model.init_fast_state()
    block_state = fast_state.blocks[0]
    titan_before = {k: v.cpu().clone() for k, v in block_state.titan_params.items()}  # type: ignore[union-attr]
    cfg = MemorizeConfig(enabled=True, steps=1, paths=(), use_fast_state=True)
    memorize_tokens(model, tokens, cfg, fast_state=fast_state)
    assert _fast_titan_delta_norm(fast_state, titan_before) == 0.0


def test_memorize_online_chunking_updates_once_per_target() -> None:
    model = _tiny_model_update_every_call()
    tokens = torch.randint(0, model.config.vocab_size, (1, 8))
    fast_state = model.init_fast_state()
    cfg = MemorizeConfig(enabled=True, online_chunk_size=1, use_fast_state=True)
    stats = memorize_tokens(model, tokens, cfg, fast_state=fast_state)
    assert stats["titan_update_events"] == float(tokens.size(1) - 1)


def test_teach_mask_restricts_memorization_updates() -> None:
    torch.manual_seed(0)
    model = _tiny_model_update_every_call()
    tokens = torch.randint(0, model.config.vocab_size, (1, 8))
    cfg = MemorizeConfig(enabled=True, steps=1, use_fast_state=True, paths=("cms_fast",))

    fast_state_masked = model.init_fast_state()
    zero_mask = torch.zeros((tokens.size(0), tokens.size(1)))
    stats_masked = memorize_tokens(
        model, tokens, cfg, fast_state=fast_state_masked, teach_mask=zero_mask
    )
    assert stats_masked["cms_fast_update_events"] == 0.0

    fast_state_full = model.init_fast_state()
    one_mask = torch.ones((tokens.size(0), tokens.size(1)))
    stats_full = memorize_tokens(
        model,
        tokens,
        cfg,
        fast_state=fast_state_full,
        teach_mask=one_mask,
    )
    assert stats_full["cms_fast_update_events"] > 0.0


def test_self_mod_lr_scales_fast_state_update_magnitude() -> None:
    torch.manual_seed(0)
    model_hi = _tiny_model_with_self_mod_lr(1e-3)
    torch.manual_seed(0)
    model_lo = _tiny_model_with_self_mod_lr(1e-4)
    tokens = torch.randint(0, model_hi.config.vocab_size, (1, 8))

    state_hi = model_hi.init_fast_state()
    state_lo = model_lo.init_fast_state()
    titan_hi_before = {k: v.cpu().clone() for k, v in state_hi.blocks[0].titan_params.items()}  # type: ignore[union-attr]
    titan_lo_before = {k: v.cpu().clone() for k, v in state_lo.blocks[0].titan_params.items()}  # type: ignore[union-attr]

    cfg = MemorizeConfig(enabled=True, steps=1, paths=("titan",), use_fast_state=True)
    memorize_tokens(model_hi, tokens, cfg, fast_state=state_hi)
    memorize_tokens(model_lo, tokens, cfg, fast_state=state_lo)

    hi = _fast_titan_delta_norm(state_hi, titan_hi_before)
    lo = _fast_titan_delta_norm(state_lo, titan_lo_before)
    assert hi > lo * 5.0
