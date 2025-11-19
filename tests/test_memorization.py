import torch

from nested_learning.hope.block import HOPEBlockConfig
from nested_learning.levels import LevelSpec
from nested_learning.model import HOPEModel, ModelConfig
from nested_learning.memorize import MemorizeConfig, memorize_tokens, snapshot_state_dict


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


def test_memorize_tokens_updates_parameters() -> None:
    model = _tiny_model()
    baseline = snapshot_state_dict(model)
    tokens = torch.randint(0, model.config.vocab_size, (1, 8))
    cfg = MemorizeConfig(enabled=True, steps=2)
    memorize_tokens(model, tokens, cfg)
    changed = any(
        not torch.allclose(baseline[name], param.cpu(), atol=1e-6)
        for name, param in model.state_dict().items()
    )
    assert changed


def test_memorize_tokens_can_be_reset() -> None:
    model = _tiny_model()
    baseline = snapshot_state_dict(model)
    tokens = torch.randint(0, model.config.vocab_size, (1, 8))
    cfg = MemorizeConfig(enabled=True, steps=1)
    memorize_tokens(model, tokens, cfg)
    model.load_state_dict(baseline)
    for name, param in model.state_dict().items():
        assert torch.allclose(baseline[name], param.cpu(), atol=1e-6)


def test_memorize_respects_surprise_threshold() -> None:
    model = _tiny_model()
    baseline = snapshot_state_dict(model)
    tokens = torch.randint(0, model.config.vocab_size, (1, 8))
    cfg = MemorizeConfig(enabled=True, steps=1, surprise_threshold=1e6)
    memorize_tokens(model, tokens, cfg)
    for name, param in model.state_dict().items():
        assert torch.allclose(baseline[name], param.cpu(), atol=1e-6)


def test_memorize_paths_filter_blocks_updates() -> None:
    model = _tiny_model()
    baseline = snapshot_state_dict(model)
    tokens = torch.randint(0, model.config.vocab_size, (1, 8))
    cfg = MemorizeConfig(enabled=True, steps=1, paths=())
    memorize_tokens(model, tokens, cfg)
    for name, param in model.state_dict().items():
        assert torch.allclose(baseline[name], param.cpu(), atol=1e-6)
