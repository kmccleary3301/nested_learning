import torch
from omegaconf import OmegaConf

from nested_learning.levels import LevelSpec
from nested_learning.model import HOPEModel, ModelConfig
from nested_learning.training import _build_optimizer, _is_memory_param_name


def _make_small_hope_model() -> HOPEModel:
    return HOPEModel(
        ModelConfig(
            vocab_size=128,
            dim=16,
            num_layers=2,
            heads=2,
            titan_level=LevelSpec(name="titan", update_period=2),
            cms_levels=(LevelSpec(name="cms_fast", update_period=1),),
            block_variant="hope_hybrid",
        )
    )


def _optimizer_param_set(optimizer: torch.optim.Optimizer) -> set[torch.nn.Parameter]:
    params: set[torch.nn.Parameter] = set()
    for group in optimizer.param_groups:
        for param in group["params"]:
            params.add(param)
    return params


def test_param_policy_all_includes_all_trainable_params() -> None:
    model = _make_small_hope_model()
    cfg = OmegaConf.create({"optim": {"type": "adamw", "lr": 1e-3, "param_policy": "all"}})
    optimizer = _build_optimizer(model, cfg, device=torch.device("cpu"))
    opt_params = _optimizer_param_set(optimizer)
    expected = {p for _n, p in model.named_parameters() if p.requires_grad}
    assert opt_params == expected
    has_memory = any(
        _is_memory_param_name(name) for name, p in model.named_parameters() if p.requires_grad
    )
    assert has_memory


def test_param_policy_exclude_memory_drops_memory_params() -> None:
    model = _make_small_hope_model()
    cfg = OmegaConf.create(
        {"optim": {"type": "adamw", "lr": 1e-3, "param_policy": "exclude_memory"}}
    )
    optimizer = _build_optimizer(model, cfg, device=torch.device("cpu"))
    opt_params = _optimizer_param_set(optimizer)
    expected = {
        p
        for name, p in model.named_parameters()
        if p.requires_grad and not _is_memory_param_name(name)
    }
    assert opt_params == expected
    contains_memory = any(
        _is_memory_param_name(name) for name, p in model.named_parameters() if p in opt_params
    )
    assert not contains_memory


def test_param_policy_only_memory_keeps_only_memory_params() -> None:
    model = _make_small_hope_model()
    cfg = OmegaConf.create(
        {"optim": {"type": "adamw", "lr": 1e-3, "param_policy": "only_memory"}}
    )
    optimizer = _build_optimizer(model, cfg, device=torch.device("cpu"))
    opt_params = _optimizer_param_set(optimizer)
    expected = {
        p
        for name, p in model.named_parameters()
        if p.requires_grad and _is_memory_param_name(name)
    }
    assert expected
    assert opt_params == expected
