from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from nested_learning.training import build_model_from_cfg


def _compose_config(name: str):
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(config_name=name)


def test_pilot_paper_faithful_config_composes() -> None:
    cfg = _compose_config("pilot_paper_faithful")
    assert cfg.model.cms_flush_partial_at_end is True
    assert cfg.model.surprise_threshold is None
    assert cfg.data.batch_size == 1
    assert cfg.train.use_fast_state is True
    assert cfg.train.fail_if_paper_faithful_disabled is True
    assert cfg.optim.param_policy == "all"
    build_model_from_cfg(cfg.model)


def test_pilot_selfmod_paper_faithful_config_composes() -> None:
    cfg = _compose_config("pilot_selfmod_paper_faithful")
    assert cfg.model.block_variant == "hope_selfmod"
    assert cfg.model.cms_flush_partial_at_end is True
    assert cfg.model.surprise_threshold is None
    assert cfg.model.self_mod_use_skip is False
    assert cfg.data.batch_size == 1
    assert cfg.train.use_fast_state is True
    assert cfg.train.fail_if_paper_faithful_disabled is True
    assert cfg.optim.param_policy == "all"
    build_model_from_cfg(cfg.model)
