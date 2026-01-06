from omegaconf import OmegaConf

from nested_learning.hope.block import HOPESelfModBlock
from nested_learning.training import build_model_from_cfg


def test_build_model_from_cfg_plumbs_selfmod_fields() -> None:
    model_cfg = OmegaConf.create(
        {
            "type": "hope",
            "vocab_size": 32,
            "dim": 16,
            "num_layers": 1,
            "heads": 4,
            "titan_level": {"name": "titan", "update_period": 1, "optimizer_key": "titan_opt"},
            "cms_levels": [{"name": "cms_fast", "update_period": 1, "optimizer_key": "cms_opt"}],
            "block_variant": "hope_selfmod",
            "self_mod_chunk_size": 3,
            "self_mod_chunk_size_memory": 7,
            "self_mod_objective": "dot",
            "self_mod_stopgrad_vhat": False,
            "self_mod_use_rank1_precond": False,
            "self_mod_use_alpha": False,
            "self_mod_momentum": 0.5,
        }
    )
    model = build_model_from_cfg(model_cfg)
    assert model.config.self_mod_chunk_size == 3
    assert model.config.self_mod_chunk_size_memory == 7
    assert model.config.self_mod_objective == "dot"
    assert model.config.self_mod_stopgrad_vhat is False
    assert model.config.self_mod_use_rank1_precond is False
    assert model.config.self_mod_use_alpha is False
    assert abs(model.config.self_mod_momentum - 0.5) < 1e-9

    block = model.blocks[0]
    assert isinstance(block, HOPESelfModBlock)
    assert block.selfmod.config.chunk_size_other == 3
    assert block.selfmod.config.chunk_size_memory == 7
    assert block.selfmod.config.objective == "dot"
    assert block.selfmod.config.stopgrad_vhat is False
    assert block.selfmod.config.use_rank1_precond is False
    assert block.selfmod.config.use_alpha is False
    assert abs(block.selfmod.config.momentum - 0.5) < 1e-9
