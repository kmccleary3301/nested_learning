import torch

from nested_learning.hope.block import HOPEAttentionBlock, HOPEBlock, HOPESelfModBlock
from nested_learning.levels import LevelSpec
from nested_learning.model import HOPEModel, ModelConfig
from nested_learning.titan.memory import TitanMemory
from nested_learning.transformer import TransformerBlock


def _base_cfg(*, block_variant: str) -> ModelConfig:
    titan = LevelSpec(name="titan", update_period=1, optimizer_key="titan_opt")
    cms = [LevelSpec(name="cms_fast", update_period=1, optimizer_key="cms_opt")]
    return ModelConfig(
        vocab_size=32,
        dim=16,
        num_layers=1,
        heads=4,
        titan_level=titan,
        cms_levels=cms,
        block_variant=block_variant,
        optimizers=None,
    )


def test_hope_hybrid_variant_contains_titan_memory() -> None:
    model = HOPEModel(_base_cfg(block_variant="hope_hybrid"))
    block = model.blocks[0]
    assert isinstance(block, HOPEBlock)
    assert isinstance(block.titan_memory, TitanMemory)


def test_hope_attention_variant_excludes_titan_memory() -> None:
    model = HOPEModel(_base_cfg(block_variant="hope_attention"))
    block = model.blocks[0]
    assert isinstance(block, HOPEAttentionBlock)
    assert not hasattr(block, "titan_memory")

    tokens = torch.randint(0, model.config.vocab_size, (2, 5))
    logits = model(tokens)
    assert logits.shape == (2, 5, model.config.vocab_size)


def test_hope_selfmod_variant_excludes_titan_memory() -> None:
    model = HOPEModel(_base_cfg(block_variant="hope_selfmod"))
    block = model.blocks[0]
    assert isinstance(block, HOPESelfModBlock)
    assert not hasattr(block, "titan_memory")
    assert hasattr(block, "selfmod")

    fast_state = model.init_fast_state()
    tokens = torch.randint(0, model.config.vocab_size, (1, 5))
    logits = model(tokens, fast_state=fast_state)
    assert logits.shape == (1, 5, model.config.vocab_size)


def test_transformer_variant_runs_with_and_without_fast_state() -> None:
    model = HOPEModel(_base_cfg(block_variant="transformer"))
    block = model.blocks[0]
    assert isinstance(block, TransformerBlock)

    tokens = torch.randint(0, model.config.vocab_size, (2, 5))
    logits = model(tokens)
    assert logits.shape == (2, 5, model.config.vocab_size)

    fast_state = model.init_fast_state()
    logits_fast = model(tokens, fast_state=fast_state)
    assert logits_fast.shape == (2, 5, model.config.vocab_size)
