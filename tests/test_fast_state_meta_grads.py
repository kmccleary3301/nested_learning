import torch

from nested_learning.levels import LevelSpec
from nested_learning.model import HOPEModel, ModelConfig
from nested_learning.training import _is_memory_param_name


def test_fast_state_preserves_outer_grads_for_memory_meta_params() -> None:
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=64,
        dim=16,
        num_layers=1,
        heads=2,
        titan_level=LevelSpec(name="titan", update_period=1),
        cms_levels=(LevelSpec(name="cms_fast", update_period=1),),
        block_variant="hope_hybrid",
    )
    model = HOPEModel(cfg)
    fast_state = model.init_fast_state()
    tokens = torch.randint(0, cfg.vocab_size, (1, 8))
    logits = model(tokens, fast_state=fast_state)
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        tokens[:, 1:].reshape(-1),
    )
    loss.backward()

    memory_param_names = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and _is_memory_param_name(name)
    ]
    assert memory_param_names, "Test expected at least one memory parameter"
    assert any(
        model.get_parameter(name).grad is not None for name in memory_param_names
    ), "Expected at least one memory parameter grad in fast_state mode"
