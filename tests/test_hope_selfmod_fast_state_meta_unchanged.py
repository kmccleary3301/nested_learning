import torch

from nested_learning.levels import LevelSpec
from nested_learning.memorize import snapshot_state_dict
from nested_learning.model import HOPEModel, ModelConfig
from nested_learning.training import compute_teach_signal


def test_hope_selfmod_fast_state_updates_do_not_mutate_meta_params() -> None:
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
    baseline = snapshot_state_dict(model)
    fast_state = model.init_fast_state()
    tokens = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        logits = model(tokens, fast_state=fast_state)
        teach = compute_teach_signal(model, logits, tokens)
        _ = model(tokens, teach_signal=teach, fast_state=fast_state)
    for name, value in model.state_dict().items():
        assert torch.allclose(baseline[name], value.cpu(), atol=1e-6)

