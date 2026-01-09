import torch

from nested_learning.levels import LevelSpec
from nested_learning.model import HOPEModel, ModelConfig


def test_fast_state_zero_deltas_matches_meta_forward() -> None:
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
    model = HOPEModel(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (1, 8))
    fast_state = model.init_fast_state()
    with torch.no_grad():
        logits_meta = model(tokens)
        logits_fast = model(tokens, fast_state=fast_state)
    assert torch.allclose(logits_meta, logits_fast, atol=1e-6)

