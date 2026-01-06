import torch

from nested_learning.levels import LevelSpec
from nested_learning.memorize import MemorizeConfig, memorize_tokens
from nested_learning.model import HOPEModel, ModelConfig


def _tiny_variant(variant: str) -> HOPEModel:
    titan = LevelSpec(name="titan", update_period=1, optimizer_key="titan_opt")
    cms = (LevelSpec(name="cms_fast", update_period=1, optimizer_key="cms_opt"),)
    cfg = ModelConfig(
        vocab_size=32,
        dim=16,
        num_layers=1,
        heads=4,
        titan_level=titan,
        cms_levels=cms,
        optimizers=None,
        teach_scale=0.1,
        block_variant=variant,
    )
    return HOPEModel(cfg).eval()


def test_hope_attention_adapts_transformer_does_not() -> None:
    tokens = torch.randint(0, 32, (1, 16), generator=torch.Generator().manual_seed(1337))
    cfg = MemorizeConfig(enabled=True, steps=1, use_fast_state=True, paths=("cms_fast",))

    torch.manual_seed(0)
    hope = _tiny_variant("hope_attention")
    state = hope.init_fast_state()
    with torch.no_grad():
        before = hope(tokens, fast_state=state).detach().clone()
    stats = memorize_tokens(hope, tokens, cfg, fast_state=state)
    with torch.no_grad():
        after = hope(tokens, fast_state=state).detach().clone()
    assert not torch.allclose(before, after)
    assert stats["cms_fast_update_events"] > 0.0

    torch.manual_seed(0)
    transformer = _tiny_variant("transformer")
    state = transformer.init_fast_state()
    with torch.no_grad():
        before = transformer(tokens, fast_state=state).detach().clone()
    stats = memorize_tokens(transformer, tokens, cfg, fast_state=state)
    with torch.no_grad():
        after = transformer(tokens, fast_state=state).detach().clone()
    assert torch.allclose(before, after, atol=0.0, rtol=0.0)
    assert stats["cms_fast_update_events"] == 0.0

