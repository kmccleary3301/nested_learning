import torch
import torch.nn.functional as F

from nested_learning.levels import LevelSpec
from nested_learning.model import HOPEModel, ModelConfig
from nested_learning.training import compute_teach_signal


def _tiny_config() -> ModelConfig:
    titan = LevelSpec(name="titan", update_period=2, optimizer_key="titan_opt")
    cms = [
        LevelSpec(name="cms_fast", update_period=1, optimizer_key="cms_opt"),
        LevelSpec(name="cms_mid", update_period=4, optimizer_key="cms_opt"),
    ]
    return ModelConfig(
        vocab_size=32,
        dim=16,
        num_layers=2,
        heads=4,
        titan_level=titan,
        cms_levels=cms,
        optimizers=None,
        teach_scale=0.1,
    )


def test_teach_signal_matches_gradient() -> None:
    torch.manual_seed(0)
    cfg = _tiny_config()
    model = HOPEModel(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, 6))

    hidden_cache: dict[str, torch.Tensor] = {}

    def hook(_, __, output: torch.Tensor) -> None:
        output.retain_grad()
        hidden_cache["hidden"] = output

    handle = model.norm.register_forward_hook(hook)
    logits = model(tokens)
    teach_signal = compute_teach_signal(model, logits, tokens)
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, cfg.vocab_size),
        tokens[:, 1:].reshape(-1),
    )
    loss.backward()
    handle.remove()

    hidden = hidden_cache["hidden"]
    assert hidden.grad is not None
    grad = hidden.grad
    assert torch.allclose(teach_signal, grad, atol=1e-5, rtol=1e-4)
