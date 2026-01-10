import torch

from nested_learning.levels import LevelSpec
from nested_learning.model import HOPEModel, ModelConfig
from nested_learning.training import compute_teach_signal


def _cms_delta_l1(state, level_name: str) -> float:
    params = state.blocks[0].cms_params[level_name]
    return float(sum(delta.abs().sum().item() for delta in params.values()))


def _logit_entropy(logits: torch.Tensor) -> float:
    logits_detached = logits[:, :-1].detach().float()
    probs = torch.softmax(logits_detached, dim=-1)
    entropy = -(probs * torch.log(probs.clamp(min=1e-9))).sum(dim=-1).mean()
    return float(entropy.item())


def _next_token_loss(logits: torch.Tensor, tokens: torch.Tensor) -> float:
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        tokens[:, 1:].reshape(-1),
    )
    return float(loss.detach().item())


def test_surprise_metric_loss_gates_updates_when_threshold_set() -> None:
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=32,
        dim=16,
        num_layers=1,
        heads=2,
        titan_level=LevelSpec(name="titan", update_period=1),
        cms_levels=(LevelSpec(name="cms_fast", update_period=2),),
        block_variant="hope_attention",
        surprise_metric="loss",
        surprise_threshold=0.0,
    )
    model = HOPEModel(cfg).eval()
    state = model.init_fast_state()
    tokens = torch.randint(0, cfg.vocab_size, (1, 8))

    with torch.no_grad():
        logits = model(tokens, fast_state=state)
        teach = compute_teach_signal(model, logits, tokens)
        loss_value = _next_token_loss(logits, tokens)
        model.set_surprise_threshold(loss_value + 1.0)
        _ = model(
            tokens,
            teach_signal=teach,
            surprise_value=loss_value,
            fast_state=state,
        )
    assert _cms_delta_l1(state, "cms_fast") == 0.0

    with torch.no_grad():
        model.set_surprise_threshold(loss_value - 1.0)
        _ = model(
            tokens,
            teach_signal=teach,
            surprise_value=loss_value,
            fast_state=state,
        )
    assert _cms_delta_l1(state, "cms_fast") > 0.0


def test_surprise_metric_entropy_gates_updates_when_threshold_set() -> None:
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=32,
        dim=16,
        num_layers=1,
        heads=2,
        titan_level=LevelSpec(name="titan", update_period=1),
        cms_levels=(LevelSpec(name="cms_fast", update_period=2),),
        block_variant="hope_attention",
        surprise_metric="logit_entropy",
        surprise_threshold=0.0,
    )
    model = HOPEModel(cfg).eval()
    state = model.init_fast_state()
    tokens = torch.randint(0, cfg.vocab_size, (1, 8))

    with torch.no_grad():
        logits = model(tokens, fast_state=state)
        teach = compute_teach_signal(model, logits, tokens)
        entropy_value = _logit_entropy(logits)
        model.set_surprise_threshold(entropy_value + 1.0)
        _ = model(
            tokens,
            teach_signal=teach,
            surprise_value=entropy_value,
            fast_state=state,
        )
    assert _cms_delta_l1(state, "cms_fast") == 0.0

    with torch.no_grad():
        model.set_surprise_threshold(entropy_value - 1.0)
        _ = model(
            tokens,
            teach_signal=teach,
            surprise_value=entropy_value,
            fast_state=state,
        )
    assert _cms_delta_l1(state, "cms_fast") > 0.0


def test_surprise_metric_requires_external_value_when_threshold_set() -> None:
    cfg = ModelConfig(
        vocab_size=32,
        dim=16,
        num_layers=1,
        heads=2,
        titan_level=LevelSpec(name="titan", update_period=1),
        cms_levels=(LevelSpec(name="cms_fast", update_period=2),),
        block_variant="hope_attention",
        surprise_metric="loss",
        surprise_threshold=0.1,
    )
    model = HOPEModel(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (1, 8))
    state = model.init_fast_state()
    with torch.no_grad():
        logits = model(tokens, fast_state=state)
        teach = compute_teach_signal(model, logits, tokens)
        try:
            _ = model(tokens, teach_signal=teach, fast_state=state)
        except ValueError as err:
            assert "requires passing surprise_value" in str(err)
        else:
            raise AssertionError("Expected ValueError when surprise_value is omitted")

