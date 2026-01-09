import torch
import torch.nn.functional as F

from nested_learning.titan.self_modifying import ResidualMLPMemory


def test_residual_mlp_memory_matches_eq91_when_dims_match() -> None:
    torch.manual_seed(0)
    mem = ResidualMLPMemory(in_dim=8, out_dim=8, hidden_dim=8, activation=F.gelu, use_skip=False)
    assert mem.w_skip is None
    x = torch.randn(2, 5, 8)
    with torch.no_grad():
        expected = x + mem.w1(mem.activation(mem.w2(x)))
        actual = mem(x)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_residual_mlp_memory_uses_projection_skip_when_dims_differ() -> None:
    mem = ResidualMLPMemory(in_dim=8, out_dim=1, hidden_dim=8, activation=F.gelu, use_skip=True)
    assert mem.w_skip is not None


def test_residual_mlp_memory_disables_projection_skip_in_faithful_mode() -> None:
    torch.manual_seed(0)
    mem = ResidualMLPMemory(in_dim=8, out_dim=1, hidden_dim=8, activation=F.gelu, use_skip=False)
    assert mem.w_skip is None
    x = torch.randn(2, 5, 8)
    with torch.no_grad():
        expected = mem.w1(mem.activation(mem.w2(x)))
        actual = mem(x)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
