import torch

from nested_learning.titan.self_modifying import SelfModifyingTitans, SelfModifyingTitansConfig


def test_selfmod_fixed_q_does_not_update_q_memory() -> None:
    torch.manual_seed(0)
    model = SelfModifyingTitans(SelfModifyingTitansConfig(dim=8, eta_scale=1.0, adaptive_q=False))
    x = torch.randn(1, 6, 8)
    state = model.init_fast_state()
    before = state.q.w2.detach().clone()
    _out, updated = model.forward_with_updates(x, state)
    assert torch.allclose(before.unsqueeze(0), updated.q.w2, atol=1e-6, rtol=1e-6)


def test_selfmod_adaptive_q_updates_q_memory() -> None:
    torch.manual_seed(0)
    model = SelfModifyingTitans(SelfModifyingTitansConfig(dim=8, eta_scale=1.0, adaptive_q=True))
    x = torch.randn(1, 6, 8)
    state = model.init_fast_state()
    before = state.q.w2.detach().clone()
    _out, updated = model.forward_with_updates(x, state)
    assert not torch.allclose(before.unsqueeze(0), updated.q.w2)

