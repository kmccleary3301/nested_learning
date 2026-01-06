import torch

from nested_learning.titan.self_modifying import SelfModifyingTitans, SelfModifyingTitansConfig


def test_self_modifying_titans_forward_shape() -> None:
    model = SelfModifyingTitans(SelfModifyingTitansConfig(dim=8))
    x = torch.randn(2, 5, 8)
    out = model(x)
    assert out.shape == x.shape


def test_self_modifying_titans_updates_fast_state() -> None:
    torch.manual_seed(0)
    model = SelfModifyingTitans(SelfModifyingTitansConfig(dim=8, eta_scale=1.0))
    x = torch.randn(1, 6, 8)
    state = model.init_fast_state()
    before = state.memory.w2.detach().clone()
    out, updated = model.forward_with_updates(x, state)
    assert out.shape == (1, 6, 8)
    assert not torch.allclose(before.unsqueeze(0), updated.memory.w2)


def test_self_modifying_titans_supports_batch_fast_state_updates() -> None:
    torch.manual_seed(0)
    model = SelfModifyingTitans(SelfModifyingTitansConfig(dim=8, eta_scale=1.0))
    x = torch.randn(2, 6, 8)
    state = model.init_fast_state()
    out, updated = model.forward_with_updates(x, state)
    assert out.shape == (2, 6, 8)
    assert updated.memory.w2.shape == (2, 8, 8)
    assert not torch.allclose(updated.memory.w2[0], updated.memory.w2[1])


def test_self_modifying_titans_chunked_outputs_match_no_update_with_single_chunk() -> None:
    torch.manual_seed(0)
    seq_len = 6
    model = SelfModifyingTitans(
        SelfModifyingTitansConfig(
            dim=8,
            eta_scale=1.0,
            chunk_size_other=seq_len,
            chunk_size_memory=seq_len,
        )
    )
    x = torch.randn(1, seq_len, 8)
    state = model.init_fast_state()
    before = state.memory.w2.detach().clone()

    out_no_update = model.forward_with_state(x, state)
    out_chunked, updated = model.forward_with_updates(x, state)

    assert torch.allclose(out_chunked, out_no_update, atol=1e-6)
    assert not torch.allclose(before.unsqueeze(0), updated.memory.w2)


def test_self_modifying_titans_dual_chunk_schedule_respects_memory_frequency() -> None:
    torch.manual_seed(0)
    model = SelfModifyingTitans(
        SelfModifyingTitansConfig(
            dim=8,
            eta_scale=1.0,
            chunk_size_other=1,
            chunk_size_memory=4,
        )
    )
    state = model.init_fast_state()
    x = torch.randn(1, 3, 8)
    before_other = state.k.w2.detach().clone()
    before_memory = state.memory.w2.detach().clone()

    _out, updated = model.forward_with_updates(x, state)

    assert not torch.allclose(before_other.unsqueeze(0), updated.k.w2)
    assert torch.allclose(before_memory.unsqueeze(0), updated.memory.w2, atol=1e-6)
