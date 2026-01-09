import torch

from nested_learning.titan.self_modifying import (
    ResidualMLPMemoryState,
    SelfModifyingTitans,
    SelfModifyingTitansConfig,
)


def test_selfmod_linear_memory_l2_grad_matches_analytic() -> None:
    torch.manual_seed(0)
    model = SelfModifyingTitans(
        SelfModifyingTitansConfig(
            dim=4,
            eta_scale=1.0,
            objective="l2",
            stopgrad_vhat=True,
            use_rank1_precond=False,
            use_alpha=False,
            local_conv_window=None,
        )
    )
    w_skip = torch.randn(4, 4)
    frozen = ResidualMLPMemoryState(
        w1=torch.zeros_like(model.m_memory.w1.weight),
        w2=torch.zeros_like(model.m_memory.w2.weight),
        w_skip=w_skip.clone(),
    )
    k = torch.randn(3, 4)
    v = torch.randn(3, 4)
    g1, g2, gskip = model._memory_grads(frozen, k, v)
    assert gskip is not None

    with torch.no_grad():
        pred = k @ w_skip.t()
        vhat = v @ w_skip.t()
        diff = pred - vhat
        expected = 2.0 * torch.einsum("bi,bj->bij", diff, k).sum(dim=0)

    assert torch.allclose(gskip, expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(g1, torch.zeros_like(g1), atol=1e-6, rtol=1e-6)
    assert torch.allclose(g2, torch.zeros_like(g2), atol=1e-6, rtol=1e-6)

