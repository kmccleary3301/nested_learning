import torch

from nested_learning.optim.m3 import M3


def test_m3_updates_and_slow_momentum() -> None:
    torch.manual_seed(0)
    param = torch.nn.Parameter(torch.ones(2, 2))
    opt = M3(
        [param],
        lr=0.1,
        beta1=0.9,
        beta2=0.9,
        beta3=0.5,
        alpha=1.0,
        ns_steps=1,
        slow_chunk=2,
        eps=1e-6,
    )
    param.grad = torch.ones_like(param)
    opt.step()
    first = param.detach().clone()
    param.grad = torch.ones_like(param)
    opt.step()
    state = opt.state[param]
    assert not torch.allclose(first, param)
    assert torch.any(state["o2"] != 0)
