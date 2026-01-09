import torch

from nested_learning.optim.m3 import M3


def test_m3_slow_momentum_applies_next_chunk_not_boundary_step() -> None:
    param = torch.nn.Parameter(torch.tensor([0.0]))
    opt = M3(
        [param],
        lr=1.0,
        beta1=1.0,
        beta2=0.0,
        beta3=1.0,
        alpha=1.0,
        eps=1.0,
        ns_steps=0,
        slow_chunk=2,
        weight_decay=0.0,
    )
    param.grad = torch.tensor([1.0])
    opt.step()
    param.grad = torch.tensor([1.0])
    opt.step()
    # With correct timing, the slow momentum (o2) is updated after step 2 and therefore
    # does not affect the step-2 update itself.
    assert torch.allclose(param.detach(), torch.tensor([-3.0]))

