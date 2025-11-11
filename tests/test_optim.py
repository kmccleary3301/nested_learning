import torch

from nested_learning.optim.deep import DeepMomentum


def test_deep_momentum_nl_preconditioner_projects_grad() -> None:
    grad = torch.randn(4, 6)
    context = torch.randn(6)
    optimizer = DeepMomentum(beta=0.0, beta2=0.0, variant="nl_l2_precond")
    update = optimizer(grad, context=context)
    unit = context / context.norm()
    projector = torch.outer(unit, unit)
    expected = grad - torch.matmul(grad, projector)
    assert torch.allclose(update, expected, atol=1e-5, rtol=1e-4)
