import torch

from nested_learning.optim.deep import DeepMomentum


def test_deep_momentum_nl_preconditioner_projects_grad() -> None:
    grad = torch.randn(4, 6)
    context = torch.randn(6)
    optimizer = DeepMomentum(beta=0.0, beta2=0.0, variant="nl_l2_precond")
    update = optimizer(grad, context=context)
    unit = context / context.norm()
    expected = grad - (grad * unit).sum(dim=-1, keepdim=True) * unit
    assert torch.allclose(update, expected, atol=1e-5, rtol=1e-4)
    assert optimizer.last_metrics["ctx_norm"] > 0
    assert optimizer.last_metrics["proj_norm"] >= 0


def test_deep_momentum_nl_preconditioner_reduces_simple_objective() -> None:
    torch.manual_seed(0)
    context = torch.randn(6)
    weights = torch.randn(6)
    grad = torch.dot(weights, context) * context
    optimizer = DeepMomentum(beta=0.0, beta2=0.0, variant="nl_l2_precond")
    update = optimizer(grad, context=context)
    with torch.no_grad():
        old_obj = 0.5 * torch.dot(weights, context) ** 2
        new_weights = weights - 0.1 * update
        new_obj = 0.5 * torch.dot(new_weights, context) ** 2
    assert new_obj < old_obj
