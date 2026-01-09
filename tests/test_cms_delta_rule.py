import torch

from nested_learning.hope.block import _chunk_loss


def test_cms_target_shift_loss_grad_is_proportional_to_delta() -> None:
    torch.manual_seed(0)
    prediction = torch.randn(2, 5, 7, requires_grad=True)
    delta = torch.randn(2, 5, 7)
    active = torch.tensor(
        [
            [1, 1, 0, 1, 1],
            [0, 1, 1, 1, 0],
        ],
        dtype=torch.float32,
    )
    mask_f = active.unsqueeze(-1)
    loss = _chunk_loss(prediction, delta, mask_f, reduction="sum")
    loss.backward()
    assert prediction.grad is not None
    expected = 2.0 * delta * mask_f
    assert torch.allclose(prediction.grad, expected, atol=1e-6, rtol=1e-6)


def test_cms_chunk_loss_sum_scales_relative_to_mean() -> None:
    torch.manual_seed(0)
    prediction = torch.randn(1, 4, 5, requires_grad=True)
    delta = torch.randn(1, 4, 5)
    mask_f = torch.ones(1, 4, 1)

    loss_sum = _chunk_loss(prediction, delta, mask_f, reduction="sum")
    loss_sum.backward()
    assert prediction.grad is not None
    grad_sum = prediction.grad.detach().clone()

    prediction.grad.zero_()
    loss_mean = _chunk_loss(prediction, delta, mask_f, reduction="mean")
    loss_mean.backward()
    assert prediction.grad is not None
    grad_mean = prediction.grad.detach().clone()

    scale = float(mask_f.sum().item())
    assert torch.allclose(grad_sum, grad_mean * scale, atol=1e-6, rtol=1e-6)
