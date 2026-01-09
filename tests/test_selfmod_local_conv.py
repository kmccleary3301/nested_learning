import torch

from nested_learning.titan.self_modifying import SelfModifyingTitans, SelfModifyingTitansConfig


def test_selfmod_local_conv_is_causal() -> None:
    torch.manual_seed(0)
    model = SelfModifyingTitans(SelfModifyingTitansConfig(dim=4, local_conv_window=4))
    assert model.local_conv is not None
    with torch.no_grad():
        model.local_conv.weight.fill_(1.0)
    x = torch.zeros(1, 6, 4)
    x[0, 4, 0] = 1.0
    y = model._apply_local_conv(x)
    assert torch.allclose(y[0, :4, 0], torch.zeros(4))
    assert y[0, 4, 0].item() != 0.0

