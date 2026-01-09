import torch

from nested_learning.device import resolve_device


def test_resolve_device_mps_falls_back_when_unavailable() -> None:
    device = resolve_device("mps")
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    assert device.type == ("mps" if mps_available else "cpu")

