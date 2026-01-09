import pytest
from omegaconf import OmegaConf

from nested_learning.training import _validate_fast_state_batch_semantics


def test_fast_state_batch_semantics_raises_when_strict() -> None:
    cfg = OmegaConf.create(
        {
            "train": {"use_fast_state": True, "fail_if_paper_faithful_disabled": True},
            "data": {"batch_size": 2},
        }
    )
    with pytest.raises(RuntimeError, match="fast state"):
        _validate_fast_state_batch_semantics(cfg)


def test_fast_state_batch_semantics_allows_batch1() -> None:
    cfg = OmegaConf.create(
        {
            "train": {"use_fast_state": True, "fail_if_paper_faithful_disabled": True},
            "data": {"batch_size": 1},
        }
    )
    _validate_fast_state_batch_semantics(cfg)

