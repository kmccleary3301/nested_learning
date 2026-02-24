"""Nested Learning (HOPE) reproduction package."""

from importlib.metadata import PackageNotFoundError, version

from .levels import LevelClock, LevelSpec  # noqa: F401

try:
    __version__ = version("nested-learning")
except PackageNotFoundError:  # pragma: no cover - editable/local source tree
    __version__ = "0.1.0"

__all__ = ["LevelClock", "LevelSpec", "__version__"]
