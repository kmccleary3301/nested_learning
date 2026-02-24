from __future__ import annotations

from contextlib import contextmanager
from importlib.resources import as_file, files
from pathlib import Path
from typing import Iterator

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

from .training import unwrap_config


def find_repo_root(start: Path | None = None) -> Path | None:
    cursor = (start or Path.cwd()).resolve()
    for candidate in (cursor, *cursor.parents):
        if (candidate / ".git").exists() and (candidate / "configs").exists():
            return candidate
    return None


@contextmanager
def resolved_config_dir(config_dir: Path | None = None) -> Iterator[Path]:
    if config_dir is not None:
        yield config_dir.resolve()
        return

    module_path = Path(__file__).resolve()
    repo_config_dir = module_path.parents[2] / "configs"
    if repo_config_dir.exists():
        yield repo_config_dir
        return

    package_configs = files("nested_learning").joinpath("configs")
    with as_file(package_configs) as pkg_dir:
        yield Path(pkg_dir)


def compose_config(
    config_name: str,
    *,
    overrides: list[str] | None = None,
    config_dir: Path | None = None,
) -> DictConfig:
    with resolved_config_dir(config_dir) as cfg_dir:
        GlobalHydra.instance().clear()
        with initialize_config_dir(version_base=None, config_dir=str(cfg_dir)):
            cfg = compose(config_name=config_name, overrides=overrides or [])
    return unwrap_config(cfg)
