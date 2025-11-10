from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf


class BaseLogger:
    def log(self, metrics: Dict[str, Any], step: int) -> None:
        raise NotImplementedError

    def finish(self) -> None:
        pass


class NullLogger(BaseLogger):
    def log(self, metrics: Dict[str, Any], step: int) -> None:
        return


class JSONLogger(BaseLogger):
    def __init__(self, path: Path):
        self.path = path
        self.records: list[Dict[str, Any]] = []

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        payload = {"step": step, **metrics}
        self.records.append(payload)

    def finish(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.records, indent=2))


class WandbLogger(BaseLogger):
    def __init__(self, cfg: DictConfig, full_cfg: DictConfig):
        import wandb

        project = cfg.get("project", "nested-learning")
        run_name = cfg.get("run_name")
        config_dict = OmegaConf.to_container(full_cfg, resolve=True)
        self.run = wandb.init(project=project, name=run_name, config=config_dict)

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        if self.run is not None:
            self.run.log(metrics, step=step)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


def init_logger(logging_cfg: DictConfig | None, full_cfg: DictConfig) -> BaseLogger:
    if logging_cfg is None or not logging_cfg.get("enabled", False):
        return NullLogger()
    backend = logging_cfg.get("backend", "wandb").lower()
    if backend == "wandb":
        return WandbLogger(logging_cfg, full_cfg)
    if backend == "json":
        path = Path(logging_cfg.get("path", "logs/train_metrics.json"))
        return JSONLogger(path)
    return NullLogger()
