from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from nested_learning.cli import app
from nested_learning.config_utils import compose_config


def test_doctor_json_output() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert "python_version" in payload
    assert "torch_version" in payload
    assert "default_device" in payload


def test_smoke_cpu_passes() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "smoke",
            "--config-name",
            "pilot_smoke",
            "--device",
            "cpu",
            "--batch-size",
            "1",
            "--seq-len",
            "8",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["device"] == "cpu"
    assert payload["logits_shape"][0] == 1
    assert payload["logits_shape"][1] == 8


def test_smoke_auto_passes() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "smoke",
            "--config-name",
            "pilot_smoke",
            "--device",
            "auto",
            "--batch-size",
            "1",
            "--seq-len",
            "8",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"


def test_audit_reports_tied_weights() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["audit", "--config-name", "pilot_paper_faithful"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["has_embed"] is True
    assert payload["has_lm_head"] is True
    assert payload["lm_tied_to_embedding"] is True


def test_train_dry_run_prints_config() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "train",
            "--config-name",
            "pilot_smoke",
            "--dry-run",
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "model:" in result.stdout
    assert "train:" in result.stdout


def test_compose_config_with_explicit_config_dir(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "mini.yaml").write_text(
        "model:\n"
        "  type: hope\n"
        "  vocab_size: 128\n"
        "  dim: 32\n"
        "  num_layers: 1\n"
        "  heads: 2\n"
        "  titan_level: {name: titan, update_period: 8, optimizer_key: titan_opt}\n"
        "  cms_levels: []\n"
        "  optimizers: {}\n"
        "data: {source: synthetic, vocab_size: 128, seq_len: 8, dataset_size: 8, batch_size: 1}\n"
        "train: {device: cpu, steps: 1, log_interval: 1}\n",
        encoding="utf-8",
    )
    cfg = compose_config("mini", config_dir=cfg_dir)
    assert cfg.model.vocab_size == 128
    assert cfg.train.device == "cpu"


def test_python_module_entrypoint_help() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo_root / "src") + (os.pathsep + existing if existing else "")
    result = subprocess.run(
        [sys.executable, "-m", "nested_learning", "--help"],
        cwd=repo_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Nested Learning CLI" in result.stdout
