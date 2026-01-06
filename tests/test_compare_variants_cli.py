import json
import subprocess
import sys
from pathlib import Path

import sentencepiece as spm
import torch
from omegaconf import OmegaConf

from nested_learning.training import build_model_from_cfg


def _train_tiny_sentencepiece(tmp_path: Path, *, vocab_size: int) -> Path:
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text(
        "\n".join(
            [
                "This is a tiny corpus for sentencepiece.",
                "Remember that the secret key is KEY-1234.",
                "Question: What is the passkey? Answer: PASSKEY-1234.",
            ]
        )
    )
    model_prefix = tmp_path / "spm_test"
    spm.SentencePieceTrainer.Train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        hard_vocab_limit=False,
        model_type="unigram",
        bos_id=1,
        eos_id=2,
        pad_id=0,
        unk_id=3,
        character_coverage=1.0,
    )
    return model_prefix.with_suffix(".model")


def _write_minimal_model_config(path: Path, *, vocab_size: int, block_variant: str) -> None:
    payload = {
        "model": {
            "vocab_size": vocab_size,
            "dim": 16,
            "num_layers": 1,
            "heads": 4,
            "block_variant": block_variant,
            "titan_level": {"name": "titan", "update_period": 1},
            "cms_levels": [{"name": "cms_fast", "update_period": 1}],
        }
    }
    path.write_text(OmegaConf.to_yaml(OmegaConf.create(payload)))


def _write_checkpoint(path: Path, config_path: Path) -> None:
    cfg = OmegaConf.load(config_path)
    model = build_model_from_cfg(cfg.model)
    torch.save({"model": model.state_dict()}, path)


def test_compare_variants_cli_smoke(tmp_path: Path) -> None:
    vocab_size = 64
    spm_model = _train_tiny_sentencepiece(tmp_path, vocab_size=vocab_size)

    config_a = tmp_path / "a.yaml"
    config_b = tmp_path / "b.yaml"
    _write_minimal_model_config(config_a, vocab_size=vocab_size, block_variant="transformer")
    _write_minimal_model_config(config_b, vocab_size=vocab_size, block_variant="transformer")

    ckpt_a = tmp_path / "a.pt"
    ckpt_b = tmp_path / "b.pt"
    _write_checkpoint(ckpt_a, config_a)
    _write_checkpoint(ckpt_b, config_b)

    out_path = tmp_path / "out.json"
    cmd = [
        sys.executable,
        "scripts/eval/compare_variants.py",
        "--a-config",
        str(config_a),
        "--a-checkpoint",
        str(ckpt_a),
        "--b-config",
        str(config_b),
        "--b-checkpoint",
        str(ckpt_b),
        "--tokenizer-path",
        str(spm_model),
        "--device",
        "cpu",
        "--smoke",
        "--output",
        str(out_path),
    ]
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    assert completed.returncode == 0
    data = json.loads(out_path.read_text())
    assert "a" in data and "b" in data
    assert "passkey" in data["a"] and "niah" in data["a"]
    assert "accuracy_base" in data["a"]["passkey"]
