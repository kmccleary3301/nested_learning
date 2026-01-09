from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.data import shard_corpus, train_tokenizer


def test_train_tokenizer_manifest_supports_text_data_files(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world\nthis is a test\nanother line\n", encoding="utf-8")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "\n".join(
            [
                "datasets:",
                "  - name: local",
                "    dataset: text",
                "    split: train",
                "    text_column: text",
                f"    data_files: {corpus}",
                "    sample_limit: 10",
                "",
            ]
        ),
        encoding="utf-8",
    )
    specs = train_tokenizer._load_specs_from_manifest(manifest)  # noqa: SLF001
    assert len(specs) == 1
    assert specs[0].dataset == "text"
    assert specs[0].split == "train"
    assert specs[0].data_files == str(corpus)
    buf = io.StringIO()
    count = train_tokenizer._write_samples(specs[0], buf)  # noqa: SLF001
    assert count == 3


def test_shard_corpus_accepts_text_data_files_with_train_split(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(("hello world " * 100).strip() + "\n", encoding="utf-8")
    out_dir = tmp_path / "shards"
    cfg = shard_corpus.ShardConfig(
        name="local",
        dataset="text",
        split="train",
        subset=None,
        text_column="text",
        tokenizer_path=Path("tests/data/tiny_tokenizer.model"),
        seq_len=4,
        sequences_per_shard=2,
        output_dir=out_dir,
        eos_id=-1,
        max_records=10,
        data_files=str(corpus),
    )
    stats = shard_corpus.shard_dataset(cfg)
    assert stats["records"] > 0
    assert stats["sequences"] > 0
    assert stats["shards"] > 0
    assert list(out_dir.glob("shard_*.npy"))


def test_train_tokenizer_allows_small_corpus_with_no_hard_vocab_limit(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(("hello world\n" * 20).strip() + "\n", encoding="utf-8")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "\n".join(
            [
                "datasets:",
                "  - name: local",
                "    dataset: text",
                "    split: train",
                "    text_column: text",
                f"    data_files: {corpus}",
                "    sample_limit: 50",
                "",
            ]
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "tokenizer"
    log_file = tmp_path / "tokenizer_log.json"
    repo_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts/data/train_tokenizer.py"),
            "--manifest",
            str(manifest),
            "--vocab-size",
            "1000",
            "--model-type",
            "unigram",
            "--output-dir",
            str(out_dir),
            "--log-file",
            str(log_file),
            "--no-hard-vocab-limit",
        ],
        check=True,
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert (out_dir / "spm_1000_unigram.model").exists()
    assert log_file.exists()
