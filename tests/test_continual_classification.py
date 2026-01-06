from pathlib import Path

import sentencepiece as spm
import torch

from nested_learning.continual_classification import ClassificationExample
from nested_learning.continual_streaming import (
    ContinualEvalConfig,
    build_streaming_tasks,
    evaluate_continual_classification,
)
from nested_learning.levels import LevelSpec
from nested_learning.memorize import MemorizeConfig
from nested_learning.model import HOPEModel, ModelConfig
from nested_learning.tokenizer import SentencePieceTokenizer


def _train_tiny_sentencepiece(tmp_path: Path, *, vocab_size: int) -> Path:
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text(
        "\n".join(
            [
                "Text: hello world Label: A",
                "Text: goodbye world Label: B",
                "Text: foo bar Label: C",
                "Text: baz qux Label: D",
            ]
        )
    )
    model_prefix = tmp_path / "spm_continual"
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


def _tiny_transformer_model(vocab_size: int) -> HOPEModel:
    cfg = ModelConfig(
        vocab_size=vocab_size,
        dim=16,
        num_layers=1,
        heads=4,
        titan_level=LevelSpec(name="titan", update_period=1),
        cms_levels=[LevelSpec(name="cms_fast", update_period=1)],
        block_variant="transformer",
    )
    return HOPEModel(cfg).eval()


def _toy_examples() -> list[ClassificationExample]:
    examples = []
    for label in ["A", "B", "C", "D"]:
        for idx in range(3):
            examples.append(ClassificationExample(text=f"example {idx} for {label}", label=label))
    return examples


def test_build_streaming_tasks_balanced_split() -> None:
    cfg = ContinualEvalConfig(task_size=2, seed=0, train_per_label=2, eval_per_label=1)
    tasks = build_streaming_tasks(_toy_examples(), cfg=cfg)
    assert len(tasks) == 2
    for task in tasks:
        assert len(task.labels) == 2
        assert len(task.train) == 4
        assert len(task.eval) == 2


def test_evaluate_continual_classification_runs(tmp_path: Path) -> None:
    vocab_size = 64
    spm_model = _train_tiny_sentencepiece(tmp_path, vocab_size=vocab_size)
    tokenizer = SentencePieceTokenizer(spm_model)
    model = _tiny_transformer_model(vocab_size)

    eval_cfg = ContinualEvalConfig(task_size=2, seed=0, train_per_label=2, eval_per_label=1)
    tasks = build_streaming_tasks(_toy_examples(), cfg=eval_cfg)

    memorize_cfg = MemorizeConfig(enabled=False)
    result, meta = evaluate_continual_classification(
        model,
        tokenizer,
        tasks,
        torch.device("cpu"),
        cfg=eval_cfg,
        memorize_cfg=memorize_cfg,
    )
    assert len(result.task_accuracy_matrix) == len(tasks)
    assert len(result.task_accuracy_matrix[0]) == len(tasks)
    assert 0.0 <= result.avg_accuracy_final <= 1.0
    assert "task_size" in meta


def test_evaluate_continual_classification_with_memorize_fast_state(tmp_path: Path) -> None:
    vocab_size = 64
    spm_model = _train_tiny_sentencepiece(tmp_path, vocab_size=vocab_size)
    tokenizer = SentencePieceTokenizer(spm_model)
    model = _tiny_transformer_model(vocab_size)

    eval_cfg = ContinualEvalConfig(task_size=2, seed=0, train_per_label=2, eval_per_label=1)
    tasks = build_streaming_tasks(_toy_examples(), cfg=eval_cfg)

    memorize_cfg = MemorizeConfig(enabled=True, steps=1, reset=False, use_fast_state=True)
    result, _meta = evaluate_continual_classification(
        model,
        tokenizer,
        tasks,
        torch.device("cpu"),
        cfg=eval_cfg,
        memorize_cfg=memorize_cfg,
    )
    assert len(result.per_task_forgetting) == len(tasks)
