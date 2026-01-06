from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class ClassificationExample:
    text: str
    label: str


@dataclass(frozen=True)
class LoadedClassificationDataset:
    name: str
    split: str
    examples: List[ClassificationExample]
    label_names: List[str]


def load_hf_classification_dataset(
    dataset: str,
    *,
    split: str,
    text_field: str,
    label_field: str,
    name: str | None = None,
    max_samples: int | None = None,
) -> LoadedClassificationDataset:
    """
    Load a HuggingFace `datasets` text classification dataset into a simple in-memory format.

    This is used by the Phase 4 continual-learning harness (CLINC/Banking/DBpedia).
    """
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "`datasets` dependency is required for continual classification."
        ) from exc

    ds = load_dataset(dataset, name=name, split=split)
    features = getattr(ds, "features", None)
    label_names: List[str] = []
    if features is not None and label_field in features:
        feature = features[label_field]
        if getattr(feature, "names", None) is not None:
            label_names = list(feature.names)

    examples: List[ClassificationExample] = []
    count = 0
    for row in ds:
        if max_samples is not None and count >= max_samples:
            break
        text = str(row[text_field])
        raw_label = row[label_field]
        if isinstance(raw_label, int) and label_names:
            label = label_names[raw_label]
        else:
            label = str(raw_label)
        examples.append(ClassificationExample(text=text, label=label))
        count += 1

    if not label_names:
        label_names = sorted({ex.label for ex in examples})

    return LoadedClassificationDataset(
        name=dataset if name is None else f"{dataset}:{name}",
        split=split,
        examples=examples,
        label_names=label_names,
    )


def load_clinc_oos(
    *,
    split: str = "test",
    max_samples: int | None = None,
) -> LoadedClassificationDataset:
    # HF dataset: "clinc_oos" with fields {"text", "intent"}.
    return load_hf_classification_dataset(
        "clinc_oos",
        split=split,
        text_field="text",
        label_field="intent",
        max_samples=max_samples,
    )


def load_banking77(
    *,
    split: str = "test",
    max_samples: int | None = None,
) -> LoadedClassificationDataset:
    # HF dataset: "banking77" with fields {"text", "label"}.
    return load_hf_classification_dataset(
        "banking77",
        split=split,
        text_field="text",
        label_field="label",
        max_samples=max_samples,
    )


def load_dbpedia14(
    *,
    split: str = "test",
    max_samples: int | None = None,
) -> LoadedClassificationDataset:
    # HF dataset: "dbpedia_14" with fields {"content", "label"}.
    return load_hf_classification_dataset(
        "dbpedia_14",
        split=split,
        text_field="content",
        label_field="label",
        max_samples=max_samples,
    )


def unique_labels(examples: Iterable[ClassificationExample]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for ex in examples:
        if ex.label in seen:
            continue
        seen.add(ex.label)
        ordered.append(ex.label)
    return ordered


def filter_examples_by_labels(
    examples: Sequence[ClassificationExample],
    *,
    allowed: set[str],
) -> List[ClassificationExample]:
    return [ex for ex in examples if ex.label in allowed]
