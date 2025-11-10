from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.eval import zeroshot


def test_commonsenseqa_builder() -> None:
    sample = {
        "question": "Where would you most likely find a revolving door?",
        "choices": {
            "label": ["A", "B", "C"],
            "text": ["bank", "library", "garden"],
        },
        "answerKey": "B",
    }
    texts, target = zeroshot.build_commonsenseqa_texts(sample)
    assert len(texts) == 3
    assert target == 1
    assert "library" in texts[target]


def test_openbookqa_builder() -> None:
    sample = {
        "question_stem": "Plants need what to make food?",
        "choices": {
            "label": ["A", "B", "C", "D"],
            "text": ["sunlight", "soil", "wind", "music"],
        },
        "answerKey": "A",
    }
    texts, target = zeroshot.build_openbookqa_texts(sample)
    assert len(texts) == 4
    assert target == 0
    assert "sunlight" in texts[target]
