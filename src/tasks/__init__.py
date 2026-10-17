"""Problems and the task registry. Problem rows are flat: {id, split, answer, **inputs}."""

from __future__ import annotations

from dataclasses import dataclass

from src.tasks.math_gates import CotControlMathGate, CotControlWarmup, RepeatSentencesMathGate, WordSuppressionMathGate
from src.tasks.maths import MathQA
from src.tasks.repeat_sentences import RepeatSentencesDirectCorrect

TASKS = {task.name: task for task in (MathQA, RepeatSentencesDirectCorrect, RepeatSentencesMathGate, CotControlMathGate,
                                      CotControlWarmup, WordSuppressionMathGate)}
_RESERVED = {"id", "split", "answer"}


def task_from_spec(spec: dict):
    """The task a run recorded (its `spec()`): the name plus any dataclass fields."""
    return TASKS[spec["name"]](**{k: v for k, v in spec.items() if k != "name"})


@dataclass(frozen=True)
class Problem:
    id: str
    split: str
    inputs: dict
    answer: int | str

    def __post_init__(self) -> None:
        if _RESERVED & set(self.inputs):
            raise ValueError(f"problem inputs may not use the reserved keys {_RESERVED}")

    @classmethod
    def from_row(cls, row: dict) -> Problem:
        return cls(row["id"], row["split"], {k: v for k, v in row.items() if k not in _RESERVED}, row["answer"])

    def to_row(self) -> dict:
        return {"id": self.id, "split": self.split, **self.inputs, "answer": self.answer}
