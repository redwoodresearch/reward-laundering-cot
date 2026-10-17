"""Integer-answer maths: the answer format every task here asks for, its parser and grader, and the plain maths task
(one turn, no rule) that the data builders sample when choosing keywords and warm-ups."""

from __future__ import annotations

import re

INTEGER_ANSWER_FORMAT = ("In your final response, output only one line, in exactly this format (an integer: digits with "
                         "an optional leading minus sign, no commas or units):\nAnswer: <number>")
OUTCOMES = ("correct", "incorrect", "no_answer", "unparsed_completion")
_FINAL_ANSWER = re.compile(r"Answer:\s*(-?\d+)")


def parse_final_answer(answer_text: str) -> int | None:
    """The integer on the final non-empty line, which must be exactly `Answer: <integer>` (the prompt's
    contract). Earlier lines and other integers are never used."""
    lines = [line.strip() for line in answer_text.splitlines() if line.strip()]
    match = _FINAL_ANSWER.fullmatch(lines[-1]) if lines else None
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:  # beyond Python's int-conversion digit limit: not a usable answer
        return None


def finished(turn: dict) -> bool:
    """A clean parse that stopped on a stop sequence and, for turns recorded with `ended`, on the episode's accepted
    end-of-turn token (some renderers also stop on other tokens, which end the episode unfinished)."""
    return turn["parse_status"] == "clean" and turn["stop_reason"] == "stop" and turn.get("ended", True)


def grade_integer(expected: int, turn: dict) -> tuple[str, int | None]:
    """(outcome, parsed answer). A turn that did not finish (unclean parse, or stopped at the length limit) is
    unparsed_completion."""
    if not finished(turn):
        return "unparsed_completion", None
    parsed = parse_final_answer(turn["answer_text"])
    if parsed is None:
        return "no_answer", None
    return ("correct" if parsed == expected else "incorrect"), parsed


class MathQA:
    """A GSM8K problem on its own (scripts/data/build_math_sources.py). `solve` returns the stored reference answer:
    a lookup, not an independent check."""

    name = "math_qa"
    grader = "math_qa_exact_v2"
    outcomes = OUTCOMES
    shown_fields = ()  # grade fields shown in Docent besides `reward` (= correct here)

    def spec(self) -> dict:
        return {"name": self.name}

    def solve(self, inputs: dict) -> int:
        return inputs["gold"]

    def describe(self) -> str:
        return "Single turn; `reward` is final-answer correctness, read from the last line only."

    def user_turns(self, inputs: dict) -> list[str]:
        return [f"{inputs['problem']}\n\n{INTEGER_ANSWER_FORMAT}\nDo not include steps or explanations in your final response."]

    def grade(self, problem, turns: list[dict]) -> dict:
        outcome, parsed = grade_integer(problem.answer, turns[-1])
        return {"outcome": outcome, "parsed_answer": parsed, "correct": outcome == "correct"}

    def reward(self, grade: dict) -> float:
        return float(grade["correct"])
