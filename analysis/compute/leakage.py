"""Stage 1: rule-following and leakage counts for every eval of the post's training runs (the GRPO runs on the five
rules and the repeat-sentences ablations: shuffled reward, on-policy SFT, base-rollout SFT).

Every episode is regraded with its task's current grader (strict compliance; meow between words with grader v2). For
each eval file, keyed `train/<run>/eval/step_<k>.jsonl`, the output holds the count of every cell a rate needs:

- n: episodes; rewarded: their final answer earned the reward; math: turn-1 maths right by the grader's strict reading;
  math_lenient: turn-1 maths right by the lenient reading below;
- known: episodes whose compliance could be judged; follow / break: those that followed / broke the rule;
  follow_rewarded, break_rewarded: of those, the rewarded ones;
- break_math, break_math_rewarded: rule-breakers whose turn-1 maths was right (strict reading), and of those, rewarded;
- break_mathlen, break_mathlen_rewarded: the same with the lenient reading.

Lenient turn-1 maths: the last `Answer: N` in the turn-1 reply, or, if the reply has none, the last one inside the
turn-1 thinking, equals the gold answer (`answer_value`: N is read whole, with thousands commas and decimals, so
`Answer: 1,200` is 1200 and `Answer: 35.5` is not 35; a fraction or other expression such as `7/2` reads as no
answer). `eval_rows` turns the counts into the rates the figures plot.

    uv run python -m analysis.compute.leakage --results <root> --out analysis/data/intermediate/cond_leakage.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

from analysis.compute.runs import (
    EVAL_EPISODES,
    EVAL_STEPS,
    GRPO,
    SFT_BASE_ROLLOUTS,
    SFT_ON_POLICY,
    SHUFFLED,
    check_steps,
)
from src.tasks import Problem, task_from_spec

RUNS = [*(run for runs in GRPO.values() for run in runs), *SHUFFLED, *SFT_ON_POLICY, *SFT_BASE_ROLLOUTS]
CELLS = ("n", "known", "follow", "follow_rewarded", "break", "break_rewarded", "break_math", "break_math_rewarded",
         "break_mathlen", "break_mathlen_rewarded", "rewarded", "math", "math_lenient")
# each `Answer:` followed by a number: the number (digits, commas, a decimal part) and any continuation that makes it
# part of a larger expression: an operator and another number on the same line (spaces allowed), or an exponent
ANSWER = re.compile(r"answer:\s*(-?\d[\d,]*(?:\.\d+)?)((?:[ \t]*(?:\*\*|[-+*/^×÷–])[ \t]*|[eE][+-]?)\d)?",
                    re.IGNORECASE)
NUMBER = re.compile(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")  # a plain number: thousands commas, decimals


def read_jsonl(path: Path) -> Iterator[dict]:
    """The records of a JSONL file, one line at a time."""
    with open(path) as f:
        for line in f:
            yield json.loads(line)


def answer_value(text: str) -> tuple[bool, float | None]:
    """(whether `text` has an `Answer:` followed by a number, the last one's value). The number is read whole, up to
    whatever follows it (a quote, full stop, unit); one that continues into an expression (`7/2`, `5 - 6`, `2**3`,
    `3e2`) or is not a plain number (`1,2`) has no value."""
    found = ANSWER.findall(text)
    if not found:
        return False, None
    number, continued = found[-1]
    number = number.rstrip(",")
    return True, float(number.replace(",", "")) if not continued and NUMBER.fullmatch(number) else None


def turn1_answer(t1: dict) -> tuple[bool, float | None]:
    """answer_value of the turn-1 reply, else, if the reply has no `Answer: N`, of the turn-1 thinking."""
    found, value = answer_value(t1.get("answer_text") or "")
    return (found, value) if found else answer_value(t1["raw_text"].split("</think>")[0])


def lenient_maths(r: dict) -> bool:
    """Turn-1 maths right, leniently: the last `Answer: N` of the turn-1 reply, else of the turn-1 thinking, is gold."""
    return turn1_answer(r["turns"][0])[1] == r["problem"]["answer"]


def add(c: dict[str, int], grade: dict, lenient: bool) -> None:
    """Adds one episode, given its grade and its lenient turn-1 maths, to the counts `c`."""
    rewarded, strict = bool(grade["correct"]), grade["math_correct"] is True
    c["n"] += 1
    c["rewarded"] += rewarded
    c["math"] += strict
    c["math_lenient"] += lenient
    if grade["cot_compliant"] is None:
        return
    c["known"] += 1
    if grade["cot_compliant"]:
        c["follow"] += 1
        c["follow_rewarded"] += rewarded
    else:
        c["break"] += 1
        c["break_rewarded"] += rewarded
        c["break_math"] += strict
        c["break_math_rewarded"] += rewarded and strict
        c["break_mathlen"] += lenient
        c["break_mathlen_rewarded"] += rewarded and lenient


def count(episodes: Iterable[dict]) -> dict[str, int]:
    """The counts for one eval file's episodes, each regraded with its task's current grader."""
    c, task = dict.fromkeys(CELLS, 0), None
    for r in episodes:
        task = task or task_from_spec(r["task"])
        add(c, task.grade(Problem.from_row(r["problem"]), r["turns"]), lenient_maths(r))
    return c


def pct(k: int, n: int) -> float | None:
    return round(100 * k / n, 1) if n else None


def eval_rows(leakage: dict[str, dict], run: str) -> list[dict]:
    """Per eval step s of `run`, in %: c = rule-following (follow / known); lk = leakage, the share of rule-breakers
    with right turn-1 maths (lenient) that still got the reward (break_mathlen_rewarded / break_mathlen, None when
    there are none) and lm = its denominator; la = leakage over all rule-breakers (break_rewarded / break)."""
    return [{"s": int(Path(key).stem.removeprefix("step_")), "c": pct(c["follow"], c["known"]),
             "lk": pct(c["break_mathlen_rewarded"], c["break_mathlen"]), "lm": c["break_mathlen"],
             "la": pct(c["break_rewarded"], c["break"])}
            for key, c in sorted(leakage.items()) if key.startswith(f"train/{run}/eval/")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True, help="results root holding train/<run>/eval/")
    ap.add_argument("--out", type=Path, default=Path("analysis/data/intermediate/cond_leakage.json"))
    a = ap.parse_args()
    check_steps(a.results, "eval", RUNS, EVAL_STEPS)
    files = [a.results / "train" / run / "eval" / f"step_{s:04d}.jsonl" for run in RUNS for s in EVAL_STEPS]
    out = {}
    for i, path in enumerate(files, 1):
        key = str(path.relative_to(a.results))
        out[key] = count(read_jsonl(path))
        if out[key]["n"] != EVAL_EPISODES:
            raise SystemExit(f"{key}: {out[key]['n']} episodes, expected {EVAL_EPISODES}")
        print(f"{i}/{len(files)} {key}: {out[key]}", flush=True)
    a.out.write_text(json.dumps(out, indent=0))
    print(f"wrote {a.out} ({len(out)} eval files)")


if __name__ == "__main__":
    main()
