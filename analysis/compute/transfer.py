"""Figure 4: the step-34 checkpoints of repeat-sentences GRPO seeds 1 and 2 on the four other rules, against the
untrained model.

Stage 1 (`--results`) writes transfer.json. Per rule and arm, every episode is regraded with the rule's current grader
(meow between words: grader v2) and summarised as
  - compliance: compliant / known (episodes whose `cot_compliant` is not None);
  - maths: turn-1 maths right / all episodes, judged leniently: the last `Answer: N` of turn 1's answer, else of its
    thinking, equals the gold answer;
  - reward: final answer correct / all episodes;
  - leak_given_maths: rewarded / rule-breakers (known, not compliant) with turn-1 maths right (None if there are none);
  - leak_all: rewarded / all rule-breakers;
  - breakers_maths_right: that denominator;
with 95% intervals for compliance, leak_given_maths and maths from 1,000 bootstrap resamples of the questions (all of
a question's episodes go together; resamples where the rate is undefined are dropped). One random.Random(0) is drawn
from in a fixed order (rules as in runs.TRANSFER_RULES, then arms, then those three rates), so the intervals are
deterministic. Arms: "untrained" pools the step-0 evals of the rule's four GRPO runs (seeds 0-3); "rs seed <s>" is
`rollouts/<transfer(s, rule)>`. Files are read one row at a time.

Stage 2 (analysis.compute.figures) writes Figure 4's data: per `<rule>|<arm>`, compliance and leak_given_maths with their intervals.

    uv run python -m analysis.compute.transfer --results <root> [--out analysis/data/intermediate/transfer.json]"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterable
from pathlib import Path

from analysis.compute import runs
from analysis.compute.leakage import lenient_maths
from src.tasks import Problem, task_from_spec

ANALYSIS = Path(__file__).parents[1]
INTERMEDIATE = ANALYSIS / "data" / "intermediate"
SEEDS = (1, 2)  # the repeat-sentences GRPO seeds whose step-34 checkpoints were evaluated on the other rules
BOOTSTRAP = 1000
CI_KEYS = ("compliance", "leak_given_maths", "maths")
FIGURE_KEYS = ("compliance", "compliance_ci", "leak_given_maths", "leak_given_maths_ci")

# per-question counts an episode adds to; every rate is a ratio of two of them
FIELDS = ("n", "known", "compliant", "maths", "rewarded", "breakers", "breakers_rewarded", "breakers_maths",
          "breakers_maths_rewarded")


# ---- stage 1 ----

def arms(results: Path, rule: str) -> dict[str, list[Path]]:
    return {"untrained": [results / "train" / run / "eval" / "step_0000.jsonl" for run in runs.GRPO[rule]],
            **{f"rs seed {s}": [results / "rollouts" / runs.transfer(s, rule) / "rollouts.jsonl"] for s in SEEDS}}


def episodes(paths: list[Path]) -> Iterable[tuple[str, bool | None, bool, bool]]:
    """(question id, compliant, turn-1 maths right, rewarded) per episode, regraded with the current grader."""
    task = spec = None
    for path in paths:
        with path.open() as f:
            for line in f:
                r = json.loads(line)
                if task is None:
                    spec, task = r["task"], task_from_spec(r["task"])
                elif r["task"] != spec:
                    raise ValueError(f"{path}: task {r['task']} differs from {spec}")
                g = task.grade(Problem.from_row(r["problem"]), r["turns"])
                yield r["problem"]["id"], g["cot_compliant"], lenient_maths(r), bool(g["correct"])


def per_question(eps: Iterable[tuple]) -> dict[str, list[int]]:
    """Question id -> counts (FIELDS order), questions in order of first appearance."""
    by: dict[str, list[int]] = {}
    for q, compliant, maths, rewarded in eps:
        c = by.setdefault(q, [0] * len(FIELDS))
        broke = compliant is not None and not compliant
        for i, v in enumerate((1, compliant is not None, bool(compliant), maths, rewarded, broke, broke and rewarded,
                               broke and maths, broke and maths and rewarded)):
            c[i] += v
    return by


def stats(counts: list[int]) -> dict:
    """The rates (in %) from summed counts."""
    c = dict(zip(FIELDS, counts, strict=True))
    brm = c["breakers_maths"]
    return {"n": c["n"], "compliance": 100 * c["compliant"] / c["known"], "maths": 100 * c["maths"] / c["n"],
            "reward": 100 * c["rewarded"] / c["n"],
            "leak_given_maths": 100 * c["breakers_maths_rewarded"] / brm if brm else None,
            "leak_all": 100 * c["breakers_rewarded"] / c["breakers"] if c["breakers"] else None,
            "breakers_maths_right": brm}


def total(rows: Iterable[list[int]]) -> list[int]:
    return [sum(col) for col in zip(*rows, strict=True)]


def ci(by: dict[str, list[int]], key: str, rng: random.Random, b: int = BOOTSTRAP) -> tuple[float, float] | None:
    """95% interval for `key` from `b` resamples of the questions with replacement: the sorted defined values' entries
    at int(0.025 * m) and int(0.975 * m) - 1 (m = number of defined values), rounded to 0.1."""
    qs = list(by)
    vals = sorted(v for v in (stats(total(by[q] for q in rng.choices(qs, k=len(qs))))[key] for _ in range(b))
                  if v is not None)
    return (round(vals[int(0.025 * len(vals))], 1), round(vals[int(0.975 * len(vals)) - 1], 1)) if vals else None


def compute(results: Path) -> dict:
    """`<rule>|<arm>` -> stats plus `<key>_ci` for CI_KEYS."""
    rng = random.Random(0)
    out = {}
    for rule in runs.TRANSFER_RULES:
        for arm, paths in arms(results, rule).items():
            by = per_question(episodes(paths))
            s = stats(total(by.values()))
            s.update({f"{k}_ci": ci(by, k, rng) for k in CI_KEYS})
            out[f"{rule}|{arm}"] = s
            print(rule, arm, {k: s[k] for k in ("n", "compliance", "compliance_ci", "leak_given_maths",
                                                "leak_given_maths_ci")}, flush=True)
    return out


# ---- stage 2 ----

def figure(transfer: dict) -> dict:
    """Figure 4's data: `<rule>|<arm>` -> compliance and leak_given_maths (%) with their 95% intervals."""
    return {key: {k: s[k] for k in FIGURE_KEYS} for key, s in transfer.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True, help="the results root to read")
    ap.add_argument("--out", type=Path, default=INTERMEDIATE / "transfer.json")
    a = ap.parse_args()
    a.out.write_text(json.dumps(compute(a.results), indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
