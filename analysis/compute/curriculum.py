"""All five rules trained together (Figure 5).

Stage 1 reads each curriculum run's evals, `train/<run>/eval/step_<k>.jsonl` (every rule on its own 100 validation
questions x 25 samples), one line at a time, and regrades every episode with its rule's current grader (meow: v2). Per
rule (the curriculum component), run and eval step, in percent:
- compliance: episodes that followed the rule, over the episodes the grader could judge;
- maths: episodes whose turn-1 answer is correct, read leniently (the last `Answer: N` in the turn-1 reply, else the
  last one in the turn-1 thinking), over all episodes; reward: episodes whose final answer is correct, over all;
- leak_given_maths: rule-breakers with correct turn-1 maths whose final answer was correct, over those rule-breakers
  (`breakers_maths_right`); leak_all: rule-breakers whose final answer was correct, over all rule-breakers;
- single_compliance, single_leak_given_maths: the same two rates for that rule's single-rule GRPO runs at the same
  eval step, pooled over their `single_runs` seeds, from cond_leakage.json.
Keys are `<component>|<seed>|<step>`; `<component>|mean|<step>` holds each rate's mean over the seeds where it is
defined (`seeds` runs).

    uv run python -m analysis.compute.curriculum --results <root> --out analysis/data/intermediate/curriculum.json

Stage 2 (analysis.compute.figures), Figure 5, per rule: each run's [step, compliance, leak_given_maths], rounded to 0.1. Grey lines (`single_c`,
`single_l`): Figure 1's single-rule lines, i.e. at each eval step the mean over the 4 GRPO seeds of each seed's
compliance and leakage (given correct lenient turn-1 maths) rounded to 0.1, from cond_leakage.json; leakage averages
the seeds that had such a rule-breaker.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.compute.leakage import lenient_maths
from analysis.compute.runs import CURRICULUM, CURRICULUM_EVAL_STEPS, EVAL_EPISODES, GRPO, RULES, check_steps
from src.tasks import Problem, task_from_spec

ROOT = Path(__file__).resolve().parents[2]
INTERMEDIATE = ROOT / "analysis/data/intermediate"
# rule -> its component name in the curriculum (five_rules_v1 in src/training/curricula.py)
COMPONENT = {"repeat_sentences": "repeat_sentences", "lowercase_thinking": "lowercase", "ok_start_sentence": "ok_start",
             "word_suppression": "word_suppression", "meow_between_words": "meow"}
RULE = {c: r for r, c in COMPONENT.items()}
COUNTS = ("n", "maths", "rewarded", "known", "follow", "break", "break_rewarded", "break_maths", "break_maths_rewarded")
RATES = ("compliance", "maths", "reward", "leak_given_maths", "single_compliance", "single_leak_given_maths")


def pct(a: int, b: int) -> float | None:
    return 100 * a / b if b else None


def r1(v: float | None) -> float | None:
    return None if v is None else round(v, 1)


def compliance(c: dict) -> float | None:
    """From a cond_leakage.json record: rule-following over the episodes with a verdict."""
    return pct(c["follow"], c["known"])


def leakage(c: dict) -> float | None:
    """From a cond_leakage.json record: rule-breakers with correct lenient turn-1 maths whose final answer was correct,
    over those rule-breakers."""
    return pct(c["break_mathlen_rewarded"], c["break_mathlen"])


def count_eval(path: Path) -> dict[str, dict]:
    """Per component of one eval file, the counts behind every rate (episodes regraded with the current grader)."""
    counts, tasks = {}, {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            k = r["component"]
            if k not in tasks:
                tasks[k] = task_from_spec(r["task"])
            g = tasks[k].grade(Problem.from_row(r["problem"]), r["turns"])
            m, ok = lenient_maths(r), bool(g["correct"])
            x = counts.setdefault(k, dict.fromkeys(COUNTS, 0))
            x["n"] += 1
            x["maths"] += m
            x["rewarded"] += ok
            if g["cot_compliant"] is None:
                continue
            x["known"] += 1
            if g["cot_compliant"]:
                x["follow"] += 1
            else:
                x["break"] += 1
                x["break_rewarded"] += ok
                x["break_maths"] += m
                x["break_maths_rewarded"] += ok and m
    return counts


def single_rule(cond_leakage: dict, rule: str, step: int) -> dict:
    """The rule's single-rule GRPO runs at eval `step`, pooled over the seeds that have that eval."""
    runs = [c for run in GRPO[rule] if (c := cond_leakage.get(f"train/{run}/eval/step_{step:04d}.jsonl"))]
    pooled = {key: sum(c[key] for c in runs) for key in ("follow", "known", "break_mathlen_rewarded", "break_mathlen")}
    return {"single_compliance": compliance(pooled) if runs else None,
            "single_leak_given_maths": leakage(pooled) if runs else None, "single_runs": len(runs)}


def record(x: dict, single: dict) -> dict:
    return {"compliance": pct(x["follow"], x["known"]), "maths": pct(x["maths"], x["n"]),
            "reward": pct(x["rewarded"], x["n"]), "leak_given_maths": pct(x["break_maths_rewarded"], x["break_maths"]),
            "leak_all": pct(x["break_rewarded"], x["break"]), "n": x["n"], "breakers_maths_right": x["break_maths"],
            **single}


def add_means(out: dict) -> dict:
    """`out` with `<component>|mean|<step>` records: each rate averaged over the seeds where it is defined."""
    out = dict(out)
    steps = sorted({int(key.split("|")[2]) for key in out})
    for component in COMPONENT.values():
        for step in steps:
            per = [v for key, v in out.items() if key.split("|")[0] == component and key.split("|")[1] != "mean"
                   and key.split("|")[2] == str(step)]
            if not per:
                continue
            mean = {}
            for m in RATES:
                vals = [v[m] for v in per if v[m] is not None]
                mean[m] = sum(vals) / len(vals) if vals else None
            out[f"{component}|mean|{step}"] = {**mean, "seeds": len(per)}
    return out


def build(results: Path, cond_leakage: dict, runs: list[str] = CURRICULUM) -> dict:
    out = {}
    for seed, run in enumerate(runs):
        files = sorted((results / "train" / run / "eval").glob("step_*.jsonl"))
        if not files:
            raise SystemExit(f"no evals for {run} under {results}")
        for path in files:
            step = int(path.stem.removeprefix("step_"))
            for k, x in count_eval(path).items():
                out[f"{k}|{seed}|{step}"] = record(x, single_rule(cond_leakage, RULE[k], step))
                print(f"{run} step {step} {k}: {x}", flush=True)
    return add_means(out)


def single_line(cond_leakage: dict, runs: list[str], rate) -> list[list]:
    """Figure 1's mean line: per eval step, the mean of the runs' `rate` (each rounded to 0.1) where it is defined."""
    per_run = [{int(Path(k).stem.removeprefix("step_")): rate(c) for k, c in cond_leakage.items()
                if k.startswith(f"train/{run}/eval/")} for run in runs]
    line = []
    for step in sorted(set().union(*per_run)):
        vals = [round(p[step], 1) for p in per_run if p.get(step) is not None]
        if vals:
            line.append([step, round(sum(vals) / len(vals), 1)])
    return line


def fig5(curriculum: dict, cond_leakage: dict) -> list[dict]:
    out = []
    for rule, name in RULES.items():
        k, seeds = COMPONENT[rule], []
        for seed in range(len(CURRICULUM)):
            steps = sorted(int(key.split("|")[2]) for key in curriculum if key.startswith(f"{k}|{seed}|"))
            seeds.append([[s, r1(curriculum[f"{k}|{seed}|{s}"]["compliance"]),
                           r1(curriculum[f"{k}|{seed}|{s}"]["leak_given_maths"])] for s in steps])
        out.append({"name": name, "seeds": seeds,
                    "single_c": single_line(cond_leakage, GRPO[rule], compliance),
                    "single_l": single_line(cond_leakage, GRPO[rule], leakage)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True, help="the results root holding train/<run>/")
    ap.add_argument("--data", type=Path, default=INTERMEDIATE, help="the intermediates directory (cond_leakage.json)")
    ap.add_argument("--out", type=Path, default=INTERMEDIATE / "curriculum.json")
    a = ap.parse_args()
    check_steps(a.results, "eval", CURRICULUM, CURRICULUM_EVAL_STEPS)
    out = build(a.results, json.loads((a.data / "cond_leakage.json").read_text()))
    expected = {f"{c}|{seed}|{step}" for c in COMPONENT.values() for seed in range(len(CURRICULUM))
                for step in CURRICULUM_EVAL_STEPS}
    if missing := sorted(expected - {k for k in out if "|mean|" not in k}):
        raise SystemExit(f"no episodes for {len(missing)} (rule, seed, step) evals, e.g. {missing[:3]}")
    if short := sorted(k for k, x in out.items() if "|mean|" not in k and x["n"] != EVAL_EPISODES):
        raise SystemExit(f"evals without {EVAL_EPISODES} episodes per rule: {short}")
    a.out.write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
