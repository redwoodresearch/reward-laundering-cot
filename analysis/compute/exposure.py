"""Which training episodes GRPO pushed up or down (Figures 2 and 2b).

Stage 1 reads each GRPO run's saved training batches, `train/<run>/train/step_<k>.jsonl`, one line at a time. Per run
and training step it counts the episodes whose rule-following the grader could judge (`n`; an unfinished turn 1 has no
verdict and is left out), how many of them broke the rule (`break`), and how many rule-following and rule-breaking
episodes were pushed up (advantage > 0) or down (advantage < 0). Episodes with advantage 0, i.e. in groups where every
sample got the same reward, moved nothing and are in neither. Rules whose grader was revised after the runs (meow:
v2) are regraded with the current grader; the others use the grade logged during training. Per run it also adds
`late_leak`: eval leakage at steps 24-34 pooled, the share of rule-breakers with correct turn-1 maths (lenient reading)
whose final answer was correct (`late_leak_n` of them), from cond_leakage.json.

    uv run python -m analysis.compute.exposure --results <root> --out analysis/data/intermediate/exposure.json

Stage 2 (analysis.compute.figures) turns exposure.json into the figure data:
- Figure 2, per rule: the 4 seeds pooled over 3-step windows of training steps; `n` episodes with a verdict, of which
  `fu`/`fd` rule-following and `bu`/`bd` rule-breaking were pushed up/down.
- Figure 2b, per run: `c`, the rule-following share of the episodes with a verdict after step 6 (steps 7-33); `up`,
  the rule-breakers pushed up after step 6; `late_leak` as above.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.compute.runs import GRPO, RULES, TRAIN_STEPS, check_steps
from src.tasks import Problem, task_from_spec
from src.tasks.constraints import GRADER_VERSION

ROOT = Path(__file__).resolve().parents[2]
INTERMEDIATE = ROOT / "analysis/data/intermediate"
LATE_EVAL = 24  # late_leak pools the evals from this step on
WINDOW = 3  # Figure 2: training steps per window
AFTER = 6  # Figure 2b: count training steps after this one


def count_step(path: Path, regrade: bool) -> dict:
    """Counts for one training step's batch; `regrade` replaces the logged grade with the current grader's."""
    c = dict.fromkeys(("n", "break", "break_up", "break_down", "follow_up", "follow_down"), 0)
    task = None
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            grade = r["grade"]
            if regrade:
                task = task or task_from_spec(r["task"])
                grade = task.grade(Problem.from_row(r["problem"]), r["turns"])
            followed, advantage = grade["cot_compliant"], r.get("advantage") or 0.0
            if followed is None:
                continue
            kind = "follow" if followed else "break"
            c["n"] += 1
            c["break"] += not followed
            if advantage > 0:
                c[f"{kind}_up"] += 1
            elif advantage < 0:
                c[f"{kind}_down"] += 1
    return c


def late_leak(cond_leakage: dict, run: str) -> tuple[float | None, int]:
    """Eval leakage pooled over `run`'s evals at steps >= LATE_EVAL, and how many rule-breakers it is over."""
    late = [c for k, c in cond_leakage.items()
            if k.startswith(f"train/{run}/eval/") and int(Path(k).stem.removeprefix("step_")) >= LATE_EVAL]
    leaked, n = sum(c["break_mathlen_rewarded"] for c in late), sum(c["break_mathlen"] for c in late)
    return (100 * leaked / n if n else None), n


def build(results: Path, cond_leakage: dict, runs: dict[str, list[str]] = GRPO) -> list[dict]:
    out = []
    for rule, rule_runs in runs.items():
        for seed, run in enumerate(rule_runs):
            files = sorted((results / "train" / run / "train").glob("step_*.jsonl"))
            if not files:
                raise SystemExit(f"no training batches for {run} under {results}")
            steps = {}
            for path in files:
                step = int(path.stem.removeprefix("step_"))
                steps[str(step)] = c = count_step(path, regrade=rule in GRADER_VERSION)
                print(f"{run} step {step}: {c}", flush=True)
            total = {k: sum(c[k] for c in steps.values()) for k in ("n", "break", "break_up", "break_down")}
            leak, leak_n = late_leak(cond_leakage, run)
            out.append({"rule": RULES[rule], "seed": seed, "run": run, "steps": steps, **total, "late_leak": leak,
                        "late_leak_n": leak_n})
    return out


def fig2(exposure: list[dict]) -> list[dict]:
    out = []
    for rule, runs in GRPO.items():
        steps = [(int(s), c) for r in exposure if r["run"] in runs for s, c in r["steps"].items()]
        wins = []
        for w in range(0, max(s for s, _ in steps) + 1, WINDOW):
            cs = [c for s, c in steps if w <= s < w + WINDOW]
            wins.append({"w": w, "n": sum(c["n"] for c in cs), "fu": sum(c["follow_up"] for c in cs),
                         "fd": sum(c["follow_down"] for c in cs), "bu": sum(c["break_up"] for c in cs),
                         "bd": sum(c["break_down"] for c in cs)})
        out.append({"name": RULES[rule], "wins": wins})
    return out


def fig2b(exposure: list[dict]) -> list[dict]:
    out = []
    for r in exposure:
        late = [c for s, c in r["steps"].items() if int(s) > AFTER]
        n, broke = sum(c["n"] for c in late), sum(c["break"] for c in late)
        out.append({"rule": r["rule"], "seed": r["seed"], "c": round(100 * (1 - broke / n), 2),
                    "up": sum(c["break_up"] for c in late), "late_leak": round(r["late_leak"], 1)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True, help="the results root holding train/<run>/")
    ap.add_argument("--data", type=Path, default=INTERMEDIATE, help="the intermediates directory (cond_leakage.json)")
    ap.add_argument("--out", type=Path, default=INTERMEDIATE / "exposure.json")
    a = ap.parse_args()
    check_steps(a.results, "train", (run for runs in GRPO.values() for run in runs), TRAIN_STEPS)
    a.out.write_text(json.dumps(build(a.results, json.loads((a.data / "cond_leakage.json").read_text()))))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
