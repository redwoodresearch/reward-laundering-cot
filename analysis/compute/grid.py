"""Stage 1, the run grid: per GRPO run of the post (the five rules, and repeat sentences with shuffled reward), from
its saved training batches and its eval counts (leakage.py, run first):

- train: per training step s, c = rule-following (% of the samples whose compliance could be judged) and l = the
  median turn-1 length (sampled tokens) of those samples.
- win: per 3-step window (steps w, w+1, w+2, labelled s = w + 1), over the groups (one question's samples) holding
  both rule-following and rule-breaking samples, the mean over groups (points; m and k with a 90% bootstrap band over
  groups):
  - g, the reward gap: the rule-followers' mean advantage minus the rule-breakers'. The advantage is the reward minus
    the group's mean reward, so this is the gap in reward rate (shuffled runs: the shuffled reward they trained on, for
    which the split below has no such reading; no figure uses their windows);
  - m (mlo, mhi), its maths part: the same gap in how often each kind knew the answer (`knew_answer`);
  - k (klo, khi), its gate part: the rest, g - m, i.e. how much more often rule-breakers than rule-followers knew the
    answer yet went unrewarded (withheld it at the gate);
  - q, the number of such groups. Windows with fewer than 3 are left out.
- late: the same split pooled over steps 7-33 (q, maths, gate; plain means, no band), and rewarded_t1_wrong_pct: the
  share of samples rewarded (on the real reward) after step 6 whose turn-1 answer alone reads wrong.
- eval: per eval step, leakage.eval_rows (rule-following, leakage given right turn-1 maths, its denominator, and
  leakage over all rule-breakers).

Episodes whose compliance cannot be judged are left out throughout, as are groups dropped from training after a failed
call (failed_groups counts them per run). Rules whose grader changed after training (meow between words, grader v2)
are classified with the current grader; the others use the grade logged at training time. Advantages are always the
ones the run trained with.

    uv run python -m analysis.compute.grid --results <root> --out analysis/data/intermediate/grid.json \
        [--leakage analysis/data/intermediate/cond_leakage.json]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from analysis.compute.leakage import eval_rows, read_jsonl, turn1_answer
from analysis.compute.runs import GRPO, RULES, SHUFFLED, TRAIN_STEPS, check_steps
from src.tasks import Problem, task_from_spec
from src.tasks.constraints import GRADER_VERSION

N_STEPS = 34  # training steps 0-33
LATE = 7  # "late" = steps 7-33
NUMBER = re.compile(r"(?<![\d.])-?\d+(?![\d.]*\d)")  # an integer not part of a decimal
# (run, rule, label), in the order the bootstrap draws from one shared RNG (seed 0)
RUNS = [*((run, "repeat_sentences", f"seed {s}") for s, run in enumerate(GRPO["repeat_sentences"])),
        *((run, "repeat_sentences", f"shuffled reward, seed {s}") for s, run in enumerate(SHUFFLED)),
        *((run, rule, f"seed {s}") for rule, runs in GRPO.items() if rule != "repeat_sentences"
          for s, run in enumerate(runs))]


def knew_answer(r: dict) -> tuple[bool, bool]:
    """(knew the answer, turn-1 answer right). Turn-1 answer: the last `Answer: N` of the turn-1 reply, else of the
    turn-1 thinking, else the last number anywhere in turn 1. Knew the answer: the turn-1 answer is right or the final
    answer is (the real outcome, also in shuffled runs), so every rewarded sample counts as knowing it. In a complete
    GRPO group the gate part g - m then equals the rule-breakers-minus-followers difference in "turn-1 answer right,
    final answer wrong": a descriptive proxy for withholding a known answer, not a measure of intent."""
    t1 = r["turns"][0]
    found, value = turn1_answer(t1)
    if not found:
        numbers = NUMBER.findall(t1["raw_text"].replace("<|im_end|>", "").replace(",", ""))
        value = int(numbers[-1]) if numbers else None
    t1_ok = value == r["problem"]["answer"]
    return t1_ok or bool(r["grade"]["correct"]), t1_ok


def step_stats(episodes: Iterable[dict], regrade: bool) -> dict:
    """One training step's batch: judged samples n, rule-following n_c, median turn-1 length len, real rewards rew,
    rewarded samples whose turn-1 answer reads wrong slip, and per group holding both kinds, (reward gap, maths part)
    in points (mixed). Groups dropped from training after a failed call (`group_status` "failed": their advantages are
    zeroed, so they hold no reward gap) are left out of everything and counted (failed_groups)."""
    groups, lens, n, n_c, rew, slip, task, failed = defaultdict(list), [], 0, 0, 0, 0, None, set()
    for r in episodes:
        if r.get("group_status") == "failed":
            failed.add(r["group"])
            continue
        if regrade:
            task = task or task_from_spec(r["task"])
            r["grade"] = task.grade(Problem.from_row(r["problem"]), r["turns"])
        c = r["grade"]["cot_compliant"]
        if c is None:
            continue
        knew, t1_ok = knew_answer(r)
        correct = bool(r["grade"]["correct"])
        n, n_c, rew, slip = n + 1, n_c + bool(c), rew + correct, slip + (correct and not t1_ok)
        lens.append(r["turns"][0]["n_sample_tokens"])
        groups[r["group"]].append((bool(c), r.get("advantage") or 0.0, knew))
    mixed = []
    for v in groups.values():
        fo, br = [x for x in v if x[0]], [x for x in v if not x[0]]
        if fo and br:
            mixed.append((100 * (sum(x[1] for x in fo) / len(fo) - sum(x[1] for x in br) / len(br)),
                          100 * (sum(x[2] for x in fo) / len(fo) - sum(x[2] for x in br) / len(br))))
    return {"rew": rew, "slip": slip, "n": n, "n_c": n_c, "len": statistics.median(lens) if lens else None,
            "mixed": mixed, "failed_groups": len(failed)}


def band(xs: list[float], rng: random.Random) -> tuple[float, float, float]:
    """Mean and 90% bootstrap band (2,000 resamples) of xs."""
    boot = sorted(sum(rng.choices(xs, k=len(xs))) / len(xs) for _ in range(2000))
    return round(sum(xs) / len(xs), 1), round(boot[100], 1), round(boot[1899], 1)


def run_row(run: str, rule: str, label: str, steps: list[tuple[int, dict]], evals: list[dict],
            rng: random.Random) -> dict:
    """One run's grid row from its per-step stats (sorted by step) and eval rows."""
    train = [{"s": s, "c": round(100 * v["n_c"] / v["n"], 1), "l": v["len"]} for s, v in steps]
    win = []
    for w in range(steps[0][0], N_STEPS, 3):
        m = [x for s, v in steps if w <= s < w + 3 for x in v["mixed"]]
        if len(m) < 3:
            continue
        g, _, _ = band([x[0] for x in m], rng)  # its band is drawn (and dropped) to keep the RNG sequence
        a, alo, ahi = band([x[1] for x in m], rng)
        k, klo, khi = band([x[0] - x[1] for x in m], rng)
        win.append({"s": w + 1, "g": g, "m": a, "mlo": alo, "mhi": ahi, "k": k, "klo": klo, "khi": khi, "q": len(m)})
    late = [x for s, v in steps if s >= LATE for x in v["mixed"]]
    slip, rew = (sum(v[key] for s, v in steps if s >= LATE) for key in ("slip", "rew"))
    return {"run": run, "rule": rule, "label": label, "failed_groups": sum(v["failed_groups"] for _, v in steps),
            "train": train, "win": win, "eval": evals,
            "late": {"q": len(late), "maths": round(sum(x[1] for x in late) / max(len(late), 1), 1),
                     "gate": round(sum(x[0] - x[1] for x in late) / max(len(late), 1), 1),
                     "rewarded_t1_wrong_pct": round(100 * slip / max(rew, 1), 1)}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True, help="results root holding train/<run>/train/")
    ap.add_argument("--out", type=Path, default=Path("analysis/data/intermediate/grid.json"))
    ap.add_argument("--leakage", type=Path, default=Path("analysis/data/intermediate/cond_leakage.json"),
                    help="leakage.py's output, for the eval rows")
    a = ap.parse_args()
    leakage = json.loads(a.leakage.read_text())
    check_steps(a.results, "train", (run for run, _, _ in RUNS), TRAIN_STEPS)
    rng, out = random.Random(0), []
    for run, rule, label in RUNS:
        files = sorted((a.results / "train" / run / "train").glob("step_*.jsonl"))
        steps = []
        for path in files:
            v = step_stats(read_jsonl(path), regrade=rule in GRADER_VERSION)
            steps.append((int(path.stem.removeprefix("step_")), v))
            print(f"{run} {path.name}: {v['n_c']}/{v['n']} rule-following, {len(v['mixed'])} mixed groups", flush=True)
        out.append(run_row(run, RULES[rule], label, steps, eval_rows(leakage, run), rng))
        late = out[-1]["late"]
        print(f"{RULES[rule]}, {label}: {len(out[-1]['win'])} windows, late maths {late['maths']} gate {late['gate']}",
              flush=True)
    a.out.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {a.out} ({len(out)} runs)")


if __name__ == "__main__":
    main()
