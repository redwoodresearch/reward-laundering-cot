"""Figure 3: eval rule-following by training step for the repeat-sentences ablations and the direct-reward control.

Stage 1 (`--results`) counts the direct-reward control's eval compliance, per run and step, into ablations.json:
  - steps 0, 17 and 34: the training run's own evals, `train/<run>/eval/step_<kkkk>.jsonl`, from each row's stored
    `grade`;
  - steps 3, 6, ..., 33: separate evals of the run's checkpoints, `rollouts/<direct_eval(run, k)>/grades_*.jsonl`.
    Each is plotted at its scheduled step; `checkpoint` records the step whose weights were evaluated (two points
    used the nearest earlier saved checkpoint).
  Compliance = compliant / known, where known = episodes whose `cot_compliant` is not None (an episode whose thinking
  did not finish is unknown and left out), as the trainer's eval compliance (src/tasks/gate.py compliance_rate).

Stage 2 (analysis.compute.figures) writes Figure 3's data: the GRPO, shuffled-reward and SFT series from cond_leakage.json (eval
compliance regraded with the current strict graders: follow / known) and the direct-reward series from
ablations.json, each point rounded to 0.1 percentage points.

    uv run python -m analysis.compute.ablations --results <root> [--out analysis/data/intermediate/ablations.json]"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from analysis.compute import runs

ANALYSIS = Path(__file__).parents[1]
INTERMEDIATE = ANALYSIS / "data" / "intermediate"
LAST_STEP = 34  # the figure shows steps 0-34 (one direct-reward run trained on to step 68)
TRAINER_EVAL_STEPS = (0, 17, 34)  # the direct-reward runs' own evals within those steps

# figure panel -> runs (seeds 0-3); the first four read cond_leakage.json, the direct-reward panel ablations.json
SERIES = {"GRPO (reward laundering)": runs.GRPO["repeat_sentences"], "Shuffled reward": runs.SHUFFLED,
          "On-policy SFT": runs.SFT_ON_POLICY, "Base-rollout SFT": runs.SFT_BASE_ROLLOUTS}
DIRECT = "Direct compliance reward"
DIRECT_NOTE = "multiple-choice task"


# ---- stage 1 ----

def count(grades) -> dict:
    """Episodes, known (cot_compliant not None) and compliant among an iterable of grade dicts."""
    c = {"episodes": 0, "known": 0, "compliant": 0}
    for g in grades:
        c["episodes"] += 1
        if g["cot_compliant"] is not None:
            c["known"] += 1
            c["compliant"] += bool(g["cot_compliant"])
    return c


def trainer_eval(path: Path) -> dict:
    """Counts over a training run's eval file, read one row at a time (rows hold full transcripts)."""
    with path.open() as f:
        return count(json.loads(line)["grade"] for line in f)


def separate_eval(eval_dir: Path) -> tuple[int, dict]:
    """(checkpoint step, counts) for a separate eval run: its one graded file, the checkpoint from its manifest."""
    manifest = json.loads((eval_dir / "manifest.json").read_text())
    if not manifest["status"].startswith("complete"):
        raise ValueError(f"{eval_dir.name}: status {manifest['status']!r}")
    (grades,) = eval_dir.glob("grades_*.jsonl")
    checkpoint = int(re.search(r"_step(\d+)$", manifest["config"]["model_path"]).group(1))
    with grades.open() as f:
        return checkpoint, count(json.loads(line) for line in f)


def compute(results: Path) -> dict:
    """run -> [{step, checkpoint, source, episodes, known, compliant}] sorted by step, for the direct-reward runs."""
    out = {}
    for run in runs.DIRECT:
        points = []
        for step in TRAINER_EVAL_STEPS:  # named files: a missing one is an error, not a shorter curve
            path = results / "train" / run / "eval" / f"step_{step:04d}.jsonl"
            points.append({"step": step, "checkpoint": step, "source": str(path.relative_to(results)),
                           **trainer_eval(path)})
            print(run, points[-1], flush=True)
        for step in runs.DIRECT_EVAL_STEPS:
            name = runs.direct_eval(run, step)
            checkpoint, counts = separate_eval(results / "rollouts" / name)
            points.append({"step": step, "checkpoint": checkpoint, "source": f"rollouts/{name}", **counts})
            print(run, points[-1], flush=True)
        steps = [p["step"] for p in points]
        if len(set(steps)) != len(steps):
            raise ValueError(f"{run}: more than one eval at a step: {sorted(steps)}")
        out[run] = sorted(points, key=lambda p: p["step"])
    return out


# ---- stage 2 ----

def pct(k: int, n: int) -> float:
    return round(100 * k / n, 1)


def leakage_series(cond_leakage: dict, run: str) -> list[list]:
    """[[step, compliance %]] from cond_leakage.json's per-eval-file counts (`train/<run>/eval/step_<kkkk>.jsonl`)."""
    pattern = re.compile(rf"train/{re.escape(run)}/eval/step_(\d+)\.jsonl")
    points = sorted((int(m.group(1)), c) for key, c in cond_leakage.items() if (m := pattern.fullmatch(key)))
    if not points:
        raise KeyError(f"no evals of {run} in cond_leakage.json")
    return [[step, pct(c["follow"], c["known"])] for step, c in points if step <= LAST_STEP]


def figure(cond_leakage: dict, ablations: dict) -> list[dict]:
    """Figure 3's data: per panel, its name and one [[step, compliance %], ...] series per seed."""
    panels = [{"name": name, "seeds": [leakage_series(cond_leakage, r) for r in rs]} for name, rs in SERIES.items()]
    direct = [[[p["step"], pct(p["compliant"], p["known"])] for p in ablations[r] if p["step"] <= LAST_STEP]
              for r in runs.DIRECT]
    return [*panels, {"name": DIRECT, "seeds": direct, "note": DIRECT_NOTE}]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True, help="the results root to read")
    ap.add_argument("--out", type=Path, default=INTERMEDIATE / "ablations.json")
    a = ap.parse_args()
    a.out.write_text(json.dumps(compute(a.results), indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
