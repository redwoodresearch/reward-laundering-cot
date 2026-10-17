"""Word suppression: rule-following against chain-of-thought length (Figure 6).

Stage 1 reads the four word-suppression GRPO runs' evals at steps 6 and 34 (`train/<run>/eval/step_<k>.jsonl`, 100
validation questions x 25 samples), one line at a time, and grades every episode with the current grader, as
cond_leakage.json does. Per eval file (keyed like cond_leakage.json): `n` episodes, `known` of them with a verdict on
the rule, `follow` of those that followed it, and `median_t1_tokens`, the median turn-1 length (thinking and answer,
sampled tokens) over the `known` episodes.

    uv run python -m analysis.compute.length --results <root> --out analysis/data/intermediate/length.json

Stage 2 (analysis.compute.figures), Figure 6, one point per run: `dc`, the change in eval rule-following from step 6 to step 34 (points; from
cond_leakage.json), and `dl`, the change in median turn-1 length (tokens; from length.json), each rounded to 0.1.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from analysis.compute.runs import GRPO
from src.tasks import Problem, task_from_spec

ROOT = Path(__file__).resolve().parents[2]
INTERMEDIATE = ROOT / "analysis/data/intermediate"
RULE, START, END = "word_suppression", 6, 34


def eval_key(run: str, step: int) -> str:
    return f"train/{run}/eval/step_{step:04d}.jsonl"


def measure(path: Path) -> dict:
    """Episode counts and the median turn-1 length of one eval file."""
    n, follow, lengths, task = 0, 0, [], None
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            task = task or task_from_spec(r["task"])
            followed = task.grade(Problem.from_row(r["problem"]), r["turns"])["cot_compliant"]
            n += 1
            if followed is None:
                continue
            follow += followed
            lengths.append(r["turns"][0]["n_sample_tokens"])
    return {"n": n, "known": len(lengths), "follow": follow,
            "median_t1_tokens": statistics.median(lengths) if lengths else None}


def build(results: Path, runs: list[str] = GRPO[RULE]) -> dict:
    out = {}
    for run in runs:
        for step in (START, END):
            out[eval_key(run, step)] = c = measure(results / eval_key(run, step))
            print(f"{run} step {step}: {c}", flush=True)
    return out


def fig6(cond_leakage: dict, length: dict) -> list[dict]:
    out = []
    for seed, run in enumerate(GRPO[RULE]):
        start, end = cond_leakage[eval_key(run, START)], cond_leakage[eval_key(run, END)]
        dc = 100 * end["follow"] / end["known"] - 100 * start["follow"] / start["known"]
        dl = length[eval_key(run, END)]["median_t1_tokens"] - length[eval_key(run, START)]["median_t1_tokens"]
        out.append({"label": f"seed {seed}", "dl": round(float(dl), 1), "dc": round(dc, 1), "kind": "seed"})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True, help="the results root holding train/<run>/")
    ap.add_argument("--out", type=Path, default=INTERMEDIATE / "length.json")
    a = ap.parse_args()
    a.out.write_text(json.dumps(build(a.results), indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
