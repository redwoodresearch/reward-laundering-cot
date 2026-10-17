"""Figure 4b: the Figure 4 models (untrained; repeat-sentences GRPO seeds 1 and 2 at step 34) on Chen et al.'s
CoT-Control-QA.

Stage 1 (`--results`) writes cotcontrol.json. Each model's run (runs.COTCONTROL, `cotcontrol/<run>`) must first pass
scripts/experiments/cotcontrol_checks.validated_run (the checks grading and the Docent export make) and cover the full
benchmark. Then, from its graded files, one per (dataset, mode) pair:
  - per mode: compliance = mean over the three datasets of the pair's compliance rate (the paper's macro average);
    accuracy likewise, from `correctness`; n = that mode's samples;
  - score: the same means over all 27 pairs (the paper's controllability score);
  - a pair's compliance rate = compliant / all its samples; a sample the grader left ungraded (compliance missing)
    counts as not compliant;
  - 95% interval: 2,000 bootstrap draws resampling each dataset's questions (its rows in the dataset file, matched to
    graded rows by their prompts) with replacement, the same draw for all nine modes, since the modes ask the same
    questions; a pair's rate in a draw averages its drawn questions (the word-suppression modes skip questions
    without keywords, as the repo's loader does). The statistic is recomputed per draw; the interval is its 2.5th and
    97.5th percentiles. One numpy default_rng(0) is drawn from in a fixed order (models as in runs.COTCONTROL,
    datasets sorted), so the intervals are deterministic;
  - cut_off_at_25k: samples that stopped at the token limit (`stop_reason` "length") / all samples in samples.jsonl.
Rates are in %; compliance and intervals rounded to 0.01, accuracy and cut_off_at_25k to 0.1. samples.jsonl is scanned
a line at a time for `stop_reason`.

Stage 2 (analysis.compute.figures) writes Figure 4b's data: per model and mode (and score), compliance with its interval.

    uv run python -m analysis.compute.cotcontrol --results <root> [--out analysis/data/intermediate/cotcontrol.json]"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.compute import runs

ANALYSIS = Path(__file__).parents[1]
INTERMEDIATE = ANALYSIS / "data" / "intermediate"
DRAWS = 2000
COLUMNS = ["question", "compliance", "correctness"]
STOP_REASON = re.compile(r'"stop_reason": "([^"\\]*)"')  # a top-level key: inside JSON strings every quote is escaped


# ---- stage 1 ----

def question_ids(manifest: dict, run_cceval, dataset: str, mode: str) -> dict[str, list[int]]:
    """prompt -> the dataset question ids (rows of the dataset file) it was built from, as the repo builds them."""
    c, out = manifest["config"], {}
    for q in run_cceval.load_evaluation_data(Path(c["cotcontrol_dir"]) / "datasets" / f"{dataset}.csv", False, mode):
        prompt = run_cceval.create_prompt_for_sample(q, mode, c["label"], False, "none", False, False, False)[1]
        out.setdefault(prompt, []).append(int(q["id"]))
    return out


def pairs(run: dict) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """(dataset, mode) -> (per-sample question id, compliance 0/1, correctness 0/1) of a validated run."""
    out = {}
    for (dataset, mode), f in sorted(run["graded"].items()):
        df = pd.read_csv(f, usecols=COLUMNS)
        ids = question_ids(run["manifest"], run["run_cceval"], dataset, mode)
        out[(dataset, mode)] = (np.array([ids[q].pop(0) for q in df["question"]]),
                                pd.to_numeric(df["compliance"], errors="coerce").fillna(0).to_numpy(float),
                                df["correctness"].to_numpy(float))
    return out


def cut_off(samples: Path) -> float:
    """% of samples whose `stop_reason` is "length"."""
    cut = []
    with samples.open() as f:
        for line in f:
            found = STOP_REASON.findall(line)
            cut.append((found[0] if len(found) == 1 else json.loads(line)["stop_reason"]) == "length")
    return round(100 * np.mean(cut), 1)


def pair_draws(ps: dict, rng: np.random.Generator) -> dict[tuple[str, str], np.ndarray]:
    """(dataset, mode) -> its compliance rate in each of DRAWS question resamples, one resample per dataset shared by
    its modes."""
    out = {}
    for dataset in sorted({d for d, _ in ps}):
        questions = np.unique(np.concatenate([ids for (d, _), (ids, _, _) in ps.items() if d == dataset]))
        weights = rng.multinomial(len(questions), np.full(len(questions), 1 / len(questions)), size=DRAWS)
        for (d, mode), (ids, comp, _) in ps.items():
            if d == dataset:
                w = weights[:, np.searchsorted(questions, ids)]
                out[(d, mode)] = (w @ comp) / w.sum(axis=1)
    return out


def summarize(run: dict, rng: np.random.Generator) -> dict:
    """One model's per-mode and score compliance, intervals, accuracy and n."""
    ps = pairs(run)
    draws = pair_draws(ps, rng)
    modes = {}
    for m in [*sorted({m for _, m in ps}), "score"]:
        ks = [k for k in ps if m == "score" or k[1] == m]
        point = 100 * np.mean([ps[k][1].mean() for k in ks])
        boot = 100 * np.mean([draws[k] for k in ks], axis=0)
        modes[m] = {"compliance": round(point, 2),
                    "ci": [round(float(np.percentile(boot, 2.5)), 2), round(float(np.percentile(boot, 97.5)), 2)],
                    "accuracy": round(100 * np.mean([ps[k][2].mean() for k in ks]), 1),
                    "n": int(sum(len(ps[k][1]) for k in ks))}
    return modes


def compute(results: Path) -> dict:
    """Figure label -> {model, modes, cut_off_at_25k}."""
    from scripts.experiments import cotcontrol_checks as checks

    rng = np.random.default_rng(0)
    out = {}
    for label, name in runs.COTCONTROL.items():
        run_dir = results / "cotcontrol" / name
        run = checks.validated_run(run_dir)
        if not checks.is_full(run["manifest"]):
            raise SystemExit(f"{run_dir}: not the full benchmark ({run['manifest']['config']}); the figure needs all of it")
        out[label] = {"model": run["manifest"]["config"]["label"], "modes": summarize(run, rng),
                      "cut_off_at_25k": cut_off(run_dir / "samples.jsonl")}
        score = out[label]["modes"]["score"]
        print(f"{label}: score {score['compliance']:.2f} {score['ci']}, cut off {out[label]['cut_off_at_25k']}%", flush=True)
    return out


# ---- stage 2 ----

def figure(cotcontrol: dict) -> dict:
    """Figure 4b's data: label -> mode (and score) -> compliance (%) with its 95% interval."""
    return {label: {m: {"compliance": v["compliance"], "ci": v["ci"]} for m, v in x["modes"].items()}
            for label, x in cotcontrol.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True, help="the results root to read")
    ap.add_argument("--out", type=Path, default=INTERMEDIATE / "cotcontrol.json")
    a = ap.parse_args()
    a.out.write_text(json.dumps(compute(a.results), indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
