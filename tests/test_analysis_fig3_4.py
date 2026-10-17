"""Figures 3 and 4: stage 1 computes what the research scripts computed, checked on small synthetic results roots
(tests/test_cotcontrol.py checks Figure 4b's computation; tests/test_figures.py checks stage 2 against the committed
data)."""

import json
import random
from collections import defaultdict
from pathlib import Path

import pytest

from analysis.compute import ablations, runs, transfer

ANALYSIS = Path(__file__).parents[1] / "analysis"
INTERMEDIATE = ANALYSIS / "data" / "intermediate"


def intermediate(name: str):
    return json.loads((INTERMEDIATE / f"{name}.json").read_text())


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_ablations_stage1_counts_known_episodes_and_records_checkpoints(tmp_path):
    verdicts = [True] * 6 + [False] * 3 + [None]  # compliance 6 / 9; the unfinished episode is left out
    for run in runs.DIRECT:
        for step in (0, 17, 34):
            write_jsonl(tmp_path / "train" / run / "eval" / f"step_{step:04d}.jsonl",
                        [{"turns": ["long transcript"], "grade": {"cot_compliant": v}} for v in verdicts])
        (tmp_path / "train" / run / "eval" / "step_0051.jsonl").write_text("past the figure's last step: never read")
        for step in runs.DIRECT_EVAL_STEPS:
            d = tmp_path / "rollouts" / runs.direct_eval(run, step)
            ckpt = step - 1 if (run, step) == (runs.DIRECT[2], 30) else step
            write_jsonl(d / "grades_v1.jsonl", [{"cot_compliant": v} for v in [True, True, False, None]])
            (d / "manifest.json").write_text(json.dumps({"status": "complete",
                                                         "config": {"model_path": f"tinker://x/{run}_step{ckpt:04d}"}}))
    out = ablations.compute(tmp_path)
    assert list(out) == runs.DIRECT
    first = out[runs.DIRECT[0]]
    assert [p["step"] for p in first] == [0, 3, 6, 9, 12, 15, 17, 18, 21, 24, 27, 30, 33, 34]
    assert first[0] == {"step": 0, "checkpoint": 0, "source": f"train/{runs.DIRECT[0]}/eval/step_0000.jsonl",
                        "episodes": 10, "known": 9, "compliant": 6}
    assert first[1] == {"step": 3, "checkpoint": 3, "source": f"rollouts/{runs.direct_eval(runs.DIRECT[0], 3)}",
                        "episodes": 4, "known": 3, "compliant": 2}
    assert next(p for p in out[runs.DIRECT[2]] if p["step"] == 30)["checkpoint"] == 29

    leakage = {f"train/{r}/eval/step_{s:04d}.jsonl": {"follow": s, "known": 40}
               for rs in ablations.SERIES.values() for r in rs for s in (0, 3, 34)}
    fig = ablations.figure(leakage, out)
    assert [p["name"] for p in fig] == [*ablations.SERIES, ablations.DIRECT]
    assert fig[0]["seeds"][0] == [[0, 0.0], [3, 7.5], [34, 85.0]]
    assert fig[-1]["seeds"][0][:2] == [[0, 66.7], [3, 66.7]] and fig[-1]["note"] == ablations.DIRECT_NOTE

    (tmp_path / "rollouts" / runs.direct_eval(runs.DIRECT[0], 3) / "manifest.json").write_text(
        json.dumps({"status": "running", "config": {"model_path": "x_step0003"}}))
    with pytest.raises(ValueError, match="running"):
        ablations.compute(tmp_path)


def research_stats(eps):
    """The research transfer script's statistics over (question, compliant, maths, rewarded) episodes, verbatim."""
    known = [e for e in eps if e[1] is not None]
    br = [e for e in known if not e[1]]
    brm = [e for e in br if e[2]]
    return {"n": len(eps), "compliance": 100 * sum(e[1] for e in known) / len(known), "maths": 100 * sum(e[2] for e in eps) / len(eps),
            "reward": 100 * sum(e[3] for e in eps) / len(eps), "leak_given_maths": 100 * sum(e[3] for e in brm) / len(brm) if brm else None,
            "leak_all": 100 * sum(e[3] for e in br) / len(br) if br else None, "breakers_maths_right": len(brm)}


def research_ci(eps, key, rng, b=1000):
    """The research transfer script's question bootstrap, verbatim."""
    by = defaultdict(list)
    for e in eps:
        by[e[0]].append(e)
    qs = list(by)
    vals = sorted(v for v in (research_stats([e for q in rng.choices(qs, k=len(qs)) for e in by[q]])[key] for _ in range(b)) if v is not None)
    return (round(vals[int(0.025 * len(vals))], 1), round(vals[int(0.975 * len(vals)) - 1], 1)) if vals else None


def test_transfer_stats_and_bootstrap_match_the_research_procedure():
    """Stage 1 sums per-question counts instead of resampling episode lists; same numbers, same random draws. Few
    rule-breakers with right maths, so some resamples leave leakage undefined."""
    make = random.Random(1)
    eps = [(f"q{make.randrange(15)}", make.choice([True, True, True, False, None]), make.random() < 0.2,
            make.random() < 0.5) for _ in range(120)]
    eps.append(("q15", True, True, True))  # every question has a known episode
    by = transfer.per_question(eps)
    assert transfer.stats(transfer.total(by.values())) == research_stats(eps)
    ours, theirs = random.Random(0), random.Random(0)
    for key in transfer.CI_KEYS:
        assert transfer.ci(by, key, ours, b=300) == research_ci(eps, key, theirs, b=300)
