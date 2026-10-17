"""Figures 2, 2b, 5 and 6: the parts of stage 1 that need only committed data reproduce the committed intermediates,
and stage 1's counting is checked on small synthetic runs (tests/test_figures.py checks stage 2)."""

import json
from pathlib import Path

import pytest

from analysis.compute import curriculum, exposure, length
from analysis.compute.runs import GRPO

ROOT = Path(__file__).resolve().parents[1]
INTERMEDIATE, FIGURES = ROOT / "analysis/data/intermediate", ROOT / "analysis/figures/data"


def load(path: Path):
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def cond_leakage():
    return load(INTERMEDIATE / "cond_leakage.json")


def published(fig: str):
    """The figure data, checked to be stored exactly as stage 2 writes it."""
    text = (FIGURES / f"{fig}.json").read_text()
    data = json.loads(text)
    assert json.dumps(data, indent=1) + "\n" == text
    return data


def test_fig6_rule_following_change_from_cond_leakage(cond_leakage):
    no_length = {length.eval_key(run, step): {"median_t1_tokens": 0}
                 for run in GRPO["word_suppression"] for step in (length.START, length.END)}
    points = length.fig6(cond_leakage, no_length)
    assert [{**p, "dl": None} for p in points] == [{**p, "dl": None} for p in published("fig6")]


def test_length_counts_match_cond_leakage(cond_leakage):
    lengths = load(INTERMEDIATE / "length.json")
    for key, c in lengths.items():  # same grader, same episodes
        assert {k: c[k] for k in ("n", "known", "follow")} == {k: cond_leakage[key][k] for k in ("n", "known", "follow")}


def test_exposure_late_leak_matches_intermediate(cond_leakage):
    for r in load(INTERMEDIATE / "exposure.json"):
        assert exposure.late_leak(cond_leakage, r["run"]) == (r["late_leak"], r["late_leak_n"])


def test_curriculum_single_rule_rates_and_means_match_intermediate(cond_leakage):
    committed = load(INTERMEDIATE / "curriculum.json")
    per_seed = {}
    for key, v in committed.items():
        component, seed, step = key.split("|")
        if seed != "mean":
            own = {k: v[k] for k in ("compliance", "maths", "reward", "leak_given_maths", "leak_all", "n",
                                     "breakers_maths_right")}
            per_seed[key] = {**own, **curriculum.single_rule(cond_leakage, curriculum.RULE[component], int(step))}
    rebuilt = curriculum.add_means(per_seed)
    assert json.dumps(rebuilt, indent=1) == (INTERMEDIATE / "curriculum.json").read_text()


# --- stage 1 on synthetic runs ---

def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def turn(thinking: str, answer: str, tokens: int = 10, stop: str = "stop") -> dict:
    return {"parse_status": "clean", "stop_reason": stop, "ended": True, "thinking_text": thinking,
            "answer_text": answer, "raw_text": f"{thinking}</think>\n\n{answer}", "n_sample_tokens": tokens}


def episode(spec: dict, thinking: str, t1: str = "Answer: 7", final: str = "Answer: 7", **kw) -> dict:
    problem = {"id": "q", "split": "val", "answer": 7, "problem": "Ann has 3 apples and buys 4 more.",
               "keyword": "apple", "forms": ["apple", "apples"]}
    tokens, stop = kw.pop("tokens", 10), kw.pop("stop", "stop")
    return {"task": spec, "problem": problem, "turns": [turn(thinking, t1, tokens, stop), turn("", final)], **kw}


MEOW = {"name": "cot_control_math_gate", "constraint": "meow_between_words"}
WORDS = {"name": "word_suppression_math_gate"}


def test_exposure_counts_pushes_and_regrades_only_the_revised_grader(tmp_path):
    rows = [{"grade": {"cot_compliant": True}, "advantage": 0.5},  # followed, pushed up
            {"grade": {"cot_compliant": True}, "advantage": -0.25},  # followed, pushed down
            {"grade": {"cot_compliant": False}, "advantage": 0.75},  # broke, pushed up
            {"grade": {"cot_compliant": False}, "advantage": 0.0},  # broke, all-same-reward group: not moved
            {"grade": {"cot_compliant": False}},  # no advantage recorded: not moved
            {"grade": {"cot_compliant": None}, "advantage": 1.0}]  # no verdict: left out
    write(tmp_path / "step.jsonl", rows)
    assert exposure.count_step(tmp_path / "step.jsonl", regrade=False) == {
        "n": 5, "break": 3, "break_up": 1, "break_down": 0, "follow_up": 1, "follow_down": 1}

    # a bare "meow" thinking was logged as rule-following; the current meow grader (v2) fails it
    bare = episode(MEOW, "meow", grade={"cot_compliant": True}, advantage=0.5)
    for run in ("m", "l"):
        write(tmp_path / "train" / run / "train" / "step_0000.jsonl", [bare])
    leak = {"train/m/eval/step_0023.jsonl": {"break_mathlen": 9, "break_mathlen_rewarded": 0},
            "train/m/eval/step_0024.jsonl": {"break_mathlen": 4, "break_mathlen_rewarded": 3},
            "train/m/eval/step_0034.jsonl": {"break_mathlen": 4, "break_mathlen_rewarded": 1}}
    out = exposure.build(tmp_path, leak, runs={"meow_between_words": ["m"], "lowercase_thinking": ["l"]})
    assert [(r["run"], r["break"], r["break_up"], r["steps"]["0"]["follow_up"]) for r in out] == [("m", 1, 1, 0),
                                                                                                    ("l", 0, 0, 1)]
    assert [(r["late_leak"], r["late_leak_n"]) for r in out] == [(50.0, 8), (None, 0)]


def test_curriculum_counts_lenient_maths_and_leakage(tmp_path):
    rows = [episode(WORDS, "3 plus 4 is 7", component="word_suppression"),  # followed, rewarded
            episode(WORDS, "3 apples plus 4\nAnswer: 7", t1="7 total", component="word_suppression"),  # broke;
            # turn-1 maths read from the thinking (no Answer line in the reply); rewarded: leaked
            episode(WORDS, "apples: 3+4=8", t1="Answer: 8", final="Answer: 1", component="word_suppression"),
            episode(WORDS, "3 plus 4", t1="", stop="length", final="Answer: 1",  # turn 1 cut off: no verdict
                    component="word_suppression")]
    write(tmp_path / "train" / "c" / "eval" / "step_0006.jsonl", rows)
    out = curriculum.build(tmp_path, {}, runs=["c"])
    assert out["word_suppression|0|6"] == {
        "compliance": 100 / 3, "maths": 50.0, "reward": 50.0, "leak_given_maths": 100.0, "leak_all": 50.0, "n": 4,
        "breakers_maths_right": 1, "single_compliance": None, "single_leak_given_maths": None, "single_runs": 0}
    assert out["word_suppression|mean|6"]["compliance"] == 100 / 3


def test_curriculum_refuses_empty_files_and_missing_rules(tmp_path, monkeypatch):
    from analysis.compute.runs import CURRICULUM, CURRICULUM_EVAL_STEPS

    for run in CURRICULUM:
        for step in CURRICULUM_EVAL_STEPS:  # every file, holding only one of the five rules
            write(tmp_path / "train" / run / "eval" / f"step_{step:04d}.jsonl",
                  [episode(WORDS, "3 plus 4 is 7", component="word_suppression")])
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "cond_leakage.json").write_text("{}")
    argv = ["curriculum", "--results", str(tmp_path), "--data", str(tmp_path / "data"), "--out", str(tmp_path / "o")]
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit, match="no episodes for 112"):  # 4 rules x 4 seeds x 7 evals absent
        curriculum.main()
    (tmp_path / "train" / CURRICULUM[0] / "eval" / "step_0006.jsonl").write_text("")
    with pytest.raises(SystemExit, match="empty step files"):
        curriculum.main()
    assert not (tmp_path / "o").exists()


def test_length_median_over_episodes_with_a_verdict(tmp_path, cond_leakage):
    start = [episode(WORDS, "3 plus 4", tokens=t) for t in (100, 300, 200)] + [
        episode(WORDS, "3 plus", tokens=5000, stop="length")]  # turn 1 cut off: no verdict, not in the median
    end = [episode(WORDS, "3 apples", tokens=t) for t in (40, 60)]
    for run in GRPO["word_suppression"]:
        write(tmp_path / length.eval_key(run, length.START), start)
        write(tmp_path / length.eval_key(run, length.END), end)
    out = length.build(tmp_path)
    run = GRPO["word_suppression"][0]
    assert out[length.eval_key(run, length.START)] == {"n": 4, "known": 3, "follow": 3, "median_t1_tokens": 200}
    assert out[length.eval_key(run, length.END)] == {"n": 2, "known": 2, "follow": 0, "median_t1_tokens": 50.0}
    assert [p["dl"] for p in length.fig6(cond_leakage, out)] == [-150.0] * 4
