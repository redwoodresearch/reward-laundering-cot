"""Figure 1's stage-1 definitions on hand-made rows (tests/test_figures.py checks stage 2 against the committed data)."""

import json
import random

import pytest

from analysis.compute import maths
from analysis.compute.grid import knew_answer, run_row, step_stats
from analysis.compute.leakage import CELLS, add, eval_rows, lenient_maths
from analysis.compute.maths import episode, summarise


def _sample(group, follows, correct, t1, advantage, gold=7):
    t1 = {"answer_text": t1, "raw_text": f"{t1}</think>{t1}", "n_sample_tokens": 10}
    return {"group": group, "advantage": advantage, "problem": {"answer": gold}, "turns": [t1],
            "grade": {"cot_compliant": follows, "correct": correct}}


def test_reward_gap_splits_into_a_maths_part_and_a_gate_part():
    batch = [_sample(0, True, True, "Answer: 7", 0.5), _sample(0, True, True, "Answer: 7", 0.5),
             _sample(0, False, False, "Answer: 7", -0.5),  # knew the answer, withheld it
             _sample(0, False, False, "so 3", -0.5),  # did not know it (no Answer line: the last number, 3)
             _sample(0, None, False, "Answer: 7", 0.0),  # compliance not judged: left out
             _sample(1, True, True, "Answer: 7", 0.0)]  # a group of rule-followers only holds no gap
    stats = step_stats(batch, regrade=False)
    assert (stats["n"], stats["n_c"], stats["mixed"]) == (5, 3, [(100.0, 50.0)])  # gap 100 = maths 50 + gate 50
    row = run_row("r", "Rule", "seed 0", [(s, stats) for s in range(3)], [], random.Random(0))
    assert row["win"] == [{"s": 1, "g": 100.0, "m": 50.0, "mlo": 50.0, "mhi": 50.0, "k": 50.0, "klo": 50.0,
                           "khi": 50.0, "q": 3}]


def test_leakage_is_conditional_on_right_turn1_maths():
    turn = {"answer_text": "I withhold it.", "raw_text": "Answer: 12</think>I withhold it."}
    assert lenient_maths({"turns": [turn], "problem": {"answer": 12}})  # an Answer line in the thinking counts
    c = dict.fromkeys(CELLS, 0)
    for follows, correct, math_ok in [(False, True, True), (False, False, True), (False, False, False),
                                      (True, True, True), (None, True, True)]:
        add(c, {"cot_compliant": follows, "correct": correct, "math_correct": math_ok}, lenient=math_ok)
    # rule-following 1/4 judged; leakage 1 of the 2 rule-breakers with right maths; 1 of all 3 rule-breakers
    assert eval_rows({"train/r/eval/step_0003.jsonl": c, "train/other/eval/step_0003.jsonl": c}, "r") == [
        {"s": 3, "c": 25.0, "lk": 50.0, "lm": 2, "la": 33.3}]


def test_turn1_maths_counts_unresolved_episodes_as_not_correct():
    def ep(i, reply):
        turns = [{"answer_text": reply, "thinking_text": ""}] if reply is not None else []
        return episode("Rule", 0, "r", 3, {"rollout_id": str(i), "problem": {"answer": 7}, "turns": turns})

    rows = [ep(0, "Answer: 7"), ep(1, "It is 7 apples."), ep(2, "Seven."), ep(3, None)]
    extracted = {"r/3/1": {"key": "r/3/1", "value": 7, "evidence_found": True, "parse_ok": True},
                 "r/3/2": {"key": "r/3/2", "value": 7, "evidence_found": False, "parse_ok": True}}
    point = summarise(rows, extracted, {})["runs"][0]["points"][0]
    assert [r["purpose"] for r in rows[1:3]] == ["unresolved", "unresolved"]
    assert (point["parser"], point["llm"], point["llm_evidence_not_found"], point["no_turn1"]) == (1, 1, 1, 1)
    assert (point["correct_pct"], point["unresolved_pct"]) == (50.0, 50.0)


@pytest.mark.parametrize("reply, gold, right", [
    ("Answer: 12", 12, True), ("Answer: 12.", 12, True), ("Answer: 1,200", 1200, True), ("Answer: 1,200", 1, False),
    ("Answer: 35.5", 35, False), ("Answer: 7/2", 7, False), ("Answer: 3e2", 3, False), ("Answer: 5-6", 5, False),
    ("Answer: 7 / 2", 7, False), ("Answer: 5 - 6", 5, False), ("Answer: 7+2", 7, False), ("Answer: 2**3", 2, False),
    ("Answer: 1,2", 1, False), ("Answer: 12 - the total", 12, True),
    ("**Answer: 12**\n1. check", 12, True),  # markdown bold, then a numbered line: not 12**1
    ('so "Answer: 45".', 45, True), ("Answer: 45 apples", 45, True), ("Answer: 1,200,000.", 1200000, True),
    ("Answer: 7/2, so Answer: 7", 7, True),  # the last Answer line is read...
    ("Answer: 7, so Answer: 7/2", 7, False),  # ...even when it is not a plain number
])
def test_turn1_answers_are_read_as_whole_numbers(reply, gold, right):
    turn = {"answer_text": reply, "raw_text": f"Answer: {gold}</think>{reply}"}  # the thinking's line is not used
    assert lenient_maths({"turns": [turn], "problem": {"answer": gold}}) is right
    sample = {"turns": [turn], "problem": {"answer": gold}, "grade": {"correct": False}}
    assert knew_answer(sample) == (right, right)  # the reward gap's reading; its last-number fallback needs no Answer


def test_groups_dropped_after_a_failed_call_are_left_out_of_the_gap():
    failed = [{**_sample(0, follows, follows, "so 3", 0.0), "group_status": "failed"} for follows in (True, False)]
    stats = step_stats([*failed, _sample(1, True, True, "Answer: 7", 0.5), _sample(1, False, False, "Answer: 7", -0.5)],
                       regrade=False)
    assert (stats["n"], stats["failed_groups"], stats["mixed"]) == (2, 1, [(100.0, 0.0)])  # both knew: all gate


def test_a_cached_extraction_counts_only_for_the_saved_input_it_was_made_from(tmp_path, monkeypatch):
    monkeypatch.setattr(maths, "GRPO", {"lowercase_thinking": ["r"]})
    (tmp_path / "train/r/eval").mkdir(parents=True)
    saved = ("q", "The result is seven.", "Answer: 7")  # the input the cached reply was made from
    cached = {"key": "r/3/0", "value": 7, "evidence": "The result is seven.", "parse_ok": True, "text_transform": None}

    def point(thinking, reply):
        (tmp_path / "train/r/eval/step_0003.jsonl").write_text(json.dumps(
            {"rollout_id": "0", "problem": {"answer": 7, "problem": "q"},
             "turns": [{"thinking_text": thinking, "answer_text": reply}]}) + "\n")
        cache = {"r/3/0": dict(cached)}
        rows = maths.episodes(tmp_path, None, (cache,), {"r/3/0": maths.input_digest(*saved)})
        return summarise(rows, cache, {})["runs"][0]["points"][0]

    assert point(*saved[1:])["correct_pct"] == 100.0
    # a changed text that still holds the cached quote is not the input the reply was about
    changed = point("The result is seven. Correction: that was the starting amount; the final count is nine.",
                    "I got 9 apples.")
    assert (changed["llm_input_changed"], changed["correct_pct"]) == (1, 0.0)
    meow = {"thinking_text": "meow so meow seven", "answer_text": "Answer: meow 7"}
    assert maths.quote_found({"evidence": "so seven", "text_transform": "demeow"}, meow)  # as the retry read it
    assert not maths.quote_found({"evidence": "so seven", "text_transform": None}, meow)
