import json

import pytest

from src.dataset import pools
from src.rollouts import grading
from src.tasks.math_gates import RepeatSentencesMathGate
from src.tasks.maths import MathQA
from src.utils.files import write_jsonl


@pytest.mark.parametrize("name", sorted(pools.POOLS))
def test_every_committed_pool_is_what_its_recipe_builds(name):
    for split, problems in pools.build_pool(name).items():
        assert pools.load_pool(name, split) == problems  # load_pool also re-checks every answer


def _write_run(tmp_path, rows, task=None, status="complete"):
    (tmp_path / "rollouts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    manifest = {"status": status, "counts": {"rollouts": len(rows)}}
    if task:
        manifest["task"] = task
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))


def test_grading_uses_the_runs_recorded_task(tmp_path, monkeypatch):
    task = MathQA().spec()
    problem = {"id": "x", "split": "val", "problem": "6 x 7?", "gold": 42, "answer": 42}
    rows = [{"task": task, "rollout_id": "x-s0", "problem": problem,
             "turns": [{"answer_text": "Answer: 42", "parse_status": "clean", "stop_reason": "stop"}]},
            # truncated: the answer appears in raw text, but there is no committed answer to grade
            {"task": task, "rollout_id": "x-s1", "problem": problem,
             "turns": [{"answer_text": "", "parse_status": "unclean", "stop_reason": "length", "raw_text": "so Answer: 42"}]}]
    _write_run(tmp_path, rows, task, status="running")
    with pytest.raises(ValueError):
        grading.grade_run(tmp_path)  # unfinished run
    monkeypatch.setattr(pools, "POOLS", {})  # grading must not depend on today's pool definitions
    _write_run(tmp_path, [rows[0], {**rows[1], "problem": {**problem, "answer": 41}}], task)
    with pytest.raises(ValueError):
        grading.grade_run(tmp_path)  # inconsistent ground truth halfway through...
    assert not grading.grades_path(tmp_path, "math_qa_exact_v2").exists()  # ...publishes nothing
    _write_run(tmp_path, rows, task)
    summary = grading.grade_run(tmp_path)
    assert summary["grader"] == "math_qa_exact_v2"
    assert summary["counts"] == {"correct": 1, "incorrect": 0, "no_answer": 0, "unparsed_completion": 1}
    assert summary["accuracy"] == 0.5  # failures stay in the denominator
    assert set(grading.load_grades(tmp_path, summary["rollouts_sha256"], "math_qa_exact_v2")) == {"x-s0", "x-s1"}
    with pytest.raises(ValueError):
        grading.load_grades(tmp_path, "0" * 64, "math_qa_exact_v2")  # grades made from other rollouts
    with pytest.raises(FileExistsError):
        grading.grade_run(tmp_path)  # a grader version is written once
    path = grading.grades_path(tmp_path, "math_qa_exact_v2")
    path.write_text(path.read_text() * 2)
    with pytest.raises(ValueError):
        grading.load_grades(tmp_path, summary["rollouts_sha256"], "math_qa_exact_v2")  # duplicate ids


def test_a_run_without_one_agreed_task_is_refused(tmp_path):
    problem = {"id": "x", "split": "val", "problem": "6 x 7?", "gold": 42, "answer": 42}
    row = {"rollout_id": "x-s0", "problem": problem, "turns": [{"answer_text": "Answer: 42", "parse_status": "clean"}]}
    for rows, task in [([row], None), ([{**row, "task": MathQA().spec()}], RepeatSentencesMathGate().spec())]:
        _write_run(tmp_path, rows, task)
        with pytest.raises(ValueError):
            grading.grade_run(tmp_path)


def test_a_failed_jsonl_write_leaves_the_old_file_untouched(tmp_path):
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [{"a": 1}])
    with pytest.raises(TypeError):
        write_jsonl(path, [{"a": 2}, {"b": object()}])  # fails on the second row, after the first was written
    assert path.read_text() == '{"a": 1}\n'


def test_gsm8k_source_keeps_integer_answers_and_drops_train_copies_of_test_questions(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from scripts.data.build_math_sources import build_gsm8k

    (tmp_path / "main").mkdir()
    pq.write_table(pa.table({"question": ["Q test", "Q frac"], "answer": ["so #### 1,200", "#### 2.5"]}),
                   tmp_path / "main/test-00000-of-00001.parquet")
    pq.write_table(pa.table({"question": ["Q  test", "Q train"], "answer": ["#### 1200", "#### -3"]}),
                   tmp_path / "main/train-00000-of-00001.parquet")
    rows, counts = build_gsm8k(tmp_path)
    assert [(r["id"], r["gold"], r["split"] in ("train", "val")) for r in rows] == [
        ("gsm8k-test-0000", 1200, False), ("gsm8k-train-0001", -3, True)]  # commas stripped, test kept as test
    assert counts == {"non_integer_answer": 1, "train_copy_of_test": 1, "kept": 2}  # whitespace-insensitive copy
