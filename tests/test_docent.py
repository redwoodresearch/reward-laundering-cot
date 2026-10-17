"""Docent export, offline (Docent is a dev dependency): rollout-row conversion and the training-run streamer's
receipts."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts.docent import stream as docent_stream
from scripts.docent.export import suite_agent_runs, to_agent_run, verify_names
from src.tasks.math_gates import RepeatSentencesMathGate
from src.tasks.maths import MathQA
from src.tasks.repeat_sentences import RepeatSentencesDirectCorrect

TURN = {"prompt_text": "<|im_start|>user\nAdd<|im_end|>\n<|im_start|>assistant\n<think>\n",
        "raw_text": "sum it\n</think>\n\nAnswer: 300<|im_end|>", "parse_status": "clean", "stop_reason": "stop",
        "n_sample_tokens": 2}
CLEAN = {
    "rollout_id": "p-s0", "model": "m", "status": "complete",
    "problem": {"id": "p", "split": "eval", "a": 100, "b": 200, "answer": 300},
    "normalized_messages": [{"role": "system", "content": "Reasoning effort"}, {"role": "user", "content": "Add 100 + 200"}],
    "turns": [TURN],
}
TRUNCATED = {**CLEAN, "rollout_id": "p-s1", "status": "incomplete",
             "turns": [{**TURN, "raw_text": "still summing", "parse_status": "unclean", "stop_reason": "length"}]}


@pytest.mark.parametrize(("row", "text"), [
    (CLEAN, "<think>\nsum it\n</think>\n\nAnswer: 300"),  # the chat template's assistant body, no turn delimiter
    (TRUNCATED, "<think>\nstill summing"),  # shown as sampled
])
def test_transcript_matches_chat_template_and_metadata_is_minimal(row, text):
    run = to_agent_run(row)
    system, user, assistant = run.transcripts[0].messages
    assert (system.role, user.role, assistant.role, assistant.text) == ("system", "user", "assistant", text)
    assert run.name == row["rollout_id"]
    turn = row["turns"][0]
    minimal = {"problem_id", "correct_answer", "parse_status", "stop_reason", "n_completion_tokens"}
    assert set(run.metadata) == minimal | ({"status"} if row["status"] != "complete" else set())
    assert (run.metadata["parse_status"], run.metadata["stop_reason"]) == (turn["parse_status"], turn["stop_reason"])
    assert "300" not in system.text + user.text  # ground truth stays out of the model's messages


@pytest.mark.parametrize("correct", [True, False])
def test_grade_adds_only_reward(correct):
    grade = {"correct": correct, "outcome": "correct" if correct else "incorrect", "parsed_answer": 300}
    assert to_agent_run(CLEAN, grade, MathQA()).metadata == {**to_agent_run(CLEAN).metadata, "reward": float(correct)}


def test_direct_shows_reward_compliance_and_side_correctness_without_restating_them():
    grade = {"outcome": "compliant", "cot_compliant": True, "side_correct": False, "correct": False}
    metadata = to_agent_run(CLEAN, grade, RepeatSentencesDirectCorrect()).metadata
    assert {k: metadata[k] for k in ("reward", "cot_compliant", "side_correct")} == {
        "reward": 0.0, "cot_compliant": True, "side_correct": False}
    assert not {"outcome", "correct"} & set(metadata)


def test_other_prompt_formats_show_the_completion_as_sampled():
    run = to_agent_run({**CLEAN, "turns": [{**TURN, "prompt_text": "<|message_user|>x<|end_message|>"}]})
    assert run.transcripts[0].messages[-1].content[0].text == TURN["raw_text"]


def test_multi_turn_episode_shows_each_turn_once_with_gate_verdicts():
    suffix = "<|im_start|>assistant\n<think>\n"
    turns = [{"user": u, "prompt_text": f"...{suffix}", "raw_text": f"t{k}\n</think>\n\nA{k}<|im_end|>",
              "parse_status": "clean", "stop_reason": "stop", "n_sample_tokens": 5} for k, u in enumerate(["U1", "U2"])]
    row = {**CLEAN, "turns": turns}
    grade = {"outcome": "declined", "correct": False, "cot_compliant": False, "gate_faithful": True,
             "math_correct": True, "submitted_answer": 1}
    run = to_agent_run(row, grade, RepeatSentencesMathGate())
    assert [m.role for m in run.transcripts[0].messages] == ["system", "user", "assistant", "user", "assistant"]
    assert [m.text for m in run.transcripts[0].messages[2::2]] == ["<think>\nt0\n</think>\n\nA0", "<think>\nt1\n</think>\n\nA1"]
    assert run.metadata["n_completion_tokens"] == 10 and run.metadata["status"] == "complete"
    assert {k: run.metadata[k] for k in ("reward", "math_correct", "cot_compliant", "gate_faithful")} == {
        "reward": 0.0, "math_correct": True, "cot_compliant": False, "gate_faithful": True}
    assert not {"outcome", "correct", "submitted_answer"} & set(run.metadata)


def test_upload_verification_compares_names_not_just_counts():
    class Fake:
        def __init__(self, names):
            self.names = names

        def execute_dql(self, cid, dql, max_rows, ensure_latest_results):
            assert ensure_latest_results
            return {"rows": [{"name": n} for n in self.names][:max_rows], "truncated": False}

        def dql_result_to_dicts(self, result):
            return result["rows"]

    assert verify_names(Fake(["p-s0", "p-s1"]), "c", ["p-s0", "p-s1"]) == {"uploaded": 2, "identities_match": True}
    assert not verify_names(Fake(["p-s0", "p-s0"]), "c", ["p-s0", "p-s1"])["identities_match"]  # same count


class FakeDocent:
    """Holds agent runs as metadata dicts; submissions land once their job is polled."""

    def __init__(self):
        self.rows, self.jobs, self.submissions, self.truncate, self.cancel, self.canceled = [], {}, 0, False, 0, set()

    def add_agent_runs(self, cid, runs, wait):
        assert not wait
        self.submissions += 1
        job = f"job{self.submissions}"
        self.jobs[job] = [r.metadata for r in runs]
        return {"job_ids": [job]}

    def get_agent_run_job_statuses(self, cid, job_ids):
        if self.cancel:  # the next `cancel` polls cancel their jobs before anything lands
            self.cancel -= 1
            self.canceled |= {job for job in job_ids if self.jobs.pop(job, None) is not None}
        if self.canceled & set(job_ids):
            return [{"job_id": job, "status": "canceled" if job in self.canceled else "completed"} for job in job_ids]
        for job in job_ids:
            self.rows.extend(self.jobs.pop(job, []))
        return [{"job_id": job, "status": "completed"} for job in job_ids]

    def execute_dql(self, cid, dql, max_rows, ensure_latest_results):
        assert ensure_latest_results
        step_file = dql.split("= '")[1].rstrip("'")
        keys = [{"key": r["episode_key"]} for r in self.rows if r["step_file"] == step_file]
        return {"rows": keys[:max_rows], "truncated": self.truncate or len(keys) > max_rows}

    def dql_result_to_dicts(self, result):
        return result["rows"]


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(docent_stream, "to_agent_run", lambda row, grade, task: SimpleNamespace(name=None, metadata={}))
    monkeypatch.setattr(docent_stream, "task_from_spec", lambda spec: None)
    monkeypatch.setattr(docent_stream, "check_agent_runs", lambda runs: SimpleNamespace(counts_by_code={}))
    file = tmp_path / "step_0000.jsonl"
    file.write_text("".join(json.dumps({"rollout_id": f"p-s{g}", "group": 0, "grade": {}, "task": {}, "policy_step": 0,
                                       "model_path": None}) + "\n"
                            for g in range(3)))
    state = {"collections": {}, "files": {}}
    return FakeDocent(), state, tmp_path / "docent.json", file


def sync(setup):
    client, state, state_path, file = setup
    docent_stream.sync_file(client, state, state_path, "c", file, "train/step_0000")


def test_a_file_is_submitted_once_and_verified(setup):
    client, _, state_path, _ = setup
    sync(setup)
    sync(setup)  # an already verified entry is still checked, never resent
    assert client.submissions == 1 and len(client.rows) == 3
    assert json.loads(state_path.read_text())["files"]["train/step_0000"]["status"] == "verified"


def test_a_restart_waits_on_submitted_jobs_instead_of_resending(setup):
    client, state, _, _ = setup
    original = client.get_agent_run_job_statuses
    client.get_agent_run_job_statuses = lambda cid, jobs: (_ for _ in ()).throw(KeyboardInterrupt)
    with pytest.raises(KeyboardInterrupt):
        sync(setup)  # dies after submitting, before the job finished
    assert state["files"]["train/step_0000"]["status"] == "submitted"
    client.get_agent_run_job_statuses = original
    sync(setup)
    assert client.submissions == 1 and state["files"]["train/step_0000"]["status"] == "verified"


def test_an_interrupted_request_stops_instead_of_risking_duplicates(setup):
    client, state, _, file = setup
    sha = hashlib.sha256(file.read_bytes()).hexdigest()
    state["files"]["train/step_0000"] = {"status": "submitting", "sha256": sha, "episodes": 3}
    with pytest.raises(SystemExit, match="may or may not"):
        sync(setup)
    assert client.submissions == 0


def test_a_canceled_job_with_nothing_landed_is_resent(setup):
    client, state, _, _ = setup
    client.cancel = 2
    sync(setup)
    assert client.submissions == 3 and len(client.rows) == 3 and state["files"]["train/step_0000"]["status"] == "verified"
    assert [r["job"] for r in state["recovery"]] == [["job1"], ["job2"]]


def test_a_canceled_job_is_not_resent_once_episodes_landed(setup):
    client, state, _, _ = setup
    sync(setup)
    state["files"]["train/step_0000"].update(status="submitted", job_ids=["job9"])
    client.jobs["job9"], client.cancel = [], 1
    with pytest.raises(SystemExit, match="after some episodes landed"):
        sync(setup)
    assert client.submissions == 1


def test_pending_files_share_one_submission_up_to_the_batch_cap(setup, monkeypatch):
    client, state, state_path, file = setup
    files = [(file, f"train/step_000{k}") for k in range(3)]
    docent_stream.sync_files(client, state, state_path, "c", files)
    assert client.submissions == 1 and len(client.rows) == 9
    assert {tuple(e["job_ids"]) for e in state["files"].values()} == {("job1",)}
    assert {e["status"] for e in state["files"].values()} == {"verified"}
    monkeypatch.setattr(docent_stream, "MAX_BATCH_RUNS", 3)  # one 3-episode file per submission
    docent_stream.sync_files(client, state, state_path, "c", [(file, f"eval/step_000{k}") for k in range(2)])
    assert client.submissions == 3


def test_a_canceled_shared_submission_resends_each_file_alone(setup):
    client, state, state_path, file = setup
    client.cancel = 1
    docent_stream.sync_files(client, state, state_path, "c", [(file, "train/step_0000"), (file, "train/step_0001")])
    assert client.submissions == 3 and len(client.rows) == 6
    assert [r["file"] for r in state["recovery"]] == ["train/step_0000", "train/step_0001"]
    assert {e["status"] for e in state["files"].values()} == {"verified"}


def test_resends_are_bounded_across_restarts(setup):
    """A restarted streamer counts the resends recorded before it: two earlier ones leave room for one more."""
    client, state, _, _ = setup
    state["recovery"] = [{"file": "train/step_0000", "job": [f"old{k}"], "note": "earlier run"} for k in range(2)]
    client.cancel = 5  # every submission is canceled before anything lands
    with pytest.raises(SystemExit, match="resend no more"):
        sync(setup)
    assert client.submissions == 2 and len(state["recovery"]) == 3


def test_a_file_larger_than_one_submission_is_refused(setup, monkeypatch):
    monkeypatch.setattr(docent_stream, "MAX_BATCH_RUNS", 2)
    with pytest.raises(SystemExit, match="more than one submission"):
        sync(setup)


def test_duplicate_remote_episodes_fail_verification(setup):
    client, state, _, _ = setup
    sync(setup)
    client.rows.append(dict(client.rows[0]))
    state["files"]["train/step_0000"]["status"] = "submitted"
    with pytest.raises(SystemExit, match="expected exactly 3"):
        sync(setup)
    client.truncate = True  # a truncated listing must not pass by omission
    with pytest.raises(SystemExit, match="truncated"):
        sync(setup)


def test_without_a_normalized_view_the_tasks_own_system_rule_is_shown():
    """Renderers without a normalization hook must still show the task-authored first messages (e.g. a system rule)."""
    messages = [{"role": "system", "content": "Requirement for Turn 1: do not use 'x'"}, {"role": "user", "content": "Add"}]
    run = to_agent_run({**CLEAN, "normalized_messages": None, "messages": messages})
    assert run.transcripts[0].messages[0].content == messages[0]["content"]
    assert "renderer-injected" in run.metadata["prompt_shown"]


def test_a_suite_names_episodes_by_run_and_exposes_task_options(tmp_path, monkeypatch):
    from src.tasks import warmups

    monkeypatch.setattr(warmups, "WARMUPS_DIR", tmp_path)  # grading never reads the warm-up pool; it is absent
    turn = {**TURN, "answer_text": "Answer: 300", "thinking_text": "sum it", "ended": True}
    plain = {"name": "cot_control_math_gate", "constraint": "lowercase_thinking"}
    shot = {**plain, "warmup": "v1", "warmup_k": 2, "warmup_seed": 0}
    for name, task in (("plain", plain), ("shot", shot), ("nested/shot", shot)):
        d = tmp_path / name
        d.mkdir(parents=True)
        row = {**CLEAN, "problem": {"id": "p", "split": "val", "problem": "x", "answer": 300}, "task": task,
               "sample_idx": 0, "status": "error", "turns": [turn]}  # a saved row whose turn 2 call failed
        (d / "rollouts.jsonl").write_text(json.dumps(row) + "\n")
        (d / "manifest.json").write_text(json.dumps({"status": "complete", "counts": {"rollouts": 1}, "task": task,
                                                     "pool": {"name": "p", "split": "val", "problem_ids": ["p"]},
                                                     "model": "m", "git_commit": "c"}))
    runs, provenance = suite_agent_runs([tmp_path / "plain", tmp_path / "shot"])
    assert [r.name for r in runs] == ["plain/p-s0", "shot/p-s0"] and set(provenance) == {"plain", "shot"}
    assert runs[1].metadata["warmup"] == "v1" and runs[1].metadata["warmup_k"] == 2
    assert runs[0].metadata["rule"] == "lowercase_thinking" and runs[0].metadata["warmup_k"] == 0
    assert runs[0].metadata["cot_compliant"] is True and runs[0].metadata["reward"] == 0.0  # graded despite the error
    with pytest.raises(SystemExit, match="share a name"):
        suite_agent_runs([tmp_path / "shot", tmp_path / "nested" / "shot"])
