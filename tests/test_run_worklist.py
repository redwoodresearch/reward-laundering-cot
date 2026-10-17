"""scripts/ops/run_worklist.py. Tests that start real systemd scopes in research.slice run only with
RUN_SYSTEMD_TESTS=1 (they need research.slice, set up as run_worklist.py's docstring says); each uses its own lock, a unique unit prefix and a
cleanup that stops any scope it left, so it never blocks, sees or survives a real batch."""

import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

RUNNER = Path(__file__).resolve().parents[1] / "scripts/ops/run_worklist.py"
SLICE_DIR = Path(f"/sys/fs/cgroup/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/research.slice")
systemd = pytest.mark.skipif(os.environ.get("RUN_SYSTEMD_TESTS") != "1" or not SLICE_DIR.exists(),
                             reason="starts real systemd scopes: set RUN_SYSTEMD_TESTS=1 (needs research.slice)")


@pytest.fixture
def prefix():
    """A unit prefix no other test or suite uses; any scope left with it is stopped afterwards."""
    p = f"rltest-{uuid.uuid4().hex[:8]}-"
    yield p
    for scope in SLICE_DIR.glob(f"{p}*.scope") if SLICE_DIR.exists() else []:
        subprocess.run(["systemctl", "--user", "stop", scope.name], check=False)


def start_runner(tmp_path, jobs, prefix, *extra, log_arg=None):
    worklist, log = tmp_path / "w.jsonl", tmp_path / "log.jsonl"
    worklist.write_text("".join(json.dumps(j) + "\n" for j in jobs))
    code = (f"import runpy, sys; g = runpy.run_path({str(RUNNER)!r}, run_name='rw'); m = g['main'].__globals__; "
            f"m['LOCK'] = __import__('pathlib').Path({str(tmp_path / 'lock')!r}); m['PREFIX'] = {prefix!r}; "
            f"sys.argv = ['rw', {str(worklist)!r}, '--log', {log_arg or str(log)!r}, '--sample-seconds', '1', *{list(extra)!r}]; sys.exit(g['main']())")
    return subprocess.Popen([sys.executable, "-c", code], stderr=subprocess.PIPE, text=True, cwd=tmp_path), log


def events(log):
    return [json.loads(line) for line in log.read_text().splitlines()]


@systemd
def test_peak_of_a_job_that_exits_between_samples_is_still_recorded(tmp_path, prefix):
    # the job runs in another directory and the log path is relative: the final measurement must still be found
    job = {"name": "alloc", "cmd": [sys.executable, "-c", "x = bytearray(96 * 2**20)"], "cwd": "/"}
    proc, log = start_runner(tmp_path, [job], prefix, log_arg="log.jsonl")
    assert proc.wait(60) == 0, proc.stderr.read()
    end = next(e for e in events(log) if e["event"] == "job_end")
    assert end["peak_mib"] >= 96


@systemd
def test_sigterm_stops_the_running_job_before_the_lock_is_released(tmp_path, prefix):
    job = {"name": "sleeper", "cmd": ["sleep", "120"]}
    proc, log = start_runner(tmp_path, [job], prefix)
    scope = SLICE_DIR / f"{prefix}sleeper.scope"
    for _ in range(50):
        if scope.exists():
            break
        time.sleep(0.2)
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(30) == 130
    assert not scope.exists()
    assert events(log)[-1]["event"] == "batch_stop"


@systemd
def test_refuses_to_start_while_a_job_from_a_dead_batch_still_runs(tmp_path, prefix):
    orphan = subprocess.Popen(["systemd-run", "--user", "--scope", "--collect", "--slice=research.slice",
                               f"--unit={prefix}orphan", "sleep", "60"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            if (SLICE_DIR / f"{prefix}orphan.scope").exists():
                break
            time.sleep(0.2)
        proc, log = start_runner(tmp_path, [{"name": "never", "cmd": ["true"]}], prefix)
        assert proc.wait(30) != 0
        assert "still running" in proc.stderr.read()
        assert not log.exists()
    finally:
        subprocess.run(["systemctl", "--user", "stop", f"{prefix}orphan.scope"], check=False)
        orphan.wait(30)


@systemd
def test_two_jobs_run_at_once_and_a_failure_starts_nothing_new_but_lets_the_other_finish(tmp_path, prefix):
    jobs = [{"name": "slow", "cmd": ["sleep", "4"]}, {"name": "fails", "cmd": ["sh", "-c", "sleep 1; exit 3"]},
            {"name": "never", "cmd": ["true"]}]
    proc, log = start_runner(tmp_path, jobs, prefix, "--parallel", "2")
    assert proc.wait(60) == 1, proc.stderr.read()
    ev = events(log)
    assert [e["name"] for e in ev if e["event"] == "job_start"] == ["slow", "fails"]
    ends = {e["name"]: e["returncode"] for e in ev if e["event"] == "job_end"}
    assert ends == {"slow": 0, "fails": 3}
    assert ev[-1] == {**ev[-1], "event": "batch_stop", "reason": "job_failed"}


@systemd
def test_jobs_appended_while_running_are_picked_up_highest_priority_first(tmp_path, prefix):
    proc, log = start_runner(tmp_path, [{"name": "first", "cmd": ["sleep", "3"]}], prefix)
    for _ in range(50):
        if log.exists() and any(e["event"] == "job_start" for e in events(log)):
            break
        time.sleep(0.2)
    with open(tmp_path / "w.jsonl", "a") as f:
        f.write(json.dumps({"name": "low", "cmd": ["true"]}) + "\n")
        f.write(json.dumps({"name": "high", "cmd": ["true"], "priority": 1}) + "\n")
    assert proc.wait(60) == 0, proc.stderr.read()
    assert [e["name"] for e in events(log) if e["event"] == "job_start"] == ["first", "high", "low"]


@systemd
def test_a_line_still_being_appended_is_not_read_until_it_is_complete(tmp_path, prefix):
    proc, log = start_runner(tmp_path, [{"name": "first", "cmd": ["sleep", "3"]}], prefix)
    for _ in range(50):
        if log.exists() and any(e["event"] == "job_start" for e in events(log)):
            break
        time.sleep(0.2)
    with open(tmp_path / "w.jsonl", "a") as f:
        f.write('{"name": "late", "cmd": ["tr')  # half-written: no closing text, no newline
        f.flush()
        time.sleep(2)
        f.write('ue"]}\n')
    assert proc.wait(60) == 0, proc.stderr.read()
    assert [e["name"] for e in events(log) if e["event"] == "job_start"] == ["first", "late"]


@systemd
def test_a_bad_worklist_line_mid_batch_stops_the_running_jobs(tmp_path, prefix):
    proc, log = start_runner(tmp_path, [{"name": "sleeper", "cmd": ["sleep", "120"]}], prefix)
    for _ in range(50):
        if (SLICE_DIR / f"{prefix}sleeper.scope").exists():
            break
        time.sleep(0.2)
    with open(tmp_path / "w.jsonl", "a") as f:
        f.write(json.dumps({"name": "sleeper", "cmd": ["true"]}) + "\n")  # a duplicate name, read on the next poll
    assert proc.wait(30) != 0
    assert not (SLICE_DIR / f"{prefix}sleeper.scope").exists()
    assert "runner error" in events(log)[-1]["reason"]


@pytest.mark.parametrize("extra", [("--parallel", "0"), ("--sample-seconds", "0"), ("--min-available-gib", "-1")])
def test_settings_that_could_never_make_progress_are_refused(tmp_path, extra):
    proc, log = start_runner(tmp_path, [{"name": "x", "cmd": ["true"]}], "unused-", *extra)
    assert proc.wait(30) == 2 and not log.exists()  # argparse error, before any lock, scope or log
