"""Follow an `rl train` run and upload each sealed step file to its `<run>/train` or `<run>/eval` Docent collection.

    uv run python -m scripts.docent.stream run_dir=results/train/<run>

`rl train docent=true` starts this itself; rerun it by hand to catch up after a crash. Pending files go up together, up
to MAX_BATCH_RUNS episodes per Docent job (a busy queue spends ~10 minutes per job, whatever its size). Per file,
docent.json records "submitting" (before the request) -> "submitted" (Docent's job ids) -> "verified" (every
episode_key present exactly once). A restart waits on submitted jobs instead of resending them. Only if Docent cancels
a file's jobs before any of its episodes landed is the file resent, at most MAX_RESENDS times (noted under `recovery`).

Recovery, when the streamer stops on a file:
- left at "submitting" (it stopped mid-request, so the upload may or may not have reached Docent): wait until no
  ingest job of the collection is pending or running. Then either delete every episode with that `step_file` from
  the collection and remove the file's entry from docent.json, or, only if a fresh query shows each expected
  episode_key exactly once with content matching the local file, set its status to "verified". Rerun.
- canceled after some of its episodes landed: delete that file's episodes from the collection, remove its entry from
  docent.json, and rerun. Never resend a file without clearing it first: Docent would hold duplicates."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

import chz
from docent import Docent
from docent.data_models.chat import check_agent_runs

from scripts.docent.export import ACCEPTED_WARNINGS, ASSISTANT_PREFIX, to_agent_run
from src.tasks import task_from_spec
from src.utils.files import write_json

SPLITS = ("train", "eval")
RL_FIELDS = ("advantage", "credited_reward", "group", "group_status")  # training episodes only
MAX_RESENDS = 3  # Docent intermittently cancels large ingest jobs (2,500-episode eval files) before any lands
MAX_BATCH_RUNS = 5000  # episodes per bundled submission (also bounds the streamer's memory)


def agent_run(row: dict, step_file: str):
    """`reward` and the task's verdicts, the policy that sampled it (`base` = the untrained model), and for training
    episodes the GRPO group and advantage. The step file's sha256 is in docent.json."""
    run = to_agent_run(row, row["grade"], task_from_spec(row["task"]))
    key = f"{step_file}/{row['rollout_id']}" + (f"/g{row['group']}" if "group" in row else "")
    run.name = key
    run.metadata.update({"policy_step": row["policy_step"], "model_path": row["model_path"] or "base",
                         **{k: row[k] for k in RL_FIELDS if k in row}, "step_file": step_file, "episode_key": key})
    return run


def collection_metadata(run_dir: Path, manifest: dict, split: str) -> dict:
    config = manifest["config"]
    return {
        "run_dir": str(run_dir.resolve()), "split": split, "model": manifest["model"], "renderer": manifest["renderer"],
        "pool": manifest["pool"], "task": manifest["task"], "grader": manifest["grader"],
        "config": config, "eval_samples_per_problem": manifest["eval_samples_per_problem"],
        "git_commit": manifest["git_commit"],
        "transcript_note": (f"The system message is injected by the {manifest['renderer']} renderer; the assistant turn "
                            f"opens with {ASSISTANT_PREFIX!r} from the prompt template (not sampled)."),
    }


def description(manifest: dict, split: str) -> str:
    objective = manifest["config"].get("objective", "grpo")  # runs before the option existed were GRPO
    if split == "eval":
        what = "held-out episodes sampled from the learner before each eval step's update"
    elif objective == "grpo":
        what = ("stored training episodes (including partial ones), with reward and GRPO advantage; `group` is a "
                "question slot within a `policy_step`, and group_status=used means selected for an update")
    elif objective == "grpo_shuffled":
        what = ("stored training episodes of the shuffled-reward control (including partial ones): `reward` is the "
                "episode's own, but its GRPO advantage comes from `credited_reward`, the group's rewards permuted "
                "among its episodes; `group` is a question slot within a `policy_step`, and group_status=used means "
                "selected for an update")
    else:
        reduction = manifest["config"].get("sft_reduction", "mean")  # runs before the option existed used mean
        what = (f"stored training episodes of the no-reward {objective} ablation ({reduction} token weights; "
                "including partial ones), every episode of a group without API errors imitated (group_status=used, "
                "advantage null)"
                + ("; they are always sampled from the base model (model_path \"base\"), so `policy_step` is only "
                   "the update they fed" if objective == "sft_base_rollouts" else ""))
    learner_rows = split == "eval" or objective != "sft_base_rollouts"
    compare = ("Compare learner checkpoints by filtering on `policy_step`" if learner_rows
               else "Training rows here all come from the base model")
    return (f"{objective} run {manifest['run_name']} ({manifest['model']} on {manifest['pool']['name']}): {what}. "
            f"{compare}; a whole-collection average mixes steps, and a step still uploading may be partial. "
            f"{task_from_spec(manifest['task']).describe()}")


def remote_keys(client: Docent, collection_id: str, step_file: str, bound: int) -> Counter:
    """Every episode_key Docent holds for this file, with multiplicity (fresh results, bounded)."""
    result = client.execute_dql(collection_id, "SELECT metadata_json->>'episode_key' AS key FROM agent_runs "
                                               f"WHERE metadata_json->>'step_file' = '{step_file}'",
                                max_rows=bound, ensure_latest_results=True)
    if result.get("truncated"):  # a partial list could pass verification by omission (server cap: 10,000 rows)
        raise SystemExit(f"{step_file}: Docent truncated the verification query at {bound} rows")
    return Counter(r["key"] for r in client.dql_result_to_dicts(result))


def wait_for_jobs(client: Docent, collection_id: str, job_ids: list[str], poll_seconds: float = 5.0) -> bool:
    """Wait until every job has completed or been canceled; True when none was canceled."""
    pending, ok = set(job_ids), True
    while pending:
        for status in client.get_agent_run_job_statuses(collection_id, list(pending)):
            if status["status"] in ("completed", "canceled"):
                pending.discard(status["job_id"])
            if status["status"] == "canceled":
                ok = False
                print(f"Docent job {status['job_id']} was canceled: {status.get('error_message')}", flush=True)
        if pending:
            time.sleep(poll_seconds)
    return ok


def submit(client: Docent, state: dict, state_path: Path, entry: dict, cid: str, runs: list) -> None:
    entry["status"] = "submitting"
    write_json(state_path, state)
    entry.update(status="submitted", job_ids=client.add_agent_runs(cid, runs, wait=False)["job_ids"])
    write_json(state_path, state)


def load(file: Path, step_file: str) -> tuple[str, list, str]:
    digest, runs = hashlib.sha256(), []
    with file.open("rb") as f:  # line by line: eval files can be large
        for line in f:
            digest.update(line)
            runs.append(agent_run(json.loads(line), step_file))
    if max(Counter(r.metadata["episode_key"] for r in runs).values(), default=1) > 1:
        raise SystemExit(f"{step_file}: episode keys are not unique")
    return step_file, runs, digest.hexdigest()


def sync_files(client: Docent, state: dict, state_path: Path, cid: str, files: list[tuple[Path, str]]) -> None:
    """Upload, wait for and verify (path, step_file) pairs in chunks of up to MAX_BATCH_RUNS episodes; the chunk's
    files never submitted before go up as one Docent submission."""
    chunk, size = [], 0
    for file, step_file in files:
        with file.open("rb") as f:  # count before projecting, so at most MAX_BATCH_RUNS episodes are held
            n = sum(1 for _ in f)
        if n > MAX_BATCH_RUNS:
            raise SystemExit(f"{step_file} has {n} episodes, more than one submission's {MAX_BATCH_RUNS}")
        if chunk and size + n > MAX_BATCH_RUNS:
            sync_chunk(client, state, state_path, cid, chunk)
            chunk, size = [], 0
        chunk.append(load(file, step_file))
        size += n
    if chunk:
        sync_chunk(client, state, state_path, cid, chunk)


def sync_file(client: Docent, state: dict, state_path: Path, cid: str, file: Path, step_file: str) -> None:
    sync_files(client, state, state_path, cid, [(file, step_file)])


def sync_chunk(client: Docent, state: dict, state_path: Path, cid: str, chunk: list[tuple[str, list, str]]) -> None:
    new = [item for item in chunk if item[0] not in state["files"]]
    for step_file, runs, _ in new:
        report = check_agent_runs(runs)
        blocking = {c: n for c, n in report.counts_by_code.items() if c not in ACCEPTED_WARNINGS}
        if blocking:
            raise SystemExit(f"{step_file}: sanity warnings {blocking}; fix the mapping first")
        if remote_keys(client, cid, step_file, bound=len(runs) + 1):
            raise SystemExit(f"{step_file}: Docent already holds episodes for a file never submitted from here")
    if new:
        entries = [state["files"].setdefault(f, {"status": "submitting", "sha256": sha, "episodes": len(runs)})
                   for f, runs, sha in new]
        write_json(state_path, state)
        job_ids = client.add_agent_runs(cid, [r for _, runs, _ in new for r in runs], wait=False)["job_ids"]
        for entry in entries:
            entry.update(status="submitted", job_ids=job_ids)
        write_json(state_path, state)
    for step_file, runs, sha in chunk:
        finish(client, state, state_path, cid, step_file, runs, sha)


def finish(client: Docent, state: dict, state_path: Path, cid: str, step_file: str, runs: list, sha: str) -> None:
    """Wait for a submitted file's job(s), resend it alone if Docent canceled them before any of it landed, and verify
    that Docent holds each of its episode keys exactly once."""
    expected = Counter(r.metadata["episode_key"] for r in runs)
    entry = state["files"][step_file]
    if entry["sha256"] != sha:
        raise SystemExit(f"{step_file} changed after it was submitted")
    if entry["status"] == "submitting":
        raise SystemExit(f"{step_file} (sha256 {sha}) may or may not have reached collection {cid}: the streamer "
                         "stopped mid-request. See 'Recovery' in scripts/docent/stream.py's docstring.")
    while not wait_for_jobs(client, cid, entry["job_ids"]):
        if remote_keys(client, cid, step_file, bound=2 * len(runs) + 1):
            raise SystemExit(f"{step_file}: Docent canceled job(s) {entry['job_ids']} after some episodes landed. "
                             "See 'Recovery' in scripts/docent/stream.py's docstring.")
        if sum(r["file"] == step_file for r in state.get("recovery", [])) >= MAX_RESENDS:  # across restarts too
            raise SystemExit(f"{step_file}: Docent canceled it {MAX_RESENDS + 1} times; resend no more")
        state.setdefault("recovery", []).append({"file": step_file, "job": entry["job_ids"], "note": (
            "Docent canceled the ingest job with 0 episodes of this file on Docent; resent automatically")})
        submit(client, state, state_path, entry, cid, runs)
    found = remote_keys(client, cid, step_file, bound=2 * len(runs) + 1)
    if found != expected:
        raise SystemExit(f"{step_file}: Docent holds {sum(found.values())} episodes ({len(found)} distinct keys), "
                         f"expected exactly {len(runs)} distinct keys once each")
    entry["status"] = "verified"
    write_json(state_path, state)
    print(f"docent: {step_file} ({len(runs)} episodes)", flush=True)


def main(run_dir: str, poll_seconds: float = 30.0, wait_for_start_seconds: float = 900.0) -> None:
    path = Path(run_dir)
    parent = os.getppid()
    deadline = time.time() + wait_for_start_seconds
    while not (path / "manifest.json").exists():
        if time.time() > deadline or os.getppid() != parent:
            raise SystemExit(f"{path} never started; nothing to stream")
        time.sleep(5)

    client = Docent()
    state_path = path / "docent.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"collections": {}, "files": {}}
    manifest = json.loads((path / "manifest.json").read_text())
    for split in SPLITS:
        if split not in state["collections"]:
            cid = client.create_collection(name=f"{manifest['run_name']}/{split}", description=description(manifest, split),
                                           metadata=collection_metadata(path, manifest, split))
            state["collections"][split] = {"id": cid, "url": f"{client.frontend_url}/dashboard/{cid}"}
            write_json(state_path, state)
            print(f"docent {split}: {state['collections'][split]['url']}", flush=True)

    while True:
        finished = json.loads((path / "manifest.json").read_text())["status"] != "running"  # read before listing files
        for split in SPLITS:
            pending = [(file, f"{split}/{file.stem}") for file in sorted((path / split).glob("step_*.jsonl"))]
            pending = [(file, f) for file, f in pending if state["files"].get(f, {}).get("status") != "verified"]
            sync_files(client, state, state_path, state["collections"][split]["id"], pending)
        if finished:
            return
        if os.getppid() != parent:
            raise SystemExit("training process exited without finishing the run; rerun this script to catch up")
        time.sleep(poll_seconds)


if __name__ == "__main__":
    chz.entrypoint(main)
