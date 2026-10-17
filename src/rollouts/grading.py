"""`grade` command: grade a finished run into `<run_dir>/grades_<grader>.jsonl` without touching its rollouts.
The task is taken from the run's own records, never from today's pool definitions. The summary goes under
`manifest["grading"][<grader>]`, tied to the sha256 of the rollouts graded. Accuracy is over every stored row,
including incomplete episodes and those cut short by an error after at least one sampled turn; an episode whose
first call failed has no row and appears only in the manifest's errors."""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from src.tasks import Problem, task_from_spec
from src.utils.files import read_jsonl, write_json, write_jsonl


def grades_path(run_dir: Path, grader: str) -> Path:
    return run_dir / f"grades_{grader}.jsonl"


def run_task(manifest: dict, rows: list[dict]):
    """The single task all rows were sampled under, as recorded in the run. (Older row formats: regrade with the run's
    recorded source commit and grader version.)"""
    if not manifest.get("task") or any(r.get("task") != manifest["task"] for r in rows):
        raise ValueError("run records must agree on an explicit task")
    return task_from_spec(manifest["task"])


def grade_run(run_dir: Path) -> dict:
    rollouts = run_dir / "rollouts.jsonl"
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("status", "").startswith("complete"):
        raise ValueError(f"run status is {manifest.get('status')!r}; grade only finished runs")
    rows, rollouts_sha256 = read_jsonl(rollouts)
    if not rows or len(rows) != manifest["counts"]["rollouts"]:
        raise ValueError(f"{rollouts} has {len(rows)} rows; manifest says {manifest['counts']['rollouts']}")
    ids = [r["rollout_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{rollouts} has duplicate rollout_ids")
    task = run_task(manifest, rows)
    out = grades_path(run_dir, task.grader)
    if out.exists():
        raise FileExistsError(f"{out} exists; grades for a grader version are written once")

    grades = []
    for row in rows:
        problem = Problem.from_row(row["problem"])
        if task.solve(problem.inputs) != problem.answer:
            raise ValueError(f"{row['rollout_id']}: stored answer {problem.answer} != {task.name} of {problem.inputs}")
        grade = task.grade(problem, row["turns"])
        grades.append({"rollout_id": row["rollout_id"], "grader": task.grader, **grade, "reward": task.reward(grade)})

    counts = Counter(g["outcome"] for g in grades)
    summary = {
        "grader": task.grader,
        "rollouts_sha256": rollouts_sha256,
        "graded": len(grades),
        "counts": {o: counts[o] for o in task.outcomes},
        "accuracy": sum(g["correct"] for g in grades) / len(grades),
        "mean_reward": sum(g["reward"] for g in grades) / len(grades),
        "graded_at": datetime.now(UTC).isoformat(),
    }
    if hasattr(task, "metrics"):
        summary["metrics"] = task.metrics(grades, rows)

    write_jsonl(out, grades)  # published only after everything above succeeded
    manifest.setdefault("grading", {})[task.grader] = summary
    write_json(manifest_path, manifest)
    return summary


def load_grades(run_dir: Path, rollouts_sha256: str, grader: str) -> dict[str, dict]:
    """Grades by rollout_id, refusing grades made from different rollouts or with duplicate ids."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    recorded = manifest.get("grading", {}).get(grader, {}).get("rollouts_sha256")
    if recorded != rollouts_sha256:
        raise ValueError(f"{grader} grades were made from rollouts {recorded}, not {rollouts_sha256}")
    grades: dict[str, dict] = {}
    for line in grades_path(run_dir, grader).read_text().splitlines():
        g = json.loads(line)
        if g["rollout_id"] in grades:
            raise ValueError(f"duplicate grade for {g['rollout_id']}")
        grades[g["rollout_id"]] = g
    return grades
