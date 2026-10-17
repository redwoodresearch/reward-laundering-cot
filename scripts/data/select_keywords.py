"""Choose each question's word-suppression keyword from its annotation (scripts/data/annotate_keywords.py) and plain
construction traces (an `rl rollouts` run of math_gsm8k on the same questions), writing data/keywords/<out>.

    uv run python -m scripts.data.select_keywords annotations=<file> usage_run=results/rollouts/<run> out=<name>.jsonl

Rule (frozen before the traces were sampled): the highest-ranked candidate whose forms occur (whole word, any case,
as graded) in at least one construction trace's thinking; if none occurs, the top candidate, flagged
`zero_observed_use`. Questions whose annotation failed its checks are skipped and reported. The traces are
construction data only, never scored."""

from __future__ import annotations

import json
from pathlib import Path

import chz

from src.tasks.constraints import suppresses

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "keywords"
K = 2  # construction traces per question (frozen)


def main(annotations: str, usage_run: str, out: str, corrections: str = "") -> None:
    """`corrections`: a JSON file {"reason": ..., "add_forms": {question id: [forms]}} of adjudicated form additions,
    applied after selection and recorded per row (the raw annotations are never edited)."""
    out_path = DATA_DIR / out
    if out_path.exists():
        raise SystemExit(f"{out_path} exists; committed data files are never rewritten")
    manifest = json.loads((Path(usage_run) / "manifest.json").read_text())
    if manifest["task"] != {"name": "math_qa"} or manifest["config"]["samples_per_problem"] != K:
        raise SystemExit(f"{usage_run} is not a plain math_qa run with {K} samples per question")
    traces: dict[str, dict[int, str]] = {}
    questions: dict[str, str] = {}
    with (Path(usage_run) / "rollouts.jsonl").open() as f:  # streamed: only the thinking is kept
        for line in f:
            row = json.loads(line)
            pid = row["problem"]["id"]
            if row["task"] != {"name": "math_qa"} or row["status"] != "complete" or len(row["turns"]) != 1:
                raise SystemExit(f"{pid} sample {row['sample_idx']}: not a complete single-turn math_qa episode")
            if row["sample_idx"] in traces.setdefault(pid, {}):
                raise SystemExit(f"{pid}: duplicate sample {row['sample_idx']}")
            if questions.setdefault(pid, row["problem"]["problem"]) != row["problem"]["problem"]:
                raise SystemExit(f"{pid}: samples disagree on the question text")
            traces[pid][row["sample_idx"]] = row["turns"][0]["thinking_text"]
    rows, skipped = [], []
    for line in Path(annotations).read_text().splitlines():
        a = json.loads(line)
        if a["errors"]:
            skipped.append(a["id"])
            continue
        if sorted(traces.get(a["id"], {})) != list(range(K)) or questions[a["id"]] != a["problem"]:
            raise SystemExit(f"{a['id']}: needs exactly samples 0..{K - 1} of the same question in {usage_run}")
        cands = a["parsed"]["candidates"]
        used = [sum(not suppresses(t, c["forms"]) for t in traces[a["id"]].values() if t) for c in cands]
        rank = next((i for i, u in enumerate(used) if u), None)
        c = cands[0 if rank is None else rank]
        rows.append({"id": a["id"], "keyword": c["word"], "forms": c["forms"], "family": c["family"], "role": c["role"],
                     "rank": 0 if rank is None else rank, "selection": "zero_observed_use" if rank is None else "used",
                     "traces_using_each_candidate": used, "n_traces": len(traces[a["id"]]),
                     "annotation_file": str(Path(annotations).resolve()), "prompt_sha256": a["prompt_sha256"],
                     "annotator": a["returned_model"], "usage_run": str(Path(usage_run).resolve())})
    if corrections:
        fix = json.loads(Path(corrections).read_text())
        by_id = {r["id"]: r for r in rows}
        if missing := set(fix["add_forms"]) - set(by_id):
            raise SystemExit(f"corrections name unselected questions {sorted(missing)}")
        for pid, forms in fix["add_forms"].items():
            r = by_id[pid]
            r["forms"] = r["forms"] + [f for f in forms if f not in r["forms"]]
            r["form_corrections"] = {"added": forms, "reason": fix["reason"], "file": str(Path(corrections).resolve())}
    out_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{out_path}: {len(rows)} keywords; skipped {skipped}; top candidate kept for "
          f"{sum(r['rank'] == 0 for r in rows)}; zero observed use {sum(r['selection'] != 'used' for r in rows)}")


if __name__ == "__main__":
    chz.entrypoint(main)
