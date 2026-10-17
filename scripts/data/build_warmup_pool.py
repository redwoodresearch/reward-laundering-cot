"""Build a warm-up pool, data/warmups/<constraint>.<version>.jsonl, from an `rl rollouts` run of the warm-up message
alone on train questions (the model's own traces): pool gsm8k_cot_warmup with task_options {"constraint": ...} for
lowercase, "OK" at sentence start and meow, or gsm8k_word_suppression_train_warmup for word suppression (whose
questions carry their keywords). data/README.md has the commands.

    uv run python -m scripts.data.build_warmup_pool source=results/rollouts/<run>[,<run>...] constraint=lowercase_thinking version=v1

Frozen procedure (set before any target result): eligible = train split, answer correct, turn ended cleanly, thinking of
min_chars..max_chars characters. In question order (runs in the order given; a later run extends an earlier one's
questions when too few were eligible), each eligible trace is made compliant by the rule's deterministic TRANSFORM and
kept if the ported grader then passes it, until M are kept (failures are recorded and skipped, not retried). The final response is kept as sampled. Word suppression has no mechanical transform: its pool is built from
a teacher's rewrites (scripts/data/rewrite_warmup.py, `rewrites=`)."""

import hashlib
import json
import re
from pathlib import Path

import chz

from src.tasks import Problem, task_from_spec
from src.tasks.math_gates import WORD_SUPPRESSION, rule_and_check
from src.tasks.warmups import WARMUPS_DIR


def _ok_start(text: str) -> str:
    """'OK ' before every sentence and line, split as upstream's grade_ok_start_sentence splits them."""
    parts = re.split(r"((?<=[.!?])\s+|\n+)", text.strip())
    return "".join(p if i % 2 or not p.strip() else "OK " + p for i, p in enumerate(parts))


TRANSFORMS = {
    "lowercase_thinking": str.lower,
    "ok_start_sentence": _ok_start,
    "meow_between_words": lambda t: "\n".join(" meow ".join(line.split()) for line in t.split("\n")),
}


def eligible_rows(source: str, constraint: str, min_chars: int, max_chars: int) -> list[dict]:
    """The frozen eligibility, in question order: train split, correct answer, turn ended cleanly, thinking of
    min_chars..max_chars characters. Each row gains `_run` (its source run) and `_grade`."""
    out = []
    for src in map(Path, source.split(",")):
        manifest = json.loads((src / "manifest.json").read_text())
        assert manifest["task"] == {"name": "cot_control_warmup", "constraint": constraint}, manifest["task"]
        task, order = task_from_spec(manifest["task"]), {pid: i for i, pid in enumerate(manifest["pool"]["problem_ids"])}
        rows = sorted(map(json.loads, (src / "rollouts.jsonl").read_text().splitlines()),
                      key=lambda r: (order[r["problem"]["id"]], r["sample_idx"]))
        for row in rows:
            turn, grade = row["turns"][0], task.grade(Problem.from_row(row["problem"]), row["turns"])
            if (row["problem"]["split"] == "train" and turn["ended"] and grade["correct"]
                    and min_chars <= len(turn["thinking_text"]) <= max_chars):
                out.append({**row, "_run": src.name, "_grade": grade})
    return out


def main(source: str, constraint: str, version: str = "v1", m: int = 16, min_chars: int = 300,
         max_chars: int = 3000, rewrites: str = "", exclude: str = "", exclude_reason: str = "") -> None:
    """`rewrites` (scripts/data/rewrite_warmup.py output) replaces the TRANSFORM: a trace's first passing rewrite.
    `exclude` (comma-separated source rollout ids) drops traces failing a stated, outcome-independent review rule,
    recorded as `exclude_reason` in every exemplar's provenance."""
    excluded = set(filter(None, exclude.split(",")))
    rows = eligible_rows(source, constraint, min_chars, max_chars)
    if rewrites:
        done = {}
        for r in sorted(map(json.loads, Path(rewrites).read_text().splitlines()), key=lambda r: r["attempt"]):
            if r["passes"] and r["rollout_id"] not in done:
                done[r["rollout_id"]] = r["rewrite"]
        transform, how = (lambda row: done.get(row["rollout_id"], "")), f"rewrite:{rewrites}"
    else:
        transform, how = (lambda row: TRANSFORMS[constraint](row["turns"][0]["thinking_text"])), constraint
    out, failed = [], []
    for row in rows:
        if len(out) == m:
            break
        if row["rollout_id"] in excluded:
            continue
        thinking = transform(row)
        rule, compliant = rule_and_check(constraint, Problem.from_row(row["problem"]).inputs)
        if not compliant(thinking):
            failed.append(row["rollout_id"])
            continue
        own_rule = {"rule": rule, "keyword": row["problem"]["keyword"], "forms": row["problem"]["forms"]} \
            if constraint == WORD_SUPPRESSION else {}
        out.append({"id": f"{constraint}-{version}-{row['problem']['id']}", "problem": row["problem"]["problem"], **own_rule,
                    "answer": row["problem"]["answer"], "thinking": thinking,
                    "response": row["turns"][0]["answer_text"].strip(),
                    "source": {"run": row["_run"], "rollout_id": row["rollout_id"], "problem_id": row["problem"]["id"],
                               "transform": how, "originally_compliant": bool(row["_grade"]["cot_compliant"]),
                               "eligibility": {"min_chars": min_chars, "max_chars": max_chars, "m": m},
                               **({"excluded": sorted(excluded), "exclude_reason": exclude_reason} if excluded else {})}})
    path = WARMUPS_DIR / f"{constraint}.{version}.jsonl"
    if path.exists():
        raise SystemExit(f"{path} exists; published pools never change (use a new version)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in out))
    print(f"{path}: {len(out)} exemplars (of {len(rows)} eligible; {len(failed)} skipped as not compliant after the "
          f"transform: {failed}); sha256 {hashlib.sha256(path.read_bytes()).hexdigest()[:16]}")


if __name__ == "__main__":
    chz.entrypoint(main)
