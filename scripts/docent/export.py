"""Batch Docent export: one collection per finished artefact.

    uv run python -m scripts.docent.export rollouts run_dir=results/rollouts/<run> [grader=<grader>]
        [collection_name=<name>]
    uv run python -m scripts.docent.export suite run_dirs=results/rollouts/<a>,<b> name=<collection>
    uv run python -m scripts.docent.export cotcontrol run_dirs=results/cotcontrol/<a>,<b> models=<a>,<b> name=<collection>

Docent is a viewer: transcripts plus a few filterable fields; the local files stay the lossless record, referenced by
path and sha256 from the collection metadata. Every export checks its runs (Docent's transcript sanity check: only
ACCEPTED_WARNINGS pass; `cotcontrol` builds and checks them batch by batch), then creates the collection and records its id in
<dir>/docent.json ("uploading") before adding runs in batches, so a failed upload never leads to a silent duplicate;
finally it lists the collection's run names afresh and compares them with the source ids, multiplicity included
("verified" or "mismatch"). A directory with a docent.json is never uploaded again. Training runs stream to Docent
instead (scripts/docent/stream.py), sharing the transcript projection below."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chz
from docent import Docent
from docent.data_models import AgentRun, Transcript
from docent.data_models.chat import (
    AssistantMessage,
    ContentText,
    check_agent_runs,
    format_check_report,
    parse_chat_message,
)

from src.rollouts.grading import load_grades, run_task
from src.tasks import Problem
from src.tasks.maths import finished
from src.utils.files import read_jsonl

# The generation prompt ends by opening the assistant turn and its think block; the completion continues
# from there, so the opener is shown in the assistant turn although it was never sampled.
ASSISTANT_PREFIX = "<think>\n"
_GENERATION_SUFFIX = "<|im_start|>assistant\n" + ASSISTANT_PREFIX
_END_OF_TURN = "<|im_end|>"
_THINK_END = "</think>"
# Docent flags assistant text containing "reasoning:"/"thinking:"; ours legitimately does, because thinking is shown
# verbatim (as the chat template renders it) and models quote prompts like "...the end of your reasoning: '<...>'".
ACCEPTED_WARNINGS = {"reasoning_embedded_as_text"}
BATCH = 50


# ---- transcript projection (shared with the training-run streamer) ----

def assistant_text(turn: dict[str, Any]) -> str:
    """Qwen3-style prompts: the assistant body as the chat template renders it, i.e. the opener plus the completion,
    minus the `<|im_end|>` turn delimiter. Other formats: the completion exactly as sampled (delimiters included).
    Truncated or unparseable completions are shown as sampled."""
    if not turn["prompt_text"].endswith(_GENERATION_SUFFIX):
        return turn["raw_text"]
    body = turn["raw_text"]
    if finished(turn):
        body = body.removesuffix(_END_OF_TURN)
    return ASSISTANT_PREFIX + body


def grade_metadata(grade: dict, task) -> dict[str, Any]:
    """The verdicts shown in Docent: `reward`, plus the task's other informative grade fields (none that restate it)."""
    return {"reward": task.reward(grade), **{k: grade[k] for k in task.shown_fields}}


def _literal(message: dict) -> dict:
    """A prompt's earlier assistant turn (e.g. a few-shot example) in the sampled turns' view: thinking in literal tags."""
    if message["role"] != "assistant" or isinstance(message["content"], str):
        return message
    text = "".join(f"<think>\n{p['thinking']}\n</think>\n\n" if p["type"] == "thinking" else p["text"]
                   for p in message["content"])
    return {"role": "assistant", "content": text}


def to_agent_run(row: dict[str, Any], grade: dict | None = None, task=None) -> AgentRun:
    turns = row["turns"]
    # Without the renderer's normalized view, the task's own first messages are shown (prompt_text is exact).
    shown = row["normalized_messages"] or row.get("messages") or [{"role": "user", "content": turns[0]["user"]}]
    messages = [parse_chat_message(_literal(m)) for m in shown]
    for k, turn in enumerate(turns):
        if k:
            messages.append(parse_chat_message({"role": "user", "content": turn["user"]}))
        messages.append(AssistantMessage(content=[ContentText(text=assistant_text(turn))], model=row["model"]))
    metadata = {
        "problem_id": row["problem"]["id"],
        "correct_answer": row["problem"]["answer"],
        "parse_status": turns[-1]["parse_status"],
        "stop_reason": turns[-1]["stop_reason"],
        "n_completion_tokens": sum(t["n_sample_tokens"] for t in turns),
    }
    if row["normalized_messages"] is None:
        metadata["prompt_shown"] = ("the task's own messages; renderer-injected text may be omitted "
                                    "(turns[*].prompt_text holds the exact rendered prompt)")
    if len(turns) > 1 or row.get("status", "complete") != "complete":
        metadata["status"] = row["status"]
    if grade is not None:
        metadata.update(grade_metadata(grade, task))
    return AgentRun(name=row["rollout_id"], transcripts=[Transcript(messages=messages)], metadata=metadata)


def verify_names(client, collection_id: str, rollout_ids: list[str]) -> dict:
    """Fresh, untruncated listing of the collection's run names, compared with the source ids (multiplicity
    included)."""
    result = client.execute_dql(collection_id, "SELECT name FROM agent_runs", max_rows=len(rollout_ids) + 1,
                                ensure_latest_results=True)
    if result.get("truncated"):
        raise SystemExit("Docent truncated the verification listing")
    names = Counter(r["name"] for r in client.dql_result_to_dicts(result))
    return {"uploaded": sum(names.values()), "identities_match": names == Counter(rollout_ids)}


# ---- the one upload path ----

def _refuse_if_uploaded(path: Path) -> None:
    if (path / "docent.json").exists():
        raise SystemExit(f"{path / 'docent.json'} exists: already uploaded (or partly; inspect it)")


def upload(receipt_dir: Path, name: str, description: str, metadata: dict, runs: list[AgentRun], **extra) -> dict:
    """Check, create, record, add in batches, verify; returns the receipt written to <receipt_dir>/docent.json."""
    report = check_agent_runs(runs)
    print(format_check_report(report))
    blocking = {code: n for code, n in report.counts_by_code.items() if code not in ACCEPTED_WARNINGS}
    if blocking:
        raise SystemExit(f"sanity warnings {blocking}; fix the mapping first")
    client = Docent()
    collection_id = client.create_collection(name=name, description=description, metadata=metadata)
    accepted = {c: n for c, n in report.counts_by_code.items() if c in ACCEPTED_WARNINGS}
    receipt = {"collection_id": collection_id, "url": f"{client.frontend_url}/dashboard/{collection_id}",
               "status": "uploading", "source_rows": len(runs), "accepted_warnings": accepted, **extra}
    (receipt_dir / "docent.json").write_text(json.dumps(receipt, indent=2))
    for i in range(0, len(runs), BATCH):
        client.add_agent_runs(collection_id, runs[i:i + BATCH])
    receipt.update(verify_names(client, collection_id, [r.name for r in runs]),
                   uploaded_at=datetime.now(UTC).isoformat())
    receipt["status"] = "verified" if receipt["identities_match"] else "mismatch"
    (receipt_dir / "docent.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
    if not receipt["identities_match"]:
        raise SystemExit("the collection does not hold exactly one run per source id; check it")
    return receipt


# ---- rollouts runs (`rl rollouts`) ----

def transcript_note(manifest: dict, first_turn: dict) -> str:
    if first_turn["prompt_text"].endswith(_GENERATION_SUFFIX):
        return (f"Each assistant turn opens with {ASSISTANT_PREFIX!r} from the {manifest['renderer']} prompt template "
                "(not sampled).")
    return (f"Assistant turns are the {manifest['renderer']} completions exactly as sampled (format delimiters "
            "included). The prompt view may omit renderer-injected text (see the run's prompt_shown); each turn's "
            "prompt_text in rollouts.jsonl is the exact rendered prompt.")


def rollouts(run_dir: str, grader: str | None = None, collection_name: str | None = None) -> None:
    """One AgentRun per rollout, named `<problem_id>-s<sample_idx>`; with `grader`, its stored grades as metadata."""
    path = Path(run_dir)
    manifest = json.loads((path / "manifest.json").read_text())
    if not manifest["status"].startswith("complete"):
        raise SystemExit(f"run status is {manifest['status']!r}; export only finished runs")
    _refuse_if_uploaded(path)
    rows, rollouts_sha256 = read_jsonl(path / "rollouts.jsonl")
    ids = {r["rollout_id"] for r in rows}
    if len(rows) != manifest["counts"]["rollouts"] or len(ids) != len(rows):
        raise SystemExit(f"rollouts.jsonl has {len(rows)} rows ({len(ids)} unique); "
                         f"manifest says {manifest['counts']['rollouts']}")
    grades = load_grades(path, rollouts_sha256, grader) if grader else {}
    if grader and set(grades) != ids:
        raise SystemExit(f"{grader} grades do not cover exactly this run's rollout_ids")
    task = run_task(manifest, rows)
    metadata = {
        "model": manifest["model"], "renderer": manifest["renderer"], "pool": manifest["pool"]["name"],
        "split": manifest["pool"]["split"], "pool_sha256": manifest["pool"]["sha256"],
        "pool_sources": manifest["pool"].get("sources"), "task": manifest.get("task"),
        "sampling": {k: manifest["config"][k] for k in ("max_tokens", "temperature", "top_p")},
        "transcript_note": transcript_note(manifest, rows[0]["turns"][0]), "run_dir": str(path.resolve()),
        "rollouts_sha256": rollouts_sha256, "counts": manifest["counts"], "cost": manifest["cost"]["summary"],
        "git_commit": manifest["git_commit"], "versions": manifest["versions"],
        **({"model_path": manifest["model_path"]} if manifest["model_path"] else {}),
        **({"grading": {"grader": grader, **manifest["grading"][grader]}} if grader else {})}
    description = (f"{manifest['model']} on {manifest['pool']['name']}/{manifest['pool']['split']}: {len(rows)} "
                   f"episodes, {f'graded by {grader}' if grader else 'ungraded'}. {task.describe()} Correct answers "
                   "are in run metadata, never in the transcript.")
    upload(path, collection_name or manifest["run_name"], description, metadata,
           [to_agent_run(row, grades.get(row["rollout_id"]), task) for row in rows])


# ---- a suite of rollouts runs (e.g. prompt variants on one pool) ----

def suite_agent_runs(dirs: list[Path]) -> tuple[list[AgentRun], dict]:
    """Every episode of every run, named `<run>/<rollout_id>`, graded (every saved row, as `rl grade` does) by the run's
    recorded task, with `run` and the task's options (e.g. warm-up settings) as filterable metadata; plus per-run
    provenance."""
    if len({d.name for d in dirs}) != len(dirs):  # `run` and provenance are keyed by directory name
        raise SystemExit("two run directories share a name")
    runs, provenance = [], {}
    for d in dirs:
        manifest = json.loads((d / "manifest.json").read_text())
        rows, sha = read_jsonl(d / "rollouts.jsonl")
        if not manifest["status"].startswith("complete") or len(rows) != manifest["counts"]["rollouts"]:
            raise SystemExit(f"{d}: not finished, or {len(rows)} rows where the manifest says {manifest['counts']['rollouts']}")
        provenance[d.name] = {"run_dir": str(d.resolve()), "rollouts_sha256": sha, "task": manifest["task"],
                              "pool": manifest["pool"]["name"], "split": manifest["pool"]["split"],
                              "model": manifest["model"], "git_commit": manifest["git_commit"],
                              "problems": len(manifest["pool"]["problem_ids"]), "episodes": len(rows)}
        task = run_task(manifest, rows)
        receipt = d / "warmup_receipt.json"  # backfilled for runs launched before manifests recorded warm-ups
        warm = manifest.get("warmup") or (json.loads(receipt.read_text()) if receipt.exists() else {})
        for row in rows:
            run = to_agent_run(row, task.grade(Problem.from_row(row["problem"]), row["turns"]), task)
            run.name = f"{d.name}/{row['rollout_id']}"
            options = {k: v for k, v in row["task"].items() if k not in ("name", "constraint")}
            shown = warm.get("assignments", {}).get(row["problem"]["id"])
            run.metadata.update({"run": d.name, "sample_idx": row["sample_idx"], "task_name": row["task"]["name"],
                                 **({"rule": row["task"]["constraint"]} if "constraint" in row["task"] else
                                    {"rule": "word_suppression", "keyword": row["problem"]["keyword"]}
                                    if row["task"]["name"] == "word_suppression_math_gate" else {}),
                                 **({"warmup_k": 0} if row["task"]["name"] in ("cot_control_math_gate",
                                                                                "word_suppression_math_gate") else {}),
                                 **{k: json.dumps(v) if isinstance(v, list) else v for k, v in options.items()},
                                 **({"warmup_ids": json.dumps(shown)} if shown else {})})
            runs.append(run)
    if len({r.name for r in runs}) != len(runs):
        raise SystemExit("two runs share a directory name")
    return runs, provenance


def suite(run_dirs: str, name: str, out: str = "results/docent_suites") -> None:
    """Several finished rollouts runs in one collection, filterable by `run`; the receipt goes to <out>/<name>."""
    dirs = [Path(d) for d in run_dirs.split(",")]
    receipt_dir = Path(out) / name
    _refuse_if_uploaded(receipt_dir)
    runs, provenance = suite_agent_runs(dirs)
    receipt_dir.mkdir(parents=True, exist_ok=True)
    upload(receipt_dir, name, (f"{len(dirs)} rollouts runs in one collection ({len(runs)} episodes); filter by `run` "
                               "(and task options such as warm-up settings). Each episode is graded by its own task; "
                               "correct answers are in metadata, never in the transcript."),
           {"runs": provenance}, runs, run_dirs=[str(d) for d in dirs])


# ---- Chen et al. CoT-Control-QA (scripts/experiments/cotcontrol_qa.py, graded by cotcontrol_grade.py) ----

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_index(samples: Path) -> dict[str, int]:
    """Byte offset of each sample in samples.jsonl, keyed by sha256 of its user prompt (which holds the question and
    the mode's requirement). A prompt sampled twice must have the same completion (the harness samples greedily and
    caches), else the join is ambiguous and refused."""
    index, seen = {}, {}
    with samples.open("rb") as f:
        while line := f.readline():
            offset = f.tell() - len(line)
            row = json.loads(line)
            key = hashlib.sha256(row["messages"][-1]["content"].encode()).hexdigest()
            text = hashlib.sha256(row["raw_text"].encode()).hexdigest()
            if seen.setdefault(key, text) != text:
                raise SystemExit(f"{samples}: one prompt has two different completions; the join is ambiguous")
            index.setdefault(key, offset)
    return index


def cotcontrol_agent_runs(run_dir: Path, model: str, graded: dict[tuple[str, str], Path]):
    """One AgentRun per graded sample, lazily, named `<model>/<dataset>/<mode>/<row>` and joined to samples.jsonl by
    the user prompt; the joined sample's reasoning and response must be the graded row's. Verdicts and answers go in
    metadata only."""
    import pandas as pd

    from src.backends.tinker.clients import get_tokenizer_and_renderer, normalized_messages

    manifest = json.loads((run_dir / "manifest.json").read_text())
    _, renderer = get_tokenizer_and_renderer(manifest["config"]["model"], manifest["renderer"])
    index = _sample_index(run_dir / "samples.jsonl")
    with (run_dir / "samples.jsonl").open("rb") as samples:
        for (dataset, mode), f in sorted(graded.items()):
            df = pd.read_csv(f, keep_default_na=False)
            for i, row in df.iterrows():
                if str(row["response"]).startswith("ERROR:"):
                    raise SystemExit(f"{f.name} row {i}: the sample errored and has no transcript")
                key = hashlib.sha256(row["question"].encode()).hexdigest()
                if key not in index:
                    raise SystemExit(f"{f.name} row {i}: no saved sample for this prompt")
                samples.seek(index[key])
                s = json.loads(samples.readline())
                if (s["reasoning"], s["response"]) != (row["reasoning"], row["response"]):
                    raise SystemExit(f"{f.name} row {i}: the graded reasoning/response is not the saved sample's")
                body = s["raw_text"]
                if s["prompt_text"].endswith(_GENERATION_SUFFIX):
                    body = ASSISTANT_PREFIX + (body.removesuffix(_END_OF_TURN) if s["is_clean"] else body)
                messages = [parse_chat_message(m) for m in normalized_messages(renderer, s["messages"])]
                messages.append(AssistantMessage(content=[ContentText(text=body)], model=model))
                yield AgentRun(name=f"{model}/{dataset}/{mode}/{i:04d}", transcripts=[Transcript(messages=messages)],
                               metadata={"model": model, "dataset": dataset, "mode": mode,
                                         "compliance": int(float(row["compliance"])) if str(row["compliance"]) not in ("", "nan") else None,
                                         "correct": str(row["correctness"]) == "True", "extracted_answer": row["extracted_answer"],
                                         "correct_answer": row["correct_answer"], "control_value": str(row["control_value"])[:300],
                                         "cut_off_at_token_limit": s["stop_reason"] == "length",
                                         "n_completion_tokens": s["n_sample_tokens"]})


def cotcontrol_preflight(dirs: list[Path], labels: list[str]) -> dict[str, dict]:
    """label -> {run_dir, manifest, score, graded files, rows, full_benchmark}, after cotcontrol_checks.validated_run
    (the checks grading uses, a saved sample per graded row, and summary.json's score recomputed from the graded rows);
    nothing is uploaded unless every run passes."""
    from scripts.experiments import cotcontrol_checks as checks

    if not labels or len(set(labels)) != len(labels) or any("/" in label or not label for label in labels):
        raise SystemExit(f"model labels must be distinct, non-empty and free of '/': {labels}")
    runs = {}
    for d, label in zip(dirs, labels, strict=True):
        run = checks.validated_run(d)
        runs[label] = {"run_dir": d, **{k: run[k] for k in ("manifest", "score", "graded", "rows")},
                       "full_benchmark": checks.is_full(run["manifest"])}
    return runs


def _batches(runs, size: int):
    batch = []
    for run in runs:
        batch.append(run)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def verify_cotcontrol(client, collection_id: str, names: list[str]) -> dict:
    """The collection's run names, listed per model/dataset/mode slice (each far below the server's row cap) and
    compared with the uploaded names, multiplicity included; plus the collection's total row count, so nothing outside
    the expected slices goes unnoticed."""
    expected, remote = Counter(names), Counter()
    for model, dataset, mode in sorted({tuple(n.split("/")[:3]) for n in names}):
        query = (f"SELECT name FROM agent_runs WHERE metadata_json->>'model' = '{model}' AND "
                 f"metadata_json->>'dataset' = '{dataset}' AND metadata_json->>'mode' = '{mode}'")
        bound = sum(1 for n in names if n.startswith(f"{model}/{dataset}/{mode}/")) + 1
        result = client.execute_dql(collection_id, query, max_rows=bound, ensure_latest_results=True)
        if result.get("truncated"):
            raise SystemExit(f"Docent truncated the {model}/{dataset}/{mode} listing at {bound} rows")
        remote.update(r["name"] for r in client.dql_result_to_dicts(result))
    total = client.dql_result_to_dicts(client.execute_dql(collection_id, "SELECT COUNT(*) AS n FROM agent_runs",
                                                          ensure_latest_results=True))[0]["n"]
    return {"uploaded": sum(remote.values()), "collection_rows": total,
            "identities_match": bool(names) and remote == expected and total == len(names)}


def cotcontrol(run_dirs: str, models: str, name: str, out: str = "results/cotcontrol") -> None:
    """Several cotcontrol_qa runs (one per model) in one collection, filterable by model, mode, dataset and verdicts.
    Every run is checked first (cotcontrol_preflight); then batches are built, checked and uploaded lazily. The receipt
    goes to <out>/docent.json; a verification mismatch fails the command."""
    dirs, labels = [Path(d) for d in run_dirs.split(",")], models.split(",")
    receipt_dir = Path(out)
    _refuse_if_uploaded(receipt_dir)
    runs = cotcontrol_preflight(dirs, labels)
    scope = ("the full benchmark: GPQA, HLE and MMLU-Pro x nine reasoning-control modes"
             if all(r["full_benchmark"] for r in runs.values()) else
             "a subset of the benchmark (see each run's scope in the collection metadata)")
    provenance = {label: {"run_dir": str(r["run_dir"]), "model_path": r["manifest"]["config"]["model_path"],
                          "cotcontrol_commit": r["manifest"]["cotcontrol_commit"],
                          "scope": {k: r["manifest"]["config"][k] for k in ("datasets", "modes", "n")},
                          "samples_sha256": _sha256_file(r["run_dir"] / "samples.jsonl"),
                          "graded_sha256": {f.name: _sha256_file(f) for f in r["graded"].values()},
                          "controllability_score": r["score"]["controllability_score"]} for label, r in runs.items()}
    receipt_dir.mkdir(parents=True, exist_ok=True)  # before the collection exists, so its receipt can be written
    client = Docent()
    collection_id = client.create_collection(name=name, metadata={"runs": provenance}, description=(
        f"Chen et al., 'Reasoning Models Struggle to Control their Chains of Thought' (arXiv 2603.05706), CoT-Control-QA, "
        f"{scope}; one sample per question, with the repo's prompts and graders (ignore_question judged by GPT-5-mini "
        "via OpenRouter). Temperature 0, 25,000-token limit, Qwen3.8 renderer with its reasoning-effort system sentence. "
        "Filter by model, mode, dataset, compliance, cut_off_at_token_limit. The assistant turn opens with "
        f"{ASSISTANT_PREFIX!r} from the prompt template (not sampled)."))
    receipt = {"collection_id": collection_id, "url": f"{client.frontend_url}/dashboard/{collection_id}",
               "status": "uploading", "run_dirs": [str(d) for d in dirs]}
    (receipt_dir / "docent.json").write_text(json.dumps(receipt, indent=2))
    names, warnings = [], Counter()
    for label, r in runs.items():
        for batch in _batches(cotcontrol_agent_runs(r["run_dir"], label, r["graded"]), BATCH):
            report = check_agent_runs(batch)
            blocking = {c: n for c, n in report.counts_by_code.items() if c not in ACCEPTED_WARNINGS}
            if blocking:
                raise SystemExit(f"sanity warnings {blocking} in {batch[0].name}..; fix the mapping first")
            warnings.update(report.counts_by_code)
            client.add_agent_runs(collection_id, batch)
            names += [run.name for run in batch]
        print(f"{label}: uploaded, {len(names)} runs so far", flush=True)
    receipt.update(verify_cotcontrol(client, collection_id, names), source_rows=len(names),
                   accepted_warnings=dict(warnings), uploaded_at=datetime.now(UTC).isoformat())
    receipt["status"] = "verified" if receipt["identities_match"] else "mismatch"
    (receipt_dir / "docent.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
    if not receipt["identities_match"]:
        raise SystemExit("the collection does not hold exactly one run per graded sample; check it")


COMMANDS = {"rollouts": rollouts, "suite": suite, "cotcontrol": cotcontrol}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        raise SystemExit(f"usage: python -m scripts.docent.export {{{'|'.join(COMMANDS)}}} key=value ...")
    chz.entrypoint(COMMANDS[sys.argv[1]], argv=sys.argv[2:])
