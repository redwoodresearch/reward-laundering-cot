"""Grade a cotcontrol_qa.py run with the CoT-Control repo's own grade_compliance_csv.grade_csv: rule-based compliance
for eight modes, and its GPT-5-mini judge (triple-checked) for ignore_question, reached through OpenRouter
(openai/gpt-5-mini) by swapping the module's OpenAI client; every judge prompt and response is saved to
judge_calls.jsonl. Meta-discussion grading runs only with meta=true. Then the paper's headline controllability score:
compliance averaged over every (dataset, mode) pair, plus per-pair compliance and accuracy.

Before any grading, cotcontrol_checks.py validates the clone, the run's scope (every row's pair and prompt) and every
existing graded file (its rows and receipt); a receipt binds a graded file to its raw file, its own bytes and the
grading options, and is written only once the graded file holds the raw file's rows.
Writes <run>/graded/*.csv (with .receipt.json beside each) and <run>/summary.json.

    python scripts/experiments/cotcontrol_grade.py out_dir=results/cotcontrol/<run>
"""

from __future__ import annotations

import json
import os
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path

import chz
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments import cotcontrol_checks as checks

JUDGE = "openai/gpt-5-mini"


@chz.chz
class Config:
    out_dir: str
    meta: bool = False  # also run the meta-discussion judge on compliant samples
    workers: int = 64
    cost_file: str = str(ROOT / "costs/total_cost.jsonl")


def judge_client(log, lock: threading.Lock, spend: dict):
    """An OpenAI-client stand-in: client.chat.completions.create(model="gpt-5-mini", ...) -> openai/gpt-5-mini on
    OpenRouter, logging every call and its cost."""
    import openai

    class OpenRouterChat:
        def __init__(self):
            self.client = openai.OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1")
            self.chat, self.completions = self, self

        def create(self, model, messages, **kwargs):
            resp = self.client.chat.completions.create(model=f"openai/{model}" if "/" not in model else model,
                                                       messages=messages, extra_body={"usage": {"include": True}}, **kwargs)
            cost = float((resp.usage.model_extra or {}).get("cost") or 0) if resp.usage else 0.0
            with lock:
                spend["usd"] += cost
                spend["calls"] += 1
                log.write(json.dumps({"time": datetime.now(UTC).isoformat(), "model": resp.model, "prompt": messages[0]["content"],
                                      "response": resp.choices[0].message.content, "kwargs": kwargs, "cost": cost}) + "\n")
                log.flush()
            return resp

    return OpenRouterChat()


def main(config: Config) -> None:
    load_dotenv(ROOT / ".env")
    out = Path(config.out_dir).resolve()
    manifest = checks.load_run(out)
    checks.check_clone(manifest["config"]["cotcontrol_dir"])
    run_cceval, grader = checks.upstream(manifest["config"]["cotcontrol_dir"])
    raw = checks.raw_files(out, manifest, run_cceval)  # every pair, row and prompt, before any grading
    options = {"meta": config.meta, "judge": JUDGE, "cotcontrol_commit": checks.PINNED_COMMIT}
    graded = checks.check_graded(raw, options, require=False)  # existing grades must match their receipts

    (out / "graded").mkdir(exist_ok=True)
    lock, spend = threading.Lock(), {"usd": 0.0, "calls": 0}
    log = open(out / "judge_calls.jsonl", "a")  # noqa: SIM115 - closed below
    grader.OpenAI = lambda api_key=None: judge_client(log, lock, spend)
    os.environ.setdefault("OPENAI_API_KEY", "unused: the judge goes through OpenRouter")
    start = datetime.now(UTC)
    try:
        for pair, path in sorted(raw.items()):
            if pair not in graded:
                target, receipt_path = checks.graded_path(path)
                grader.grade_csv(str(path), str(target), num_workers=config.workers, skip_meta_discussion=not config.meta)
                checks.check_rows(path, target)
                receipt_path.write_text(json.dumps(checks.receipt(path, target, options), indent=1))
                graded[pair] = target
    finally:
        log.close()
        if spend["usd"]:
            with open(config.cost_file, "a") as f:
                f.write(json.dumps({"start_timestamp": start.isoformat(), "end_timestamp": datetime.now(UTC).isoformat(),
                                    "full_args": sys.argv, "git_commit": None, "description": f"cotcontrol judge ({out.name})",
                                    "run_cost": spend["usd"], "cost_by_model": {JUDGE: spend["usd"]},
                                    "provisional_models": []}) + "\n")
    score = checks.score(graded)
    summary = {"controllability_score": score["controllability_score"], "pairs": score["pairs"],
               "full_benchmark": checks.is_full(manifest), "options": options,
               "scope": {k: manifest["config"][k] for k in ("datasets", "modes", "n")},
               "by_mode": score["by_mode"], "rows": score["rows"], "judge": spend, "graded_at": datetime.now(UTC).isoformat()}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("controllability_score", "pairs", "full_benchmark", "by_mode", "judge")}, indent=1))


if __name__ == "__main__":
    chz.nested_entrypoint(main)
