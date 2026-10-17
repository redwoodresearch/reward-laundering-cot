"""Rewrite warm-up source traces to follow a rule no deterministic edit can (word suppression: each trace's own
keyword), with a teacher model; every attempt is recorded for build_warmup_pool (`rewrites=`), which takes the first
passing rewrite per trace. The output directory is reserved before any paid call and each attempt is saved as it
arrives, so an interrupted run keeps what it paid for.

    uv run python -m scripts.data.rewrite_warmup source=results/rollouts/<run>[,<run>] constraint=word_suppression \
        teacher=Qwen/Qwen3.8-27B n=10 attempts=1 out=results/warmup_rewrites/<name>

Candidates = the source runs' traces passing build_warmup_pool's eligibility, in question order (first `n`). The
teacher sees only the trace (not the problem's answer) and returns the rewrite as its final response. Recorded per
attempt: source ids, original and rewritten text, grader verdict, whether every integer of the original and the
correct answer still appear. Passing the grader is necessary, not sufficient: inspect the accepted rewrites."""

import asyncio
import json
import re
from pathlib import Path

import chz
from dotenv import load_dotenv

from scripts.data.build_warmup_pool import eligible_rows
from src.backends.tinker.clients import close_service_client, get_tokenizer_and_renderer
from src.backends.tinker.sampling import Sampler
from src.tasks import Problem
from src.tasks.math_gates import rule_and_check
from src.utils.costs import CostTracker

INSTRUCTIONS = {
    "word_suppression": ("Rewrite the reasoning below so that it never uses the word '{keyword}' in any of these forms: "
                         "{forms} (in any letter case). Keep every step, every number and the final answer, and keep it "
                         "about as long as the original: do not summarise or drop any part, change only the wording "
                         "(refer to the same thing in other words). Output only the rewritten reasoning, nothing else."),
}


def numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text))


async def run(rows: list[dict], constraint: str, teacher: str, attempts: int, out: Path, max_tokens: int) -> None:
    load_dotenv()
    tracker = CostTracker(Path("costs/total_cost.jsonl"), run_description=f"rewrite {out.name}")
    sampler = Sampler(model=teacher, tracker=tracker, max_concurrency=64)
    _, renderer = get_tokenizer_and_renderer(sampler.model, sampler.renderer_name)
    async def one(row: dict, attempt: int) -> dict:
        original = row["turns"][0]["thinking_text"]
        inputs = Problem.from_row(row["problem"]).inputs
        compliant = rule_and_check(constraint, inputs)[1]
        instruction = INSTRUCTIONS[constraint].format(keyword=inputs.get("keyword", ""),
                                                      forms=", ".join(f"'{f}'" for f in inputs.get("forms", [])))
        message = f"{instruction}\n\nReasoning:\n<<<\n{original}\n>>>"
        prompt = renderer.build_generation_prompt([{"role": "user", "content": message}]).to_ints()
        result = await sampler.sample_tokens(prompt, sample_idx=attempt, max_tokens=max_tokens)
        rewrite = result.answer_text.strip()
        return {"rollout_id": row["rollout_id"], "run": row["_run"], "problem_id": row["problem"]["id"],
                "attempt": attempt, "teacher": teacher, "max_tokens": max_tokens, "instruction": instruction, "original": original,
                "rewrite": rewrite, "clean": result.is_clean, "passes": result.is_clean and compliant(rewrite),
                "numbers_kept": sorted(numbers(original) - numbers(rewrite)) == [],
                "missing_numbers": sorted(numbers(original) - numbers(rewrite)),
                "answer_kept": str(row["problem"]["answer"]) in numbers(rewrite)}

    counts = {"attempts": 0, "pass": 0, "numbers_kept": 0, "answer_kept": 0}
    with (out / "rewrites.jsonl").open("a") as f:  # each attempt is saved as it arrives
        for done in asyncio.as_completed([one(r, a) for r in rows for a in range(attempts)]):
            result = await done
            f.write(json.dumps(result) + "\n")
            f.flush()
            counts["attempts"] += 1
            if result["passes"]:
                counts["pass"] += 1
                counts["numbers_kept"] += result["numbers_kept"]
                counts["answer_kept"] += result["answer_kept"]
    print(f"{out}: {counts['pass']}/{counts['attempts']} attempts pass the grader; of those, numbers kept "
          f"{counts['numbers_kept']}, answer kept {counts['answer_kept']}")


def main(source: str, constraint: str, out: str, teacher: str = "Qwen/Qwen3.8-27B", n: int = 10, attempts: int = 1,
         min_chars: int = 300, max_chars: int = 3000, max_tokens: int = 32768) -> None:
    if constraint not in INSTRUCTIONS:
        raise SystemExit(f"no rewrite instruction for {constraint!r}")
    rows = eligible_rows(source, constraint, min_chars, max_chars)[:n]
    Path(out).mkdir(parents=True, exist_ok=False)  # reserved before any paid call
    status = "errored"
    try:
        asyncio.run(run(rows, constraint, teacher, attempts, Path(out), max_tokens))
        status = "success"
    finally:
        close_service_client(status)


if __name__ == "__main__":
    chz.entrypoint(main)
