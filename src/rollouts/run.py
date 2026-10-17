"""`rollouts` command: sample a slice of a problem pool into `<run_dir>/rollouts.jsonl` (one lossless row per
episode, flushed as each finishes; see episodes.py) and `<run_dir>/manifest.json` (config, pool hash, counts, cost,
errors). Ground truth is stored as data only; it never enters the model's messages."""

from __future__ import annotations

import asyncio
import hashlib
import json
import traceback
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import chz

from src.backends.tinker.clients import get_tokenizer_and_renderer, renderer_settings
from src.backends.tinker.sampling import Sampler
from src.dataset.pools import load_pool, pool_path, pool_sources, pool_task
from src.rollouts.episodes import plan_episode, sample_episode
from src.tasks import Problem
from src.utils.cache import FileCache
from src.utils.costs import PUBLISHED_FILE, CostTracker, context_window
from src.utils.files import write_json
from src.utils.provenance import git_commit, git_dirty


async def run(config) -> None:
    # Validate the pool, task and renderer before creating the run directory.
    run_name = config.run_name or f"{config.pool}_{config.split}_{datetime.now(UTC):%Y%m%d_%H%M%S}"
    run_dir = Path(config.output_dir) / run_name
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} exists; runs are never overwritten")
    task = pool_task(config.pool, config.task_options)
    pool_file = pool_path(config.pool, config.split)
    problems = load_pool(config.pool, config.split)[config.problem_offset : config.problem_offset + config.n_problems]
    if len(problems) < config.n_problems:
        raise ValueError(f"{pool_file} has only {len(problems)} problems from offset {config.problem_offset}")

    tracker = CostTracker(Path(config.cost_file), run_description=f"rollouts {run_name}")
    sampler = Sampler(
        model=config.model_name,
        model_path=config.model_path,
        renderer_name=config.renderer_name,
        cache=FileCache(config.cache_dir) if config.use_cache else None,
        tracker=tracker,
        max_concurrency=config.max_concurrency,
    )
    tokenizer, renderer = get_tokenizer_and_renderer(sampler.model, sampler.renderer_name)
    plans = {problem.id: plan_episode(task, problem, tokenizer, renderer) for problem in problems}

    run_dir.mkdir(parents=True)
    manifest_path = run_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "run_name": run_name,
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "config": chz.asdict(config),
        "task": task.spec(),
        **({"warmup": w} if (w := getattr(task, "warmup_receipt", lambda ps: None)(problems)) else {}),
        "pool": {
            "name": config.pool,
            "split": config.split,
            "file": str(pool_file),
            "sha256": hashlib.sha256(pool_file.read_bytes()).hexdigest(),
            "problem_ids": [p.id for p in problems],
            "sources": pool_sources(config.pool),
        },
        "model": sampler.model,
        "model_path": sampler.model_path,
        "renderer": sampler.renderer_name,
        "renderer_settings": renderer_settings(tokenizer, renderer, plans[problems[0].id].prompt),
        "pricing_snapshot": PUBLISHED_FILE.name,
        "context_window": context_window(sampler.model),
        "tokenizer": getattr(tokenizer, "name_or_path", None),
        "versions": {pkg: version(pkg) for pkg in ("tinker", "tinker-cookbook", "transformers")},
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
    }
    write_json(manifest_path, manifest)

    async def one(problem: Problem, sample_idx: int) -> dict[str, Any]:
        return await sample_episode(sampler, task, problem, plans[problem.id], run_name=run_name, sample_idx=sample_idx,
                                    seed=config.seed_offset + sample_idx, max_tokens=config.max_tokens, temperature=config.temperature,
                                    top_p=config.top_p)

    jobs = [asyncio.ensure_future(one(p, i)) for p in problems for i in range(config.samples_per_problem)]
    counts = {"expected": len(jobs), "rollouts": 0, "complete": 0, "incomplete": 0, "errors": 0}
    errors: list[dict] = []

    def finish(status: str, **extra) -> None:
        manifest.update({"status": status, "finished_at": datetime.now(UTC).isoformat(), "counts": counts,
                         "errors": errors, **extra, "cost": {"usd": tracker.run_cost, "summary": tracker.summary(),
                                  "provisional_models": sorted(tracker.provisional_models)}})
        write_json(manifest_path, manifest)

    try:
        with (run_dir / "rollouts.jsonl").open("w") as f:
            for done in asyncio.as_completed(jobs):
                out = await done
                if out["error"]:
                    errors.append(out["error"])
                    counts["errors"] += 1
                if "row" in out:
                    f.write(json.dumps(out["row"]) + "\n")
                    f.flush()
                    counts["rollouts"] += 1
                    if out["row"]["status"] != "error":  # errors are already counted above
                        counts[out["row"]["status"]] += 1
    except BaseException as exc:  # e.g. Ctrl-C: stop the remaining calls, keep what was stored, record why
        for job in jobs:
            job.cancel()
        await asyncio.gather(*jobs, return_exceptions=True)
        finish("failed", error={"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        raise
    finish("complete" if not errors else "complete_with_errors")

    print(f"{run_dir}: {counts['rollouts']}/{counts['expected']} episodes stored ({counts['complete']} complete, "
          f"{counts['incomplete']} incomplete, {counts['errors']} errors); cost {tracker.summary()} (cache hits are free)")
    if errors:
        raise SystemExit(f"{len(errors)} episodes hit errors; see {manifest_path}")
