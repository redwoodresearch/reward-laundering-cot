"""`train` command: GRPO on a pool's train split, as in the reward-laundering experiment (outputs: README).

Each step samples `group_size` episodes per problem for `batch_size` problems, centres rewards within each group and
takes one importance-sampling step on the groups whose rewards differ. Step files are written atomically and never
rewritten, so the Docent streamer only ever reads whole steps."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import time
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import chz
import tinker
import wandb
from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.data_processing import assemble_training_data, compute_advantages, trajectory_to_data
from tinker_cookbook.rl.metrics import compute_kl_sample_train
from tinker_cookbook.rl.types import Trajectory, TrajectoryGroup, Transition
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.lr_scheduling import compute_schedule_lr_multiplier

from src.backends.tinker.clients import (
    get_service_client,
    get_tokenizer_and_renderer,
    renderer_settings,
    resolve_renderer_name,
)
from src.backends.tinker.sampling import Sampler
from src.dataset.pools import load_pool, pool_path, pool_sources, pool_task
from src.rollouts.episodes import plan_episode, sample_episode
from src.tasks import Problem
from src.training.curricula import CURRICULA
from src.utils.costs import PUBLISHED_FILE, CostTracker, context_window
from src.utils.files import write_json, write_jsonl
from src.utils.provenance import git_commit, git_dirty
from src.utils.stats import eval_precision, samples_for_margin


def _at(tree: dict, path: str):
    for key in path.split("/"):
        tree = tree.get(key) if isinstance(tree, dict) else None
    return tree


CHART_NOTES = ("eval/* = the fixed held-out questions, sampled from the policy before that step's update; the planned "
               "approximate 95% margin of eval/reward is +/-{margin:.0%} on these questions (not unseen ones) when "
               "every planned sample arrives (summaries.jsonl records whether it did). "
               "reward = the task's reward rate over stored episodes, incomplete ones counting 0: for the maths gates, "
               "the final (turn-2) answer is correct; for the direct task, the thinking follows the rule and the "
               "multiple-choice answer is correct. compliance = the thinking follows the run's rule (the recorded "
               "task's predicate), among episodes whose compliance could be judged. math_accuracy = the turn-1 answer "
               "is correct. leakage = rule-breakers whose final answer was still correct; leakage_math_correct = the "
               "same among rule-breakers whose turn-1 answer was correct. side_accuracy = multiple-choice correctness. "
               "thinking_skipped = thinking closed immediately. *_tokens = median completion length. "
               "train/groups_with_signal (grpo, grpo_shuffled) = fraction of the step's groups that completed without "
               "errors and got differing rewards (only those update the policy); train/groups_trained (sft objectives) "
               "= fraction of groups without API errors, all of whose episodes are imitated. cost/usd = cumulative "
               "estimated Tinker cost. Details: summaries.jsonl.")
TRAIN_CHARTS = ("reward", "completion_tokens")  # headline entries also charted for training episodes


def _summary(task, rows: list[dict], expected: int, failed: int) -> dict:
    """The task's full summary plus outcome and failure counts; the source of every chart."""
    return {**task.metrics([r["grade"] for r in rows], rows),
            "outcomes": {o: sum(r["grade"]["outcome"] == o for r in rows) for o in task.outcomes},
            "episodes_expected": expected, "episodes_received": len(rows), "failed_episodes": failed}


def _chart_components(parts: list[Component], detail: dict, groups_used: float | None, objective: str) -> dict[str, float]:
    """A curriculum's charts: each component's headline numbers under eval/<key>/ and train/<key>/."""
    chart = {}
    for part in parts:
        sub = {split: detail[split]["components"][part.key] for split in ("train", "eval") if split in detail}
        chart.update({f"{k.split('/')[0]}/{part.key}/{k.split('/', 1)[1]}": v for k, v in _chart(part.task, sub, None).items()})
    return _chart(parts[0].task, {}, groups_used, objective) | chart


def _chart(task, detail: dict, groups_used: float | None, objective: str = "grpo") -> dict[str, float]:
    """The handful of wandb series: the task's headline numbers on eval, its reward (and completion length, where the
    task charts it) on training, and the fraction of training groups that carried any signal. Everything else stays
    in summaries.jsonl."""
    chart = {}
    for split, names in (("eval", task.headline), ("train", [n for n in TRAIN_CHARTS if n in task.headline])):
        if split in detail:
            chart.update({f"{split}/{n}": _at(detail[split], task.headline[n]) for n in names})
    if groups_used is not None:
        chart["train/groups_with_signal" if objective.startswith("grpo") else "train/groups_trained"] = groups_used
    return {k: v for k, v in chart.items() if v is not None}  # e.g. a rate with no known values


def _shuffled(rewards: list[float], seed: int, step: int, group: int) -> list[float]:
    """A uniform permutation of a group's rewards (the identity included), seeded by (seed, step, group)."""
    order = list(range(len(rewards)))
    random.Random(f"{seed}-{step}-{group}").shuffle(order)
    return [rewards[j] for j in order]


def _batch_problems(problems: list[Problem], step: int, batch_size: int, seed: int) -> list[Problem]:
    """Step `step`'s problems: epochs of the pool, each in its own seeded order."""
    out = []
    for i in range(step * batch_size, (step + 1) * batch_size):
        epoch, j = divmod(i, len(problems))
        order = list(range(len(problems)))
        random.Random(seed + epoch).shuffle(order)
        out.append(problems[order[j]])
    return out


def step_items(parts: list, step: int, batch_size: int, seed: int) -> list[tuple[int, Problem]]:
    """Step `step`'s (component index, problem) pairs. A curriculum draws batch_size / len(parts) questions per rule,
    each rule in its own independently seeded order (the same question can come up under two rules)."""
    if len(parts) == 1:
        return [(0, p) for p in _batch_problems(parts[0].train, step, batch_size, seed)]
    per = batch_size // len(parts)
    return [(k, p) for k, part in enumerate(parts) for p in _batch_problems(part.train, step, per, seed + 10_000 * k)]


def learning_rate(config, step: int) -> float:
    """The LR of update `step` (0..n_steps-1): the schedule over lr_schedule_steps (default n_steps)."""
    return config.learning_rate * compute_schedule_lr_multiplier(config.lr_schedule, step,
                                                                 config.lr_schedule_steps or config.n_steps)


def _trajectory(row: dict) -> Trajectory:
    """One transition per turn; each turn's prompt extends the previous turn's, so the cookbook makes one datum."""
    return Trajectory(
        transitions=[Transition(ob=tinker.ModelInput.from_ints(t["prompt_tokens"]),
                                ac=TokensWithLogprobs(t["completion_tokens"], t["completion_logprobs"]),
                                reward=0.0, episode_done=k == len(row["turns"]) - 1)
                     for k, t in enumerate(row["turns"])],
        final_ob=tinker.ModelInput.empty())


@dataclass(frozen=True)
class Component:
    """One rule a run trains on: its task (prompt and grader) and question splits. A plain run has one; a curriculum
    (src/training/curricula.py) several, each drawing batch_size / len(components) questions per step."""
    key: str
    pool: str
    task: Any
    train: list[Problem]
    eval: list[Problem]


class Trainer:
    def __init__(self, config, run_dir: Path):
        self.config = config
        self.run_dir = run_dir
        if config.curriculum and (config.task_options or config.docent):
            raise ValueError("a curriculum sets each component's task_options (pass none) and is not exported to Docent "
                             "(scripts/docent reads one task per run; pass docent=false)")
        specs = CURRICULA[config.curriculum] if config.curriculum else ((config.pool, config.pool, config.task_options),)
        self.parts: list[Component] = []
        for key, pool, options in specs:
            task = pool_task(pool, options)
            if not (hasattr(task, "metrics") and hasattr(task, "headline")):  # fail before any paid setup
                raise ValueError(f"{pool}'s task {task.name} has no training metrics/headline")
            # eval_n_problems=0 disables eval: its split is then neither read nor recorded
            self.parts.append(Component(key, pool, task, load_pool(pool, "train"),
                                        load_pool(pool, config.eval_split)[: config.eval_n_problems] if config.eval_n_problems else []))
        self.multi = len(self.parts) > 1
        if config.batch_size % len(self.parts):
            raise ValueError(f"batch_size {config.batch_size} must split evenly over {len(self.parts)} components")
        self.task = self.parts[0].task  # a plain run's task
        if config.lr_schedule_steps is not None and config.lr_schedule_steps < config.n_steps:
            raise ValueError(f"lr_schedule_steps {config.lr_schedule_steps} is shorter than n_steps {config.n_steps}")
        self.train_problems, self.eval_problems = self.parts[0].train, self.parts[0].eval
        # Repeat each eval question so each component's reward rate has an approximate 95% margin of `eval_margin`.
        self.eval_samples = samples_for_margin(len(self.eval_problems), config.eval_margin) if self.eval_problems else 0
        self.tracker = CostTracker(Path(config.cost_file), run_description=f"train {run_dir.name}")
        self.model_path: str | None = None  # None = the base model (step 0's LoRA is the identity)
        self.renderer_name = resolve_renderer_name(config.model_name, config.renderer_name)
        self.tokenizer, self.renderer = get_tokenizer_and_renderer(config.model_name, self.renderer_name)
        # the same GSM8K question appears under several rules, so plans are keyed by (component, question)
        self.plans = {(k, p.id): plan_episode(part.task, p, self.tokenizer, self.renderer)
                      for k, part in enumerate(self.parts) for p in part.train + part.eval}

    @property
    def name(self) -> str:
        """The pool, or curriculum:<name> with its pools."""
        c = self.config
        return f"curriculum:{c.curriculum} ({', '.join(p.pool for p in self.parts)})" if c.curriculum else c.pool

    def sampler(self) -> Sampler:
        return Sampler(model=self.config.model_name, model_path=self.model_path, renderer_name=self.renderer_name,
                       tracker=self.tracker, max_concurrency=self.config.max_concurrency)

    async def episodes(self, sampler: Sampler, items: list[tuple[int, Problem]], n: int, seed: int, tag: str) -> list[list[dict]]:
        """`n` episodes per (component, problem), seeded per (step, problem slot, sample); turns of one episode share
        its seed."""
        c = self.config

        async def one(i: int, k: int, p: Problem, g: int) -> dict:
            out = await sample_episode(sampler, self.parts[k].task, p, self.plans[(k, p.id)],
                                       run_name=f"{self.run_dir.name}/{tag}", sample_idx=g, seed=seed + i * n + g,
                                       max_tokens=c.max_tokens, temperature=c.temperature, top_p=c.top_p)
            if out["error"]:
                out["error"] = {"step_file": tag, "group": i, "sample_idx": g, "problem_id": p.id,
                                **({"component": self.parts[k].key} if self.multi else {}),
                                "seed": seed + i * n + g, "model_path": sampler.model_path, **out["error"]}
            return out

        outs = await asyncio.gather(*(asyncio.gather(*(one(i, k, p, g) for g in range(n)))
                                      for i, (k, p) in enumerate(items)))
        errors = [o["error"] for group in outs for o in group if o["error"]]
        if errors:  # every failed call, with enough identity to find or retry it
            with (self.run_dir / "errors.jsonl").open("a") as f:
                f.writelines(json.dumps(e) + "\n" for e in errors)
        return outs

    def graded(self, out: dict, k: int, problem: Problem) -> dict:
        row, task = out["row"], self.parts[k].task
        grade = task.grade(problem, row["turns"])
        return {**row, **({"component": self.parts[k].key} if self.multi else {}), "grade": grade, "reward": task.reward(grade)}

    def summaries(self, items: list[tuple[int, Problem]], per_item: list[list[dict]], outs: list[list[dict]], n: int) -> dict:
        """_summary per component (a plain run: the one summary, unnested)."""
        def one(k: int) -> dict:
            idx = [i for i, (kk, _) in enumerate(items) if kk == k]
            return _summary(self.parts[k].task, [r for i in idx for r in per_item[i]], len(idx) * n,
                            failed=sum(1 for i in idx for o in outs[i] if o["error"]))
        return {"components": {part.key: one(k) for k, part in enumerate(self.parts)}} if self.multi else one(0)

    async def evaluate(self, step: int, sampler: Sampler) -> dict:
        start = time.time()
        items = [(k, p) for k, part in enumerate(self.parts) for p in part.eval]
        outs = await self.episodes(sampler, items, self.eval_samples, seed=10**9 + step * 10**6, tag=f"eval/step_{step:04d}")
        per_question = [[{**self.graded(o, k, p), "policy_step": step, "model_path": sampler.model_path}
                         for o in group if "row" in o] for (k, p), group in zip(items, outs, strict=True)]
        write_jsonl(self.run_dir / "eval" / f"step_{step:04d}.jsonl", [r for group in per_question for r in group])
        detail = self.summaries(items, per_question, outs, self.eval_samples)
        for k, d in enumerate(detail["components"].values() if self.multi else [detail]):
            # An episode cut short by an API error keeps its row but is not an observation of the policy.
            d["reward_precision"] = eval_precision([[r["reward"] for r in group if "error" not in r]
                                                    for (kk, _), group in zip(items, per_question, strict=True) if kk == k],
                                                   self.eval_samples, self.config.eval_margin)
        return {"eval": {**detail, "seconds": time.time() - start}}

    async def train_step(self, step: int, sampler: Sampler, training_client) -> dict:
        c = self.config
        start = time.time()
        items = step_items(self.parts, step, c.batch_size, c.seed)
        if c.objective == "sft_base_rollouts":  # the same problems and seeds, sampled from the untrained model
            sampler = Sampler(model=c.model_name, model_path=None, renderer_name=self.renderer_name,
                              tracker=self.tracker, max_concurrency=c.max_concurrency)
        outs = await self.episodes(sampler, items, c.group_size, seed=step * 10**6, tag=f"train/step_{step:04d}")
        sampled = time.time()

        # A group with any failed call is dropped whole: its survivors' advantages would be biased.
        rows, groups, used_rows, statuses, errors = [], [], [], [], 0
        per_item = []
        for i, ((k, problem), group) in enumerate(zip(items, outs, strict=True)):
            failed = any(o["error"] for o in group)
            errors += sum(1 for o in group if o["error"])
            graded = [self.graded(o, k, problem) for o in group if "row" in o]
            rewards = credited = [r["reward"] for r in graded]
            if c.objective == "grpo_shuffled":  # the advantage follows a permuted reward, not the episode's own
                credited = _shuffled(rewards, c.shuffle_seed, step, i)
            if c.objective.startswith("grpo"):
                used = not failed and len(set(rewards)) > 1
                advantages = [r - sum(credited) / len(credited) for r in credited] if used else [0.0] * len(rewards)
            else:  # no reward signal: every episode of a complete group is imitated
                used, advantages = not failed, [None] * len(rewards)
            statuses.append("failed" if failed else ("used" if used else "constant"))
            for r, a, cr in zip(graded, advantages, credited, strict=True):
                rows.append({**r, "policy_step": step, "model_path": sampler.model_path, "group": i, "group_sample": r["sample_idx"],
                             "advantage": a, "group_status": statuses[-1],
                             **({"credited_reward": cr} if c.objective == "grpo_shuffled" else {})})
            per_item.append(rows[len(rows) - len(graded):])
            if used:
                groups.append(TrajectoryGroup([_trajectory(r) for r in graded], credited, [{} for _ in graded]))
                used_rows += rows[-len(graded):]
        write_jsonl(self.run_dir / "train" / f"step_{step:04d}.jsonl", rows)

        detail = self.summaries(items, per_item, outs, c.group_size)
        # Groups with differing rewards are the only ones that move the policy.
        detail.update({"groups": {s: statuses.count(s) for s in ("used", "constant", "failed")},
                       "sample_seconds": sampled - start})

        lr = learning_rate(c, step)
        detail["optim"] = {"lr": lr}
        if groups:
            if c.objective.startswith("grpo"):
                data, _ = assemble_training_data(groups, compute_advantages(groups))  # every token carries its advantage
                batch = [tinker.Datum(model_input=d.model_input,
                                      loss_fn_inputs={k: v for k, v in d.loss_fn_inputs.items() if k != "mask"})
                         for d in data]
                loss_fn = "importance_sampling"
            else:  # cross-entropy on each episode's sampled tokens (the GRPO mask): weight 1 each, or 1/length (mean)
                data = [d for g in groups for t in g.trajectories_G for d in trajectory_to_data(t, 0.0)]
                batch = [tinker.Datum(model_input=d.model_input, loss_fn_inputs={
                    "target_tokens": d.loss_fn_inputs["target_tokens"],
                    "weights": tinker.TensorData.from_torch(mask / mask.sum() if c.sft_reduction == "mean" else mask)})
                    for d in data for mask in [d.loss_fn_inputs["mask"].to_torch().float()]]
                loss_fn = "cross_entropy"
            train_tokens = sum(d.model_input.length for d in data)
            self.tracker.add_tinker_cost(c.model_name, train_tokens=train_tokens)
            fwd = await training_client.forward_backward_async(batch, loss_fn=loss_fn)
            opt = await training_client.optim_step_async(tinker.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8))
            fwd_out = await fwd.result_async()
            learner = [o["logprobs"].to_torch() for o in fwd_out.loss_fn_outputs]
            # The learner's logprobs of the sampled tokens (completion-aligned; one datum per used episode, in order),
            # saved before the optimizer result so its failure cannot lose them.
            write_jsonl(self.run_dir / "optim" / f"step_{step:04d}.jsonl", [
                {"rollout_id": r["rollout_id"], "group": r["group"], "group_sample": r["group_sample"],
                 "learner_logprobs": lp[d.loss_fn_inputs["mask"].to_torch() > 0].tolist()}
                for r, d, lp in zip(used_rows, data, learner, strict=True)])
            opt_out = await opt.result_async()
            # kl_sample_train_* is the sampler-vs-learner logprob gap: on the same weights for grpo and sft_on_policy,
            # base model vs learner for sft_base_rollouts (whose sampler entropy is the base model's); never a KL to base.
            kl = compute_kl_sample_train(data, learner)
            extra = {} if c.objective.startswith("grpo") else {"nll": compute_mean_nll(
                [o["logprobs"] for o in fwd_out.loss_fn_outputs], [d.loss_fn_inputs["weights"] for d in batch])}
            detail["optim"].update({**fwd_out.metrics, **(opt_out.metrics or {}), **kl, **extra, "datums": len(data),
                                    "train_tokens": train_tokens})
            self.model_path = await self.save(training_client, step + 1)
        detail["train_seconds"] = time.time() - sampled
        return {"train": detail}

    async def save(self, training_client, step: int, kind: str = "sampler") -> str:
        name = f"{self.run_dir.name}_step{step:04d}"
        if kind == "sampler":
            path = (await (await training_client.save_weights_for_sampler_async(name=name)).result_async()).path
        else:
            path = (await (await training_client.save_state_async(name=name)).result_async()).path
        c = self.config
        what = "sampler weights" if kind == "sampler" else "full training state"
        with Path(c.checkpoint_index).open("a") as f:  # noqa: ASYNC230 - one tiny append
            f.write(json.dumps({
                "run": self.run_dir.name, "run_dir": str(self.run_dir.resolve()), "step": step, "kind": kind,
                "path": path, "model": c.model_name, "task": self.task.spec() if not self.multi else
                {part.key: part.task.spec() for part in self.parts}, "pool": self.name,
                "datasets": {f"{part.key}/{role}" if self.multi else role: str(pool_path(part.pool, split))
                             for part in self.parts for role, split in (("train", "train"), ("eval", c.eval_split))
                             if role == "train" or c.eval_n_problems},
                "description": f"{what} after {step} {c.objective} training steps on {self.name} ({c.batch_size} problems x "
                               f"{c.group_size} samples per step, lr {c.learning_rate:g} {c.lr_schedule}, LoRA rank "
                               f"{c.lora_rank})",
                "saved_at": datetime.now(UTC).isoformat()}) + "\n")
        return path


def _component_receipt(part: Component, config) -> dict:
    """What a run records about one component: its task, grader, warm-up and question files."""
    return {"task": part.task.spec(), "grader": part.task.grader,
            **({"warmup": w} if (w := getattr(part.task, "warmup_receipt", lambda ps: None)(part.train + part.eval)) else {}),
            "pool": {"name": part.pool, "sources": pool_sources(part.pool),
                     **{role: {"split": split, "file": str(pool_path(part.pool, split)),
                               "sha256": hashlib.sha256(pool_path(part.pool, split).read_bytes()).hexdigest()}
                        for role, split in (("train", "train"), ("eval", config.eval_split))
                        if role == "train" or config.eval_n_problems},
                     "eval_problem_ids": [p.id for p in part.eval]}}


async def run(config) -> Path:
    run_name = config.run_name or f"{config.curriculum or config.pool}_{config.objective}_{datetime.now(UTC):%Y%m%d_%H%M%S}"
    run_dir = Path(config.output_dir) / run_name
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} exists; runs are never overwritten")
    trainer = Trainer(config, run_dir)
    for part in trainer.parts:
        if len(part.train) < config.batch_size // len(trainer.parts):
            raise ValueError(f"{part.key}: its share of batch_size exceeds its {len(part.train)} train problems")

    for sub in ("train", "eval", "optim"):
        (run_dir / sub).mkdir(parents=True)
    Path(config.checkpoint_index).parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "run_name": run_name, "status": "running", "started_at": datetime.now(UTC).isoformat(),
        "config": chz.asdict(config),
        **(_component_receipt(trainer.parts[0], config) if not trainer.multi else
           {"curriculum": config.curriculum, "components": {part.key: _component_receipt(part, config) for part in trainer.parts}}),
        "eval_samples_per_problem": trainer.eval_samples,
        "model": config.model_name, "renderer": trainer.renderer_name,
        "renderer_settings": renderer_settings(trainer.tokenizer, trainer.renderer,
                                               trainer.plans[(0, trainer.train_problems[0].id)].prompt),
        "pricing_snapshot": PUBLISHED_FILE.name, "context_window": context_window(config.model_name),
        "tokenizer": getattr(trainer.tokenizer, "name_or_path", None),
        "versions": {pkg: version(pkg) for pkg in ("tinker", "tinker-cookbook", "transformers", "wandb")},
        "git_commit": git_commit(), "git_dirty": git_dirty(),
    }
    write_json(run_dir / "manifest.json", manifest)

    logger = None
    try:  # from here on, any failure is recorded in the manifest (the Docent streamer stops on it)
        if config.wandb_project:  # no machine-stats charts: this machine only orchestrates; the GPUs are Tinker's
            wandb.setup(settings=wandb.Settings(x_disable_stats=True))
        logger = ml_log.setup_logging(str(run_dir), wandb_project=config.wandb_project, wandb_name=run_name,
                                      config=config, do_configure_logging_module=False)
        if config.wandb_project and wandb.run is None:  # the cookbook falls back to local logging silently
            raise RuntimeError(f"wandb_project={config.wandb_project!r} but no wandb run started (WANDB_API_KEY?)")
        # The cookbook prints each step's table through its module logger; show it without enabling SDK/HTTP logs.
        console = logging.getLogger(ml_log.__name__)
        console.setLevel(logging.INFO)
        if not console.handlers:
            console.addHandler(logging.StreamHandler())
        if wandb.run is not None:  # the config itself is logged by setup_logging; add what is derived from it
            wandb.run.notes = CHART_NOTES.format(margin=config.eval_margin)
            wandb.run.config.update({"eval_samples_per_problem": trainer.eval_samples})
            manifest["wandb_url"] = wandb.run.url
            write_json(run_dir / "manifest.json", manifest)
        training_client = await get_service_client().create_lora_training_client_async(
            base_model=config.model_name, rank=config.lora_rank, seed=config.lora_seed)
        for step in range(config.n_steps + 1):
            due = step % config.eval_every == 0 or step == config.n_steps
            start = time.time()
            detail: dict[str, Any] = {"step": step}
            if due and step:  # full optimizer state of the policy this step evaluates (before its own update)
                manifest["final_state"] = await trainer.save(training_client, step, kind="state")
            sampler = trainer.sampler()  # this step's policy, shared by eval and training samples
            async with asyncio.TaskGroup() as tg:  # a failure in one job cancels the other before we record it
                jobs = [tg.create_task(trainer.evaluate(step, sampler))] if due and config.eval_n_problems else []
                if step < config.n_steps:
                    jobs.append(tg.create_task(trainer.train_step(step, sampler, training_client)))
            for job in jobs:
                detail.update(job.result())
            detail.update({"seconds": time.time() - start, "cost_usd": trainer.tracker.run_cost})
            groups = detail.get("train", {}).get("groups")
            used = groups and groups["used"] / config.batch_size
            chart = {**(_chart_components(trainer.parts, detail, used, config.objective) if trainer.multi
                        else _chart(trainer.task, detail, used, config.objective)), "cost/usd": trainer.tracker.run_cost}
            with (run_dir / "summaries.jsonl").open("a") as f:  # local record first: reporting below may fail
                f.write(json.dumps({**detail, "chart": chart}) + "\n")
            manifest.update({"steps_done": min(step + 1, config.n_steps), "model_path": trainer.model_path,
                             "cost": trainer.tracker.summary()})
            write_json(run_dir / "manifest.json", manifest)
            if groups and groups["failed"] == config.batch_size:
                raise RuntimeError(f"step {step}: every training group hit an API error (errors.jsonl); stopping")
            logger.log_metrics(chart, step=step)
            for split in ("train", "eval"):
                for key, d in (detail.get(split, {}).get("components", {}) if trainer.multi else {"": detail.get(split, {})}).items():
                    problems = {k: d[k] for k in ("failed_episodes",) if d.get(k)}
                    problems.update({k: n for k, n in d.get("outcomes", {}).items() if k in ("incomplete", "invalid_decision") and n})
                    if problems:
                        print(f"step {step} {split}{' ' + key if key else ''}: {problems} (see errors.jsonl / the step file)", flush=True)
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest.update({"status": "failed", "error": {"type": type(exc).__name__, "message": str(exc),
                                                       "traceback": traceback.format_exc()}})
        raise
    finally:
        manifest.update({"finished_at": datetime.now(UTC).isoformat(), "cost": trainer.tracker.summary(),
                         "cost_usd": trainer.tracker.run_cost,
                         "provisional_models": sorted(trainer.tracker.provisional_models)})
        write_json(run_dir / "manifest.json", manifest)
        if logger is not None:
            logger.close()
    print(f"{run_dir}: {config.n_steps} steps, final sampler {trainer.model_path}; cost {trainer.tracker.summary()}")
    return run_dir

