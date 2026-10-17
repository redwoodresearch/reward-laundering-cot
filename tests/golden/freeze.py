"""What the published experiments saw, computed through the production code paths: the episode planner's first-prompt
messages for the first questions of every published task setup (warm-ups included), the trainer's question draws
(single-rule and the five-rule curriculum, from its recorded definition), the shuffled-reward permutation, the
trainer's learning rates, and grader verdicts on real saved episodes. golden.json holds these values as computed by
the research code before the public-release refactor (its `baseline` records the revision); tests/test_golden.py
requires the code to reproduce them exactly."""

from __future__ import annotations

from functools import cache

from src.dataset.pools import load_pool, pool_task
from src.rollouts import episodes
from src.tasks import Problem
from src.training.curricula import CURRICULA
from src.training.grpo import Component, _shuffled, learning_rate, step_items
from tests.fakes import FakeRenderer, FakeTokenizer

# (label, pool, task_options, splits) for every task setup behind the post's figures
SETUPS = [
    ("repeat_sentences", "gsm8k_gate", "", ("train", "val")),
    ("lowercase", "gsm8k_lowercase_thinking_rl_gate", '{"warmup": "v1", "warmup_k": 1}', ("train", "val")),
    ("ok_start", "gsm8k_ok_start_sentence_rl_gate", '{"warmup": "v1", "warmup_k": 1}', ("train", "val")),
    ("word_suppression", "gsm8k_word_suppression_rl_gate", '{"warmup": "v1", "warmup_k": 2}', ("train", "val")),
    ("meow", "gsm8k_meow_between_words_rl_gate", '{"warmup": "v1", "warmup_k": 1}', ("train", "val")),
    ("direct", "repeat_direct_correct", "", ("train", "eval")),
]
N_PROMPTS = 4
POST_SCHEDULE = {"n_steps": 34, "lr_schedule_steps": 68}  # every published run's learning-rate settings


def first_messages(task, problem: Problem) -> dict:
    """The first prompt's messages as the episode planner assembles them, and the later user turns (rendering is
    stubbed: messages are what the tasks write; the renderer is the tinker cookbook's)."""
    original = episodes.continuation_tokens
    episodes.continuation_tokens = lambda *args, **kwargs: []
    try:
        plan = episodes.plan_episode(task, problem, FakeTokenizer(), FakeRenderer())
    finally:
        episodes.continuation_tokens = original
    return {"id": problem.id, "first": plan.messages, "later_users": plan.users[1:]}


@cache
def prompts() -> dict:
    out = {}
    for label, pool, options, splits in SETUPS:
        task = pool_task(pool, options)
        out[label] = {"spec": task.spec(), "sizes": {s: len(load_pool(pool, s)) for s in splits},
                      **{s: [first_messages(task, p) for p in load_pool(pool, s)[:N_PROMPTS]] for s in splits}}
    return out


def _component(key: str, pool: str, options: str) -> Component:
    return Component(key, pool, pool_task(pool, options), load_pool(pool, "train"), [])


def orders() -> dict:
    """Training question ids per step, as the trainer draws them: single-rule runs (batch 8, seeds 0-3) and the
    five-rule curriculum (batch 10) from CURRICULA's recorded definition."""
    single = {f"{label}/seed{seed}": [[p.id for _, p in step_items([_component(label, pool, options)], step, 8, seed)]
                                      for step in (0, 1, 33)]
              for label, pool, options, _ in SETUPS for seed in range(4)}
    parts = [_component(*c) for c in CURRICULA["five_rules_v1"]]
    curriculum = {f"seed{seed}": [[p.id for _, p in step_items(parts, step, 10, seed)] for step in (0, 1, 33)]
                  for seed in range(4)}
    return {"single": single, "curriculum": curriculum}


def curriculum_definition() -> list:
    return [list(c) for c in CURRICULA["five_rules_v1"]]


def shuffles() -> list:
    return [_shuffled(list(range(16)), seed, step, group) for seed in (0, 1) for step in (0, 5) for group in (0, 7)]


def learning_rates() -> list:
    from main import TrainConfig

    config = TrainConfig(**POST_SCHEDULE)
    return [learning_rate(config, step) for step in range(config.n_steps)]


def grades(cases: list[dict]) -> list:
    out = []
    for case in cases:
        task = pool_task(case["pool"], case["task_options"])
        grade = task.grade(Problem.from_row(case["problem"]), case["turns"])
        out.append({"grade": grade, "reward": task.reward(grade)})
    return out
