"""One episode = a task's fixed user turns sampled in order. Each later prompt is the exact earlier prompt and
completion tokens plus the next user turn (the history is never re-rendered). An episode stops at the first turn that
did not finish, and keeps the turns already sampled if a later call fails. Shared by rollouts, training and eval."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.backends.tinker.clients import continuation_tokens, normalized_messages
from src.backends.tinker.sampling import Sampler, SampleResult
from src.utils.costs import context_window


@dataclass(frozen=True)
class EpisodePlan:
    """Everything an episode needs that can be computed (and fail) before any paid call."""

    users: list[str]
    messages: list[dict]  # the first prompt's messages as the task wrote them (system message included, if any)
    normalized: list[dict] | None  # the first prompt's messages as rendered (None: the renderer cannot say)
    prompt: list[int]
    continuations: list[list[int]]
    end_of_turn: int


def plan_episode(task, problem, tokenizer, renderer) -> EpisodePlan:
    users = task.user_turns(problem.inputs)
    system = getattr(task, "system_prompt", lambda inputs: None)(problem.inputs)
    prefix = getattr(task, "prefix_messages", lambda inputs, problem_id: [])(problem.inputs, problem.id)  # few-shot
    first = ([{"role": "system", "content": system}] if system else []) + prefix + [{"role": "user", "content": users[0]}]
    return EpisodePlan(users, first, normalized_messages(renderer, first), renderer.build_generation_prompt(first).to_ints(),
                       [continuation_tokens(tokenizer, renderer, u, system) for u in users[1:]],
                       renderer.get_stop_sequences()[0])


def ended(turn: dict, end_of_turn: int) -> bool:
    """Whether a sampled turn ended cleanly with the renderer's end-of-turn token."""
    return (turn["parse_status"] == "clean" and turn["stop_reason"] == "stop"
            and turn["completion_tokens"][-1:] == [end_of_turn])


def _turn(user: str, result: SampleResult) -> dict[str, Any]:
    turn = asdict(result)
    turn["completion_tokens"] = turn.pop("tokens")
    turn["completion_logprobs"] = turn.pop("logprobs")
    turn["parse_status"] = "clean" if turn.pop("is_clean") else "unclean"
    return {"user": user, **turn}


async def sample_episode(sampler: Sampler, task, problem, plan: EpisodePlan, *, run_name: str, sample_idx: int,
                         seed: int, max_tokens: int, temperature: float, top_p: float) -> dict[str, Any]:
    """{"row": the lossless episode row, "error": ...}; no row if the first call failed."""
    turns: list[dict] = []
    prompt, error = plan.prompt, None
    rollout_id = f"{problem.id}-s{sample_idx}"
    try:
        for k, user in enumerate(plan.users):
            if k:
                prompt = prompt + turns[-1]["completion_tokens"] + plan.continuations[k - 1]
            window = context_window(sampler.model)  # never past the context window, and never truncating history
            allowance = max_tokens if window is None else min(max_tokens, window - len(prompt))
            if allowance <= 0:
                raise ValueError(f"turn {k + 1}: prompt of {len(prompt)} tokens fills the {window}-token context")
            result = await sampler.sample_tokens(prompt, sample_idx=seed, max_tokens=allowance,
                                                 temperature=temperature, top_p=top_p)
            turns.append(_turn(user, result) | ({"max_tokens": allowance} if allowance < max_tokens else {}))
            turns[-1]["ended"] = ended(turns[-1], plan.end_of_turn)  # graders honour this too
            if not turns[-1]["ended"]:
                break  # nothing to continue from
    except Exception as exc:  # noqa: BLE001 - keep sampled turns; the caller decides what a failure means
        error = {"rollout_id": rollout_id, "turn": len(turns), "type": type(exc).__name__, "message": str(exc)}
    if not turns:
        return {"error": error}
    complete = error is None and len(turns) == len(plan.users) and ended(turns[-1], plan.end_of_turn)
    row = {
        "run_name": run_name,
        "rollout_id": rollout_id,
        "task": task.spec(),
        "problem": problem.to_row(),
        "sample_idx": sample_idx,
        "model": sampler.model,
        "model_path": sampler.model_path,
        "renderer": sampler.renderer_name,
        "sampling": {"max_tokens": max_tokens, "temperature": temperature, "top_p": top_p, "seed": seed},
        "normalized_messages": plan.normalized,
        "messages": plan.messages,
        "status": "error" if error else ("complete" if complete else "incomplete"),
        "turns": turns,
    }
    if error:
        row["error"] = error
    return {"row": row, "error": error}
