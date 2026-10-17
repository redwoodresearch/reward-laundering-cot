"""Per-process Tinker clients, tokenizers and renderers. Retries are left to the SDK's RetryConfig."""

from __future__ import annotations

from functools import lru_cache

import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer


@lru_cache(maxsize=1)
def get_service_client() -> tinker.ServiceClient:  # TINKER_API_KEY comes from .env via main.py
    return tinker.ServiceClient()


def close_service_client(status: str) -> None:
    """Finish this process's Tinker session, if one was opened. Every sampling/training client is a sub-session of
    it, so this is the only thing to close; otherwise the server times the session out once heartbeats stop."""
    if not get_service_client.cache_info().currsize:
        return
    try:
        get_service_client().close(status).result()
    finally:
        get_service_client.cache_clear()
        get_sampling_client.cache_clear()


@lru_cache(maxsize=8)
def get_sampling_client(base_model: str, model_path: str | None = None) -> tinker.SamplingClient:
    """Sampling client for the base model, or for a trained checkpoint (`tinker://...` path)."""
    if model_path is not None:
        return get_service_client().create_sampling_client(model_path=model_path)
    return get_service_client().create_sampling_client(base_model=base_model)


# Recommended renderers that disable thinking or drop it from history -> the cookbook's variant that keeps both on
# (a gate's later turn must see the earlier turn's thinking, and every task here is about thinking).
THINKING_RENDERERS = {"deepseekv3": "deepseekv3_thinking", "kimi_k26": "kimi_k26_preserve_thinking",
                      "nemotron3": "nemotron3_preserve_thinking",
                      "nemotron3_ultra": "nemotron3_ultra_preserve_thinking"}
HISTORY_CHECK = "HISTORY-THINKING-CHECK"


def resolve_renderer_name(model: str, renderer_name: str | None) -> str:
    """The explicit renderer, else the cookbook's recommended one for the model, switched to its thinking-preserving
    variant where one exists (THINKING_RENDERERS)."""
    if renderer_name:
        return renderer_name
    recommended = model_info.get_recommended_renderer_name(model)
    return THINKING_RENDERERS.get(recommended, recommended)


@lru_cache(maxsize=8)
def get_tokenizer_and_renderer(model: str, renderer_name: str) -> tuple[Tokenizer, renderers.Renderer]:
    """The cookbook renderer, with historical thinking kept wherever the renderer has that switch (e.g. qwen3_5,
    deepseekv3_thinking strip it by default): a gate's later turn must see the earlier turn's thinking."""
    tokenizer = get_tokenizer(model)
    renderer = renderers.get_renderer(renderer_name, tokenizer, model_name=model)
    if getattr(renderer, "strip_thinking_from_history", False):
        renderer.strip_thinking_from_history = False
    return tokenizer, renderer


def normalized_messages(renderer: renderers.Renderer, messages: list[dict]) -> list[dict] | None:
    """The messages as rendered (e.g. with Qwen3.8's injected system sentence), excluding the assistant
    generation prefix. Wraps the cookbook's private `_normalize_messages`; an opt-in test checks that
    rendering them reproduces the prompt tokens. None for renderers without that hook (each turn's `prompt_text`
    still records the exact prompt, injected text included)."""
    normalize = getattr(renderer, "_normalize_messages", None)
    return normalize(messages) if normalize is not None else None


def continuation_tokens(tokenizer: Tokenizer, renderer: renderers.Renderer, user_content: str,
                        system: str | None = None) -> list[int]:
    """Tokens that follow a cleanly stopped assistant turn (ending in the renderer's first stop token) to add a user
    turn and open the next assistant turn. Appending these to the exact earlier tokens avoids re-rendering the
    history, which would normalize the model's own whitespace. Derived from a dummy conversation, which must render as
    an extension of the first prompt and keep the assistant's thinking in history; the continuation starts after the
    first stop token following the dummy reply. `system`: the episode's own system message, if any, so the dummy has
    the same shape."""
    first = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": "x"}]
    reply = {"role": "assistant", "content": [{"type": "thinking", "thinking": HISTORY_CHECK}, {"type": "text", "text": "y"}]}
    prompt = renderer.build_generation_prompt(first).to_ints()
    full = renderer.build_generation_prompt([*first, reply, {"role": "user", "content": user_content}]).to_ints()
    name = type(renderer).__name__
    if full[: len(prompt)] != prompt:
        raise ValueError(f"{name}: a later turn's prompt does not extend the first prompt (history is re-rendered)")
    extension, end_of_turn = full[len(prompt):], renderer.get_stop_sequences()[0]
    if HISTORY_CHECK not in tokenizer.decode(extension):
        raise ValueError(f"{name}: drops the assistant's thinking from history; use a thinking-preserving renderer")
    end = next((i for i, t in enumerate(extension) if t == end_of_turn), None)
    if end is None or HISTORY_CHECK not in tokenizer.decode(extension[:end]) \
            or user_content not in tokenizer.decode(extension[end + 1:]):
        raise ValueError(f"{name}: the assistant reply does not end in the stop token {end_of_turn} before the next turn")
    return extension[end + 1:]


def renderer_settings(tokenizer: Tokenizer, renderer: renderers.Renderer, first_prompt: list[int]) -> dict:
    """What a run's renderer actually did: its class, the effective history-thinking switch (None: no such switch),
    and one decoded first prompt, which shows any injected system or reasoning-effort text."""
    return {"class": type(renderer).__name__,
            "strip_thinking_from_history": getattr(renderer, "strip_thinking_from_history", None),
            "example_first_prompt": tokenizer.decode(first_prompt)}
