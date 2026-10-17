"""Cached, cost-tracked sampling from a Tinker model or checkpoint.

Each sample is one Tinker call seeded by `sample_idx`, cached under the exact prompt tokens and sampling
parameters sent. Unclean parses keep `answer_text`/`thinking_text` empty; `raw_text`/`tokens` hold the full
output. Whether a turn finished (e.g. not stopped at the length limit) is decided by the episode and grader.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from typing import Any

import tinker
from tinker_cookbook.renderers import get_text_content

from src.backends.tinker.clients import get_sampling_client, get_tokenizer_and_renderer, resolve_renderer_name
from src.utils.cache import FileCache
from src.utils.costs import CostTracker, require_price

# Bump if the meaning of a cache entry changes (not just its inputs).
_CACHE_UUID = "tinker-sample-v3"  # v3 stores per-token logprobs


@dataclass
class SampleResult:
    tokens: list[int]
    logprobs: list[float]  # one per sampled token
    stop_reason: str
    is_clean: bool  # the renderer found a proper end of turn
    raw_text: str
    answer_text: str  # "" unless is_clean
    thinking_text: str  # "" unless is_clean
    n_prompt_tokens: int
    n_sample_tokens: int
    prompt_tokens: list[int]
    prompt_text: str


@dataclass
class Sampler:
    """The base model (`model_path=None`) or a `tinker://` checkpoint of it. `renderer_name=None` uses the
    cookbook's recommended renderer, switched to its thinking-preserving variant where one exists
    (clients.resolve_renderer_name); `cache`/`tracker` are optional."""

    model: str
    model_path: str | None = None
    renderer_name: str | None = None
    cache: FileCache | None = None
    tracker: CostTracker | None = None
    max_concurrency: int = 32
    semaphore: asyncio.Semaphore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_concurrency < 1:
            raise ValueError(f"max_concurrency must be >= 1, got {self.max_concurrency}")
        if self.tracker is not None:
            require_price(self.model)  # fail before paying for a sample we could not price
        self.renderer_name = resolve_renderer_name(self.model, self.renderer_name)
        self.semaphore = asyncio.Semaphore(self.max_concurrency)

    async def sample_tokens(self, prompt_tokens: list[int], *, sample_idx: int = 0, max_tokens: int = 8192,
                            temperature: float = 1.0, top_p: float = 1.0) -> SampleResult:
        """Sample a completion of exact prompt tokens (e.g. a multi-turn history built by concatenation)."""
        tokenizer, renderer = get_tokenizer_and_renderer(self.model, self.renderer_name)
        prompt = tinker.ModelInput.from_ints(prompt_tokens)
        params = tinker.SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                                       seed=sample_idx, stop=renderer.get_stop_sequences())

        async def compute() -> dict[str, Any]:
            client = get_sampling_client(self.model, self.model_path)
            async with self.semaphore:
                resp = await client.sample_async(prompt=prompt, num_samples=1, sampling_params=params)
            seq = resp.sequences[0]
            if self.tracker is not None:
                self.tracker.add_tinker_cost(self.model, prefill_tokens=len(prompt_tokens), sample_tokens=len(seq.tokens))
            if seq.logprobs is None or len(seq.logprobs) != len(seq.tokens) or not all(map(math.isfinite, seq.logprobs)):
                raise ValueError("sampler did not return one finite logprob per token")
            return {"tokens": list(seq.tokens), "logprobs": list(seq.logprobs), "stop_reason": seq.stop_reason}

        if self.cache is None:
            value = await compute()
        else:
            key = {"model": self.model, "model_path": self.model_path, "prompt_tokens": prompt_tokens,
                   "params": params.model_dump(), "uuid": _CACHE_UUID}
            value = await self.cache.aget_or_compute_set(key, compute)

        tokens = value["tokens"]
        message, termination = renderer.parse_response(tokens)
        # An unclean parse returns the raw completion as content, so only a clean one yields answer/thinking.
        answer_text = thinking_text = ""
        if termination.is_clean:
            answer_text = get_text_content(message)
            if isinstance(message["content"], list):
                thinking_text = "\n".join(p["thinking"] for p in message["content"] if p["type"] == "thinking")
        return SampleResult(
            tokens=tokens,
            logprobs=value["logprobs"],
            stop_reason=value["stop_reason"],
            is_clean=termination.is_clean,
            raw_text=tokenizer.decode(tokens),
            answer_text=answer_text,
            thinking_text=thinking_text,
            n_prompt_tokens=len(prompt_tokens),
            n_sample_tokens=len(tokens),
            prompt_tokens=prompt_tokens,
            prompt_text=tokenizer.decode(prompt_tokens),
        )
