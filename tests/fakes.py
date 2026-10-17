"""Fake renderer/tokenizer/sampling client: completions are token lists, ending in STOP means a clean turn."""

from types import SimpleNamespace

import tinker
from tinker_cookbook.renderers import ParseTermination

STOP = [0]


def sequence(tokens, stop_reason):
    """A sampled sequence as Tinker returns it, with a logprob per token."""
    return SimpleNamespace(sequences=[SimpleNamespace(tokens=tokens, logprobs=[-0.5] * len(tokens), stop_reason=stop_reason)])


class FakeTokenizer:
    def decode(self, tokens):  # the fake prompt [1, 2, 3] ends by opening a <think> block, as Qwen3-style prompts do
        return " ".join("<think>" if t == 3 else str(t) for t in tokens)


class FakeRenderer:
    def build_generation_prompt(self, messages):
        return tinker.ModelInput.from_ints([1, 2, 3])

    def get_stop_sequences(self):
        return STOP

    def _normalize_messages(self, messages):
        return [{"role": "system", "content": "injected"}, *messages]

    def parse_response(self, tokens):
        if tokens[-1:] == STOP:
            content = [{"type": "thinking", "thinking": "reasoning"}, {"type": "text", "text": "4"}]
            return {"role": "assistant", "content": content}, ParseTermination.STOP_SEQUENCE
        return {"role": "assistant", "content": "<think>\nunfinished"}, ParseTermination.MALFORMED


class FakeSamplingClient:
    def __init__(self, tokens):
        self.tokens = tokens
        self.calls = []

    async def sample_async(self, prompt, num_samples, sampling_params):
        self.calls.append(sampling_params)
        stop_reason = "stop" if self.tokens[-1:] == STOP else "length"
        return sequence(self.tokens, stop_reason)
