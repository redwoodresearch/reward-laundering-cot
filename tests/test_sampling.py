"""Sampler wiring against a fake Tinker client and renderer (no network, no key).

Tests marked `real_renderer` check our assumptions about the installed Qwen3.8 renderer; they download
the tokenizer from HF on first run, so they are opt-in: `pytest -m real_renderer`.
"""

import atexit
import json
from types import SimpleNamespace

import pytest

from src.backends.tinker import sampling
from src.backends.tinker.sampling import Sampler
from src.utils import costs
from src.utils.cache import FileCache
from src.utils.costs import CostTracker
from tests.fakes import STOP, FakeRenderer, FakeSamplingClient, FakeTokenizer

MODEL = "Qwen/Qwen3.8-27B"
PROMPT = [1, 2, 3]
MESSAGES = [{"role": "user", "content": "What is 2+2?"}]  # for the real-renderer tests


@pytest.fixture(autouse=True)
def fake_renderer(monkeypatch):
    monkeypatch.setattr(sampling, "get_tokenizer_and_renderer", lambda model, name: (FakeTokenizer(), FakeRenderer()))
    monkeypatch.setattr(sampling, "resolve_renderer_name", lambda model, name: name or "fake")


def use_client(monkeypatch, tokens):
    client = FakeSamplingClient(tokens)
    monkeypatch.setattr(sampling, "get_sampling_client", lambda model, model_path=None: client)
    return client


@pytest.fixture
def tracker(tmp_path):
    t = CostTracker(tmp_path / "costs.jsonl")
    atexit.unregister(t._save_on_exit)
    return t


async def test_clean_sample_parses_and_request_carries_seed_and_stop(monkeypatch):
    client = use_client(monkeypatch, [7, 8, 0])
    result = await Sampler(model=MODEL).sample_tokens(PROMPT, sample_idx=7, max_tokens=64, temperature=0.5)
    assert (result.answer_text, result.thinking_text, result.is_clean, result.raw_text) == ("4", "reasoning", True, "7 8 0")
    params = client.calls[0]
    assert (params.seed, params.max_tokens, params.temperature, params.stop) == (7, 64, 0.5, STOP)


async def test_truncated_sample_has_no_answer(monkeypatch):
    use_client(monkeypatch, [7, 8, 9])
    result = await Sampler(model=MODEL).sample_tokens(PROMPT, max_tokens=3)
    assert not result.is_clean and result.stop_reason == "length"
    assert result.answer_text == "" and result.thinking_text == ""
    assert result.raw_text == "7 8 9"


@pytest.mark.parametrize("logprobs", [None, [-0.5, -0.5], [float("nan"), -0.5, -0.5]])
async def test_missing_misaligned_or_nonfinite_logprobs_are_rejected(monkeypatch, logprobs):
    client = use_client(monkeypatch, [7, 8, 0])

    async def bad(prompt, num_samples, sampling_params):
        return SimpleNamespace(sequences=[SimpleNamespace(tokens=[7, 8, 0], logprobs=logprobs, stop_reason="stop")])

    client.sample_async = bad
    with pytest.raises(ValueError, match="finite logprob"):
        await Sampler(model=MODEL).sample_tokens(PROMPT)


async def test_cache_hit_skips_call_and_cost(monkeypatch, tracker, tmp_path):
    client = use_client(monkeypatch, [7, 0])
    sampler = Sampler(model=MODEL, cache=FileCache(tmp_path / "cache"), tracker=tracker)
    first = await sampler.sample_tokens(PROMPT)
    cost_after_first = tracker.run_cost
    second = await sampler.sample_tokens(PROMPT)
    assert len(client.calls) == 1
    assert cost_after_first > 0 and tracker.run_cost == cost_after_first
    assert second.tokens == first.tokens
    assert tracker.summary().endswith(" estimate")  # published rate: still an estimate, no longer "provisional"


async def test_cache_key_is_the_exact_request(monkeypatch, tmp_path):
    client = use_client(monkeypatch, [0])
    cache = FileCache(tmp_path / "cache")
    await Sampler(model=MODEL, cache=cache).sample_tokens(PROMPT, sample_idx=0)
    await Sampler(model=MODEL, cache=cache).sample_tokens(PROMPT, sample_idx=1)
    await Sampler(model=MODEL, model_path="tinker://run/sampler_weights/step-1", cache=cache).sample_tokens(PROMPT)
    result = await Sampler(model=MODEL, cache=cache).sample_tokens([4, 5, 6], sample_idx=0)  # different prompt
    assert len(client.calls) == 4 and result.prompt_tokens == [4, 5, 6]


def test_unpriced_model_fails_before_any_call(monkeypatch, tracker):
    client = use_client(monkeypatch, [0])
    with pytest.raises(KeyError):
        Sampler(model="not/a-model", tracker=tracker)
    assert client.calls == []


def test_zero_concurrency_rejected():
    with pytest.raises(ValueError):
        Sampler(model=MODEL, max_concurrency=0)


@pytest.mark.real_renderer
def test_real_qwen3_8_renderer_contract(monkeypatch):
    """The installed cookbook's recommended Qwen3.8 renderer still behaves as the Sampler assumes."""
    monkeypatch.undo()
    from src.backends.tinker.clients import get_tokenizer_and_renderer, resolve_renderer_name

    name = resolve_renderer_name(MODEL, None)
    assert name == "qwen3_8_xhigh_reasoning"
    tokenizer, renderer = get_tokenizer_and_renderer(MODEL, name)
    assert [tokenizer.decode([t]) for t in renderer.get_stop_sequences()] == ["<|im_end|>"]
    clean, term = renderer.parse_response(tokenizer.encode("hmm</think>\n\n4<|im_end|>", add_special_tokens=False))
    assert term.is_clean and [p["type"] for p in clean["content"]] == ["thinking", "text"]
    _, term = renderer.parse_response(tokenizer.encode("still thinking", add_special_tokens=False))
    assert not term.is_clean


@pytest.mark.real_renderer
@pytest.mark.parametrize("system", [None, "You are terse."])
def test_normalized_messages_reproduce_prompt_tokens(monkeypatch, system):
    """Rendering normalized_messages must give the exact token IDs of the original prompt."""
    monkeypatch.undo()
    from src.backends.tinker.clients import get_tokenizer_and_renderer, normalized_messages

    _, renderer = get_tokenizer_and_renderer(MODEL, "qwen3_8_xhigh_reasoning")
    messages = ([{"role": "system", "content": system}] if system else []) + MESSAGES
    normalized = normalized_messages(renderer, messages)
    assert normalized[0]["role"] == "system" and normalized[0]["content"].startswith("Reasoning effort is set to xhigh")
    expected = renderer.build_generation_prompt(messages).to_ints()
    assert renderer.build_generation_prompt(normalized).to_ints() == expected


@pytest.mark.real_renderer
def test_continuation_tokens_extend_exact_history(monkeypatch):
    """A later turn's prompt keeps the model's exact tokens (re-rendering would normalize its whitespace) and
    otherwise matches what the chat template renders."""
    monkeypatch.undo()
    from src.backends.tinker.clients import continuation_tokens, get_tokenizer_and_renderer

    tokenizer, renderer = get_tokenizer_and_renderer(MODEL, "qwen3_8_xhigh_reasoning")
    first = renderer.build_generation_prompt(MESSAGES).to_ints()
    for text in ["\n\n  spaced thinking  \n\n</think>\n\nAnswer: 4<|im_end|>", "plain\n</think>\n\nAnswer: 4<|im_end|>"]:
        completion = tokenizer.encode(text, add_special_tokens=False)
        suffix = continuation_tokens(tokenizer, renderer, "Turn 2:\ncheck")
        assert tokenizer.decode(suffix) == "\n<|im_start|>user\nTurn 2:\ncheck<|im_end|>\n<|im_start|>assistant\n<think>\n"
        assert tokenizer.decode(first + completion + suffix).endswith(text + tokenizer.decode(suffix))


def test_the_cost_row_names_the_commit_at_launch_not_at_exit(monkeypatch, tmp_path):
    monkeypatch.setattr(costs, "git_commit", lambda: "launch-commit")
    t = CostTracker(tmp_path / "costs.jsonl")
    atexit.unregister(t._save_on_exit)
    t.add_tinker_cost(MODEL, prefill_tokens=1, sample_tokens=1)
    monkeypatch.setattr(costs, "git_commit", lambda: "exit-commit")  # HEAD moved while the run was going
    t._save_on_exit()
    assert json.loads((tmp_path / "costs.jsonl").read_text())["git_commit"] == "launch-commit"


@pytest.mark.real_renderer
@pytest.mark.parametrize("model, continuation", [
    ("Qwen/Qwen3.8-27B", "\n<|im_start|>user\nTurn 2:\ncheck<|im_end|>\n<|im_start|>assistant\n<think>\n"),
    ("thinkingmachines/Inkling", "<|message_user|><|content_text|>Turn 2:\ncheck<|end_message|>"),
    ("zai-org/GLM-5.3:peft:262144", "Turn 2:\ncheck<|assistant|><think>"),  # the stop <|user|> is the next header
    ("moonshotai/Kimi-K2.6",
     "<|im_user|>user<|im_middle|>Turn 2:\ncheck<|im_end|><|im_assistant|>assistant<|im_middle|><think>"),
    ("deepseek-ai/DeepSeek-V3.1", "<｜User｜>Turn 2:\ncheck<｜Assistant｜><think>"),
    ("nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
     "\n<|im_start|>user\nTurn 2:\ncheck<|im_end|>\n<|im_start|>assistant\n<think>\n"),
    ("Qwen/Qwen3.5-397B-A17B", "\n<|im_start|>user\nTurn 2:\ncheck<|im_end|>\n<|im_start|>assistant\n<think>\n"),
])
def test_continuations_keep_thinking_across_formats(monkeypatch, model, continuation):
    """Each supported format's later-turn prompt extends the first one, keeps the earlier thinking (renderers that
    strip it by default are switched), and continues after the first stop token."""
    monkeypatch.undo()
    from src.backends.tinker.clients import continuation_tokens, get_tokenizer_and_renderer, resolve_renderer_name

    tokenizer, renderer = get_tokenizer_and_renderer(model, resolve_renderer_name(model, None))
    assert tokenizer.decode(continuation_tokens(tokenizer, renderer, "Turn 2:\ncheck")) == continuation


@pytest.mark.real_renderer
def test_gpt_oss_is_rejected_for_multi_turn(monkeypatch):
    """Its sampled stop (<|return|>) never appears in rendered history (<|end|>), so appending would be wrong."""
    monkeypatch.undo()
    from src.backends.tinker.clients import continuation_tokens, get_tokenizer_and_renderer, resolve_renderer_name

    model = "openai/gpt-oss-120b"
    tokenizer, renderer = get_tokenizer_and_renderer(model, resolve_renderer_name(model, None))
    with pytest.raises(ValueError, match="stop token"):
        continuation_tokens(tokenizer, renderer, "Turn 2:\ncheck")


def test_the_continuation_preflight_renders_the_episodes_system_message():
    """The dummy conversation that derives later turns must have the episode's shape, system message included."""
    from src.backends.tinker.clients import continuation_tokens

    seen = []

    class Recording(FakeRenderer):
        def build_generation_prompt(self, messages):
            seen.append(messages)
            return super().build_generation_prompt(messages)

    renderer = Recording()
    with pytest.raises(ValueError):  # the fake renders every prompt alike, so the checks fail; only the calls matter
        continuation_tokens(FakeTokenizer(), renderer, "u2", system="RULE")
    assert seen and all(m[0] == {"role": "system", "content": "RULE"} for m in seen)
