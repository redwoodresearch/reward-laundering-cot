"""The rollouts command end to end against fakes, plus its two refuse-before-spending checks."""

import asyncio
import json

import chz
import pytest

from main import RolloutsConfig
from src.backends.tinker import sampling
from src.dataset.pools import load_pool
from src.rollouts import episodes
from src.rollouts import run as rollouts
from src.tasks.maths import MathQA
from src.utils import costs
from tests.fakes import STOP, FakeRenderer, FakeTokenizer, sequence


class SeededClient:
    """seed 0 -> clean, seed 1 -> hits max_tokens (unclean), seed 2 -> raises."""

    def __init__(self):
        self.calls = 0

    async def sample_async(self, prompt, num_samples, sampling_params):
        self.calls += 1
        if sampling_params.seed == 2:
            raise RuntimeError("boom")
        tokens, stop = ([5, 6, *STOP], "stop") if sampling_params.seed == 0 else ([5, 6, 7], "length")
        return sequence(tokens, stop)


@pytest.fixture
def client(monkeypatch):
    client = SeededClient()
    fake = lambda model, name: (FakeTokenizer(), FakeRenderer())
    monkeypatch.setattr(sampling, "get_tokenizer_and_renderer", fake)
    monkeypatch.setattr(rollouts, "get_tokenizer_and_renderer", fake)
    monkeypatch.setattr(sampling, "resolve_renderer_name", lambda model, name: "fake")
    monkeypatch.setattr(sampling, "get_sampling_client", lambda model, model_path=None: client)
    monkeypatch.setattr(costs.atexit, "register", lambda fn: None)
    return client


def config(tmp_path):
    return RolloutsConfig(pool="math_gsm8k", split="val", n_problems=2, samples_per_problem=3, use_cache=False, run_name="t",
                          cost_file=str(tmp_path / "costs.jsonl"), output_dir=str(tmp_path / "runs"))


async def test_run_records_every_rollout_and_isolates_failures(tmp_path, client):
    with pytest.raises(SystemExit):  # one sample per problem fails -> nonzero exit, nothing else lost
        await rollouts.run(config(tmp_path))
    run_dir = tmp_path / "runs" / "t"
    rows = {r["rollout_id"]: r for r in map(json.loads, (run_dir / "rollouts.jsonl").read_text().splitlines())}
    manifest = json.loads((run_dir / "manifest.json").read_text())

    ids = [p.id for p in load_pool("math_gsm8k", "val")[:2]]
    assert manifest["pool"]["problem_ids"] == ids and len(manifest["pool"]["sha256"]) == 64
    assert set(rows) == {f"{i}-s{s}" for i in ids for s in (0, 1)}
    assert {e["rollout_id"] for e in manifest["errors"]} == {f"{i}-s2" for i in ids}
    assert manifest["status"] == "complete_with_errors"
    assert manifest["counts"] == {"expected": 6, "rollouts": 4, "complete": 2, "incomplete": 2, "errors": 2}

    (unclean,) = rows[f"{ids[0]}-s1"]["turns"]
    assert rows[f"{ids[0]}-s1"]["status"] == "incomplete"
    assert (unclean["parse_status"], unclean["stop_reason"], unclean["answer_text"]) == ("unclean", "length", "")
    assert unclean["raw_text"] == "5 6 7" and unclean["completion_tokens"] == [5, 6, 7]
    assert len(unclean["completion_logprobs"]) == 3  # one logprob per sampled token, needed for training
    clean = rows[f"{ids[0]}-s0"]
    assert clean["status"] == "complete" and clean["turns"][0]["prompt_tokens"] == [1, 2, 3]
    assert clean["normalized_messages"][0] == {"role": "system", "content": "injected"}
    assert clean["task"] == manifest["task"] == {"name": "math_qa"}
    assert clean["turns"][0]["user"] == MathQA().user_turns(clean["problem"])[0]


class TurnClient:
    """Counts continuation markers in the prompt to know the turn: seed 1 fails on turn 2, and seed 2's turn 2
    parses cleanly but reports the length limit."""

    async def sample_async(self, prompt, num_samples, sampling_params):
        tokens = prompt.to_ints()
        if sampling_params.seed == 1 and tokens.count(9) == 1:
            raise RuntimeError("api down")
        stop = "length" if sampling_params.seed == 2 and tokens.count(9) == 1 else "stop"
        return sequence([7, *STOP], stop)


async def test_two_turn_episodes_extend_exact_tokens_and_keep_partial_turns(tmp_path, client, monkeypatch):
    turn_client = TurnClient()
    monkeypatch.setattr(sampling, "get_sampling_client", lambda model, model_path=None: turn_client)
    monkeypatch.setattr(episodes, "continuation_tokens", lambda tokenizer, renderer, user, system=None: [9])
    with pytest.raises(SystemExit):
        await rollouts.run(chz.replace(config(tmp_path), pool="gsm8k_gate", n_problems=1))
    run_dir = tmp_path / "runs" / "t"
    rows = {r["sample_idx"]: r for r in map(json.loads, (run_dir / "rollouts.jsonl").read_text().splitlines())}
    full, partial = rows[0], rows[1]
    assert full["status"] == "complete" and len(full["turns"]) == 2
    # each turn's prompt is the previous prompt + its exact completion + the next user turn's tokens
    assert [t["prompt_tokens"] for t in full["turns"]] == [[1, 2, 3], [1, 2, 3, 7, 0, 9]]
    assert partial["status"] == "error" and len(partial["turns"]) == 1 and partial["error"]["turn"] == 1
    assert rows[2]["status"] == "incomplete"  # a clean parse that hit the length limit did not finish the episode
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["counts"] == {"expected": 3, "rollouts": 3, "complete": 1, "incomplete": 1, "errors": 1}
    assert manifest["pool"]["sources"]["source"]["file"].endswith("data/sources/math_gsm8k.jsonl")


async def test_existing_run_dir_is_refused(tmp_path, client):
    (tmp_path / "runs" / "t").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        await rollouts.run(config(tmp_path))
    assert client.calls == 0


async def test_a_renderer_without_normalization_records_none(tmp_path, client, monkeypatch):
    class NoNormalize(FakeRenderer):
        _normalize_messages = None

    monkeypatch.setattr(rollouts, "get_tokenizer_and_renderer", lambda model, name: (FakeTokenizer(), NoNormalize()))
    with pytest.raises(SystemExit):  # the fake client fails one sample by design
        await rollouts.run(config(tmp_path))
    rows = [json.loads(line) for line in (tmp_path / "runs" / "t" / "rollouts.jsonl").read_text().splitlines()]
    assert rows and all(r["normalized_messages"] is None for r in rows)


async def test_an_interrupted_run_records_what_it_stored_and_why(tmp_path, client, monkeypatch):
    class OneThenHang:
        async def sample_async(self, prompt, num_samples, sampling_params):
            if sampling_params.seed:  # every sample but the first never returns
                await asyncio.sleep(3600)
            return sequence([5, 6, *STOP], "stop")

    monkeypatch.setattr(sampling, "get_sampling_client", lambda model, model_path=None: OneThenHang())
    run = asyncio.ensure_future(rollouts.run(config(tmp_path)))
    path = tmp_path / "runs" / "t" / "rollouts.jsonl"
    async with asyncio.timeout(10):  # a regression before the first write fails instead of hanging
        while not (path.exists() and path.read_text()):
            await asyncio.sleep(0.01)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run
    manifest = json.loads((tmp_path / "runs" / "t" / "manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["error"]["type"] == "CancelledError"
    stored = len(path.read_text().splitlines())  # each problem's first sample returns; the rest hang
    assert manifest["counts"]["rollouts"] == stored >= 1 and "git_dirty" in manifest
