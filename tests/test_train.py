import asyncio
import hashlib
import itertools
import json
from types import SimpleNamespace

import pytest
import torch

from main import TrainConfig
from src.backends.tinker import sampling
from src.dataset import pools
from src.dataset.pools import load_pool, pool_path
from src.rollouts import episodes
from src.tasks.constraints import TARGET
from src.tasks.repeat_sentences import RepeatSentencesDirectCorrect
from src.training import grpo as train
from src.utils import costs
from src.utils.stats import eval_precision, samples_for_margin
from tests.fakes import STOP, FakeRenderer, FakeTokenizer, sequence

COMPLIANT = 7


class CompliantRenderer(FakeRenderer):
    """A completion starting with COMPLIANT thinks only the target sentence; any other thinks without it."""

    def parse_response(self, tokens):
        thinking = TARGET if tokens[0] == COMPLIANT else "reasoning"
        content = [{"type": "thinking", "thinking": thinking}, {"type": "text", "text": "ANSWER: A"}]
        return {"role": "assistant", "content": content}, super().parse_response(tokens)[1]


class SeededClient:
    """Odd seeds comply, even seeds do not, so every group of two has one reward of each; seed 0 fails."""

    async def sample_async(self, prompt, num_samples, sampling_params):
        if sampling_params.seed == 0:
            raise RuntimeError("boom")
        return sequence([COMPLIANT if sampling_params.seed % 2 else 5, 5, *STOP], "stop")


class Future:
    def __init__(self, value):
        self.value = value

    async def result_async(self):
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


class FakeTrainingClient:
    """Learner logprob of datum j at position k is -(0.5 + k + 10j), so misaligned or swapped values show."""

    def __init__(self):
        self.batches, self.lrs, self.loss_fns, self.optimizer_error = [], [], [], None

    async def forward_backward_async(self, data, loss_fn):
        self.loss_fns.append(loss_fn)
        self.batches.append(data)
        outputs = [{"logprobs": SimpleNamespace(to_torch=lambda j=j, d=d: -(0.5 + torch.arange(d.model_input.length)
                                                                            + 10 * j))} for j, d in enumerate(data)]
        return Future(SimpleNamespace(metrics={"loss:sum": 1.0}, loss_fn_outputs=outputs))

    async def optim_step_async(self, adam):
        self.lrs.append(adam.learning_rate)
        return Future(self.optimizer_error or SimpleNamespace(metrics={"grad_norm": 0.1}))

    async def save_weights_for_sampler_async(self, name):
        return Future(SimpleNamespace(path=f"tinker://fake/{name}"))

    async def save_state_async(self, name):
        return Future(SimpleNamespace(path=f"tinker://fake/state/{name}"))


class ComplianceOnly(RepeatSentencesDirectCorrect):
    """The direct task rewarding compliance alone, so a fake completion's reward depends only on its first token
    (CompliantRenderer always answers "A", which is rarely the correct letter)."""

    name = "test_compliance_only"

    def reward(self, grade: dict) -> float:
        return float(grade["cot_compliant"] is True)


@pytest.fixture
def compliance_pool(monkeypatch):
    """A test-only pool `compliance`: the direct task's committed questions, read in place, under ComplianceOnly."""
    real = pools.pool_path

    def path(name, split):
        return real("repeat_direct_correct" if name == "compliance" else name, split)

    monkeypatch.setitem(pools.POOLS, "compliance", pools.PoolSpec(ComplianceOnly(), pools.SideQuestionsOnly(20261011)))
    monkeypatch.setattr(pools, "pool_path", path)
    monkeypatch.setattr(train, "pool_path", path)


def _config(tmp_path, **overrides) -> TrainConfig:
    """A tiny local run on the direct task: 2 steps of 2 problems x 2 samples, a 2-problem eval, no W&B or Docent."""
    return TrainConfig(**{"pool": "compliance", "eval_split": "eval", "n_steps": 2, "batch_size": 2, "group_size": 2,
                          "eval_n_problems": 2, "eval_margin": 0.5, "wandb_project": None, "docent": False, "run_name": "t",
                          "output_dir": str(tmp_path / "runs"), "cost_file": str(tmp_path / "costs.jsonl"),
                          "checkpoint_index": str(tmp_path / "checkpoints.jsonl"), **overrides})


@pytest.fixture
def trainer_client(monkeypatch, compliance_pool):
    def fake(model, name):
        assert name == "fake", "the renderer name must be resolved before use"
        return FakeTokenizer(), CompliantRenderer()

    training_client = FakeTrainingClient()
    logged = []

    async def create(**kwargs):
        training_client.create_kwargs = kwargs
        return training_client

    monkeypatch.setattr(sampling, "get_tokenizer_and_renderer", fake)
    monkeypatch.setattr(train, "get_tokenizer_and_renderer", fake)
    for module in (sampling, train):
        monkeypatch.setattr(module, "resolve_renderer_name", lambda model, name: name or "fake")
    monkeypatch.setattr(sampling, "get_sampling_client", lambda model, model_path=None: SeededClient())
    monkeypatch.setattr(train, "get_service_client", lambda: SimpleNamespace(create_lora_training_client_async=create))
    monkeypatch.setattr(train.ml_log, "setup_logging", lambda *a, **k: SimpleNamespace(
        log_metrics=lambda metrics, step: logged.append(metrics), close=lambda: None))
    monkeypatch.setattr(costs.atexit, "register", lambda fn: None)
    training_client.logged = logged
    return training_client


async def test_grpo_steps_seal_episodes_drop_failed_groups_and_update_the_sampler(tmp_path, trainer_client):
    config = _config(tmp_path, eval_every=1)
    run_dir = await train.run(config)

    step0 = [json.loads(line) for line in (run_dir / "train/step_0000.jsonl").read_text().splitlines()]
    # Seed 0 (step 0, group 0, sample 0) failed, so group 0 is dropped whole; group 1 has one compliant sample.
    assert [(r["group"], r["group_sample"], r["group_status"], r["reward"], r["advantage"]) for r in step0] == [
        (0, 1, "failed", 1.0, 0.0), (1, 0, "used", 0.0, -0.5), (1, 1, "used", 1.0, 0.5)]
    [error] = [json.loads(line) for line in (run_dir / "errors.jsonl").read_text().splitlines()]
    assert {k: error[k] for k in ("step_file", "group", "sample_idx", "seed", "type", "message")} == {
        "step_file": "train/step_0000", "group": 0, "sample_idx": 0, "seed": 0, "type": "RuntimeError", "message": "boom"}
    assert all(r["model_path"] is None and r["policy_step"] == 0 for r in step0)
    step1 = [json.loads(line) for line in (run_dir / "train/step_0001.jsonl").read_text().splitlines()]
    assert {r["model_path"] for r in step1} == {"tinker://fake/t_step0001"}
    assert [r["advantage"] for r in step1] == [-0.5, 0.5, -0.5, 0.5]

    # One datum per used episode, mask stripped, advantages placed on completion tokens only.
    assert [len(batch) for batch in trainer_client.batches] == [2, 4]
    assert trainer_client.loss_fns == ["importance_sampling"] * 2
    # Datum = prompt [1, 2, 3] + completion [7, 5, STOP], shifted by one: only the 3 completion targets carry the
    # advantage and the sampler's logprobs.
    datum = trainer_client.batches[0][1]
    assert "mask" not in datum.loss_fn_inputs
    assert datum.loss_fn_inputs["target_tokens"].to_torch().tolist() == [2, 3, COMPLIANT, 5, *STOP]
    assert datum.loss_fn_inputs["advantages"].to_torch().tolist() == [0, 0, 0.5, 0.5, 0.5]
    assert datum.loss_fn_inputs["logprobs"].to_torch().tolist() == [0, 0, -0.5, -0.5, -0.5]
    # Learner logprobs: only the used group (1, not the failed group 0), aligned with each completion's 3 tokens.
    optim = [json.loads(line) for line in (run_dir / "optim/step_0000.jsonl").read_text().splitlines()]
    assert [(o["group"], o["group_sample"], o["learner_logprobs"]) for o in optim] == [
        (1, 0, [-2.5, -3.5, -4.5]), (1, 1, [-12.5, -13.5, -14.5])]  # the 3 completion positions of datums 0 and 1
    assert trainer_client.lrs == [pytest.approx(config.learning_rate), pytest.approx(config.learning_rate / 2)]  # linear

    assert sorted(p.name for p in (run_dir / "eval").iterdir()) == ["step_0000.jsonl", "step_0001.jsonl", "step_0002.jsonl"]
    final_eval = [json.loads(line) for line in (run_dir / "eval/step_0002.jsonl").read_text().splitlines()]
    assert {r["model_path"] for r in final_eval} == {"tinker://fake/t_step0002"}
    assert len(final_eval) == 4 and {r["sample_idx"] for r in final_eval} == {0, 1}  # margin 0.5 -> 2 samples each
    index = [json.loads(line) for line in (tmp_path / "checkpoints.jsonl").read_text().splitlines()]
    # Full state is saved at each nonzero eval step, before that step's update (so it is the evaluated policy).
    assert [(c["step"], c["kind"]) for c in index] == [(1, "sampler"), (1, "state"), (2, "sampler"), (2, "state")]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["final_state"] == "tinker://fake/state/t_step0002"
    first, second = trainer_client.logged[0], trainer_client.logged[1]
    assert sorted(first) == sorted([  # the whole chart set of the direct task's headline metrics
        "eval/reward", "eval/compliance", "eval/side_accuracy", "eval/thinking_skipped", "eval/completion_tokens",
        "train/reward", "train/completion_tokens", "train/groups_with_signal", "cost/usd"])
    assert (first["train/groups_with_signal"], second["train/groups_with_signal"]) == (0.5, 1.0)
    assert first["train/reward"] == 2 / 3 and "train/groups_with_signal" not in trainer_client.logged[2]
    summaries = [json.loads(line) for line in (run_dir / "summaries.jsonl").read_text().splitlines()]
    assert summaries[0]["train"]["groups"] == {"used": 1, "constant": 0, "failed": 1}
    assert (summaries[0]["train"]["episodes_expected"], summaries[0]["train"]["failed_episodes"]) == (4, 1)
    assert summaries[0]["train"]["optim"]["grad_norm"] == 0.1 and summaries[0]["train"]["outcomes"]["compliant"] == 2
    assert "ci95" in summaries[2]["eval"]["reward_rate"] and "train" not in summaries[2]


async def test_grpo_shuffled_credits_each_groups_rewards_to_permuted_episodes(tmp_path, trainer_client, monkeypatch):
    """The same groups update with the same advantage values as GRPO, but an episode's advantage follows the reward
    permuted onto it, keyed by (shuffle_seed, step, group); its own reward is still recorded. The permutation is
    replaced by a known one (reversal) so the test checks the wiring; test_golden pins the real permutations."""
    calls = []

    def reverse(rewards, seed, step, group):
        calls.append((seed, step, group))
        return list(reversed(rewards))

    monkeypatch.setattr(train, "_shuffled", reverse)
    run_dir = await train.run(_config(tmp_path, objective="grpo_shuffled", shuffle_seed=3, n_steps=4,
                                      eval_every=4))
    rows = [json.loads(line) for s in range(4) for line in (run_dir / f"train/step_{s:04d}.jsonl").read_text().splitlines()]
    used = [r for r in rows if r["group_status"] == "used"]
    keys = []
    for (step, group), grouped in itertools.groupby(used, lambda r: (r["policy_step"], r["group"])):
        members = list(grouped)
        keys.append((3, step, group))
        credited = [r["reward"] for r in reversed(members)]
        assert [r["credited_reward"] for r in members] == credited
        assert [r["advantage"] for r in members] == [c - sum(credited) / len(credited) for c in credited]
    assert set(keys) <= set(calls) and keys  # every used group was permuted under its own key
    assert any(r["credited_reward"] != r["reward"] for r in used)  # the permutation actually moved credit
    datum = trainer_client.batches[0][-1]
    assert trainer_client.loss_fns[0] == "importance_sampling"
    assert datum.loss_fn_inputs["advantages"].to_torch().tolist()[-1] == used[1]["advantage"]
    assert "train/groups_with_signal" in trainer_client.logged[0]


def test_a_reward_shuffle_can_produce_every_ordering_and_is_seeded():
    orders = {tuple(train._shuffled([0, 1, 2], seed, 0, 0)) for seed in range(200)}
    assert len(orders) == 6  # every ordering, the identity included
    assert train._shuffled([0, 1, 2, 3], 7, 2, 5) == train._shuffled([0, 1, 2, 3], 7, 2, 5)


def test_batches_cover_each_epoch_once_in_a_seeded_order():
    problems = list(range(5))
    first = [p for step in range(5) for p in train._batch_problems(problems, step, 2, seed=0)]
    assert sorted(first[:5]) == problems and sorted(first[5:]) == problems
    assert first == [p for step in range(5) for p in train._batch_problems(problems, step, 2, seed=0)]


async def test_a_fresh_lora_is_initialized_from_lora_seed(tmp_path, trainer_client):
    await train.run(_config(tmp_path, n_steps=1, eval_n_problems=0, lora_seed=7))
    assert trainer_client.create_kwargs["seed"] == 7


@pytest.mark.parametrize("objective", ["sft_on_policy", "sft_base_rollouts"])
@pytest.mark.parametrize(("reduction", "weight"), [("sum", 1.0), ("mean", 1 / 3)])
async def test_sft_objectives_imitate_every_episode_of_complete_groups(tmp_path, trainer_client, objective, reduction,
                                                                       weight):
    """No reward signal: each complete group's episodes are trained with cross-entropy on their sampled tokens, each
    weighted 1 (sum) or 1/length (mean); sft_base_rollouts samples training episodes from the base model, evals the
    policy."""
    run_dir = await train.run(_config(tmp_path, objective=objective, sft_reduction=reduction, eval_every=1,
                                      lr_schedule_steps=4))
    step0 = [json.loads(line) for line in (run_dir / "train/step_0000.jsonl").read_text().splitlines()]
    assert [(r["group"], r["group_status"], r["advantage"]) for r in step0] == [
        (0, "failed", None), (1, "used", None), (1, "used", None)]
    assert trainer_client.loss_fns == ["cross_entropy"] * 2 and [len(b) for b in trainer_client.batches] == [2, 4]
    datum = trainer_client.batches[0][1]
    assert set(datum.loss_fn_inputs) == {"target_tokens", "weights"}
    assert datum.loss_fn_inputs["weights"].to_torch().tolist() == pytest.approx([0, 0, weight, weight, weight])
    assert trainer_client.lrs == [pytest.approx(1e-4), pytest.approx(0.75e-4)]  # a 4-step linear horizon
    step1 = [json.loads(line) for line in (run_dir / "train/step_0001.jsonl").read_text().splitlines()]
    trained = {"sft_on_policy": "tinker://fake/t_step0001", "sft_base_rollouts": None}[objective]
    assert {r["model_path"] for r in step1} == {trained}
    eval1 = [json.loads(line) for line in (run_dir / "eval/step_0001.jsonl").read_text().splitlines()]
    assert {r["model_path"] for r in eval1} == {"tinker://fake/t_step0001"}  # evals always sample the policy
    assert run_dir.name == "t" and "sft" in json.loads((tmp_path / "checkpoints.jsonl").read_text().splitlines()[0])[
        "description"]


def test_a_schedule_horizon_shorter_than_the_run_is_refused(tmp_path, compliance_pool):
    with pytest.raises(ValueError, match="lr_schedule_steps"):
        train.Trainer(_config(tmp_path, n_steps=4, lr_schedule_steps=2), tmp_path / "r")


async def test_disabled_eval_neither_reads_nor_records_its_split(tmp_path, trainer_client):
    """eval_n_problems=0 must work for an eval split the pool does not have."""
    await train.run(_config(tmp_path, n_steps=1, eval_n_problems=0, eval_split="no_such_split"))
    manifest = json.loads((tmp_path / "runs" / "t" / "manifest.json").read_text())
    assert "eval" not in manifest["pool"] and manifest["pool"]["eval_problem_ids"] == []


def test_eval_precision_comes_from_the_design_and_flags_missing_samples():
    assert samples_for_margin(100, 0.02) == 25  # 2,500 samples: +/-1.96 points even at a 50% rate
    # Identical observations still carry the design's uncertainty; the spread diagnostic shows how much questions differ.
    full = eval_precision([[1.0] * 25] * 50 + [[0.0] * 25] * 50, planned_per_item=25, margin=0.02)
    assert full["half_width"] == pytest.approx(0.0196) and full["target_met"] and full["question_spread"] > 0.09
    short = eval_precision([[1.0] * 25] * 99 + [[]], planned_per_item=25, margin=0.02)  # one question got nothing
    assert short["items_short"] == 1 and short["half_width"] is None and not short["target_met"]


async def test_a_step_where_every_group_failed_stops_the_run_after_saving_it(tmp_path, trainer_client, monkeypatch):
    class Down:
        async def sample_async(self, prompt, num_samples, sampling_params):
            raise RuntimeError("api down")

    monkeypatch.setattr(sampling, "get_sampling_client", lambda model, model_path=None: Down())
    config = _config(tmp_path, pool="repeat_direct_correct", eval_n_problems=0)
    with pytest.raises(RuntimeError, match="every training group"):
        await train.run(config)
    run_dir = tmp_path / "runs" / "t"
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["steps_done"] == 1
    assert "every training group" in manifest["error"]["traceback"]
    assert len((run_dir / "errors.jsonl").read_text().splitlines()) == 4
    [summary] = [json.loads(line) for line in (run_dir / "summaries.jsonl").read_text().splitlines()]
    assert summary["train"]["episodes_received"] == 0  # and no reward is invented for a step with no episodes:
    assert set(summary["chart"]) == {"train/groups_with_signal", "cost/usd"}


async def test_a_failing_update_cancels_the_running_eval_and_keeps_its_cause(tmp_path, trainer_client, monkeypatch):
    canceled = []

    class EvalHangs(SeededClient):
        async def sample_async(self, prompt, num_samples, sampling_params):
            if sampling_params.seed >= 10**9:  # eval seeds: never finish unless canceled
                try:
                    await asyncio.sleep(3600)
                except asyncio.CancelledError:
                    canceled.append(sampling_params.seed)
                    raise
            return await super().sample_async(prompt, num_samples, sampling_params)

    monkeypatch.setattr(sampling, "get_sampling_client", lambda model, model_path=None: EvalHangs())
    trainer_client.optimizer_error = RuntimeError("optimizer sentinel")
    config = _config(tmp_path)
    with pytest.raises(ExceptionGroup):
        await asyncio.wait_for(train.run(config), timeout=10)
    assert len(canceled) == 4  # every eval call (2 questions x 2 samples) was canceled before run returned
    run_dir = tmp_path / "runs" / "t"
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "failed" and "optimizer sentinel" in manifest["error"]["traceback"]
    assert not (run_dir / "eval/step_0000.jsonl").exists() and (run_dir / "optim/step_0000.jsonl").exists()


async def test_eval_split_val_is_sampled_and_recorded(tmp_path, trainer_client, monkeypatch):
    monkeypatch.setattr(episodes, "continuation_tokens", lambda tokenizer, renderer, user, system=None: [9])  # two-turn task
    config = _config(tmp_path, pool="gsm8k_gate", eval_split="val", n_steps=1)
    run_dir = await train.run(config)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    val_file = pool_path("gsm8k_gate", "val")
    assert (manifest["pool"]["eval"]["split"], manifest["pool"]["eval"]["file"]) == ("val", str(val_file))
    assert manifest["pool"]["eval"]["sha256"] == hashlib.sha256(val_file.read_bytes()).hexdigest()
    index = [json.loads(line) for line in (tmp_path / "checkpoints.jsonl").read_text().splitlines()]
    assert index and all(c["datasets"]["eval"] == str(val_file) for c in index)
    val_ids = {p.id for p in load_pool("gsm8k_gate", "val")}
    rows = [json.loads(line) for line in (run_dir / "eval/step_0000.jsonl").read_text().splitlines()]
    assert rows and {r["problem"]["id"] for r in rows} <= val_ids and all(len(r["turns"]) == 2 for r in rows)


def test_a_task_without_training_metrics_is_refused_before_any_setup(tmp_path):
    with pytest.raises(ValueError, match="no training metrics"):
        train.Trainer(TrainConfig(pool="math_gsm8k", eval_split="val", cost_file=str(tmp_path / "costs.jsonl")),
                      tmp_path / "run")


class _TruncatedClient(SeededClient):
    async def sample_async(self, prompt, num_samples, sampling_params):
        if sampling_params.seed == 3:
            return sequence([7, 5], "length")
        return await super().sample_async(prompt, num_samples, sampling_params)


@pytest.mark.parametrize("objective", ["sft_on_policy", "sft_base_rollouts"])
async def test_sft_trains_constant_reward_groups_on_every_sampled_turn(tmp_path, trainer_client, monkeypatch, objective):
    """Unlike GRPO, SFT trains a group whose rewards are all equal (here 0); every sampled token of every turn (a
    truncated turn 1 included) is weighted (here with mean: the episode's weights sum to 1), and prompts are never
    trained."""
    monkeypatch.setattr(episodes, "continuation_tokens", lambda *args: [20, 21])  # a two-turn task
    monkeypatch.setattr(sampling, "get_sampling_client", lambda model, model_path=None: _TruncatedClient())
    run = await train.run(_config(tmp_path, pool="gsm8k_gate", eval_split="val", objective=objective, n_steps=1,
                                  eval_n_problems=0, lr_schedule_steps=68, sft_reduction="mean"))
    rows = [json.loads(line) for line in (run / "train/step_0000.jsonl").read_text().splitlines()]
    used = [r for r in rows if r["group_status"] == "used"]
    assert {r["reward"] for r in used} == {0.0} and [r["status"] for r in used] == ["complete", "incomplete"]
    assert [len(r["turns"]) for r in used] == [2, 1] and trainer_client.loss_fns == ["cross_entropy"]
    for row, datum in zip(used, trainer_client.batches[0], strict=True):
        n = sum(len(t["completion_tokens"]) for t in row["turns"])
        expected = [0.0] * datum.model_input.length
        for turn in row["turns"]:
            start = len(turn["prompt_tokens"]) - 1
            expected[start:start + len(turn["completion_tokens"])] = [1 / n] * len(turn["completion_tokens"])
        assert datum.loss_fn_inputs["weights"].to_torch().tolist() == pytest.approx(expected)
        tokens = row["turns"][-1]["prompt_tokens"] + row["turns"][-1]["completion_tokens"]
        assert datum.loss_fn_inputs["target_tokens"].to_torch().tolist() == tokens[1:]
    summary = json.loads((run / "summaries.jsonl").read_text().splitlines()[0])
    assert summary["train"]["optim"]["datums"] == 2 and summary["train"]["optim"]["train_tokens"] > 0
    assert summary["chart"]["train/groups_trained"] == 0.5 and "train/groups_with_signal" not in summary["chart"]


async def test_a_curriculum_trains_and_evaluates_every_component(tmp_path, trainer_client, monkeypatch):
    monkeypatch.setattr(episodes, "continuation_tokens", lambda tokenizer, renderer, user, system=None: [9])  # two-turn tasks
    monkeypatch.setitem(train.CURRICULA, "two", (("rs", "gsm8k_gate", ""), ("lc", "gsm8k_lowercase_thinking_rl_gate", "")))
    run_dir = await train.run(_config(tmp_path, pool="unused", curriculum="two", eval_split="val", n_steps=1, batch_size=4))
    rows = [json.loads(line) for line in (run_dir / "train/step_0000.jsonl").read_text().splitlines()]
    by = {k: [r for r in rows if r["component"] == k] for k in ("rs", "lc")}
    assert all(len({r["group"] for r in v}) == 2 for v in by.values())  # 2 questions each (the fake fails seed 0)
    assert {r["task"]["name"] for r in by["rs"]} != {r["task"]["name"] for r in by["lc"]}
    assert {r["problem"]["id"] for r in by["lc"]} <= {p.id for p in load_pool("gsm8k_lowercase_thinking_rl_gate", "train")}
    evals = [json.loads(line) for line in (run_dir / "eval/step_0000.jsonl").read_text().splitlines()]
    assert {r["component"] for r in evals} == {"rs", "lc"} and len(evals) == 2 * 2 * samples_for_margin(2, 0.5)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["curriculum"] == "two" and set(manifest["components"]) == {"rs", "lc"}
    summary = json.loads((run_dir / "summaries.jsonl").read_text().splitlines()[0])
    assert set(summary["eval"]["components"]) == set(summary["train"]["components"]) == {"rs", "lc"}
    assert {"eval/rs/compliance", "eval/lc/compliance", "train/rs/reward"} <= set(trainer_client.logged[0])
    index = [json.loads(line) for line in (tmp_path / "checkpoints.jsonl").read_text().splitlines()]
    assert index and index[0]["pool"].startswith("curriculum:two") and "lc/eval" in index[0]["datasets"]


def test_a_curriculum_needs_an_even_batch_split_and_no_docent(tmp_path, monkeypatch):
    monkeypatch.setitem(train.CURRICULA, "two", (("rs", "gsm8k_gate", ""), ("lc", "gsm8k_lowercase_thinking_rl_gate", "")))
    with pytest.raises(ValueError, match="split evenly"):
        train.Trainer(_config(tmp_path, curriculum="two", eval_split="val", batch_size=3), tmp_path / "r")
    with pytest.raises(ValueError, match="Docent"):
        train.Trainer(_config(tmp_path, curriculum="two", eval_split="val", docent=True), tmp_path / "r")


async def test_a_curriculum_centres_rewards_within_each_rules_groups_and_trains_on_them(tmp_path, trainer_client, monkeypatch):
    # two compliance-rewarded components: odd seeds comply, so every complete group of two has rewards 0 and 1
    monkeypatch.setitem(train.CURRICULA, "pair", (("a", "compliance", ""), ("b", "compliance", "")))
    run_dir = await train.run(_config(tmp_path, pool="unused", curriculum="pair", n_steps=1, batch_size=4, eval_n_problems=0))
    rows = [json.loads(line) for line in (run_dir / "train/step_0000.jsonl").read_text().splitlines()]
    used = [r for r in rows if r["group_status"] == "used"]
    assert {r["component"] for r in used} == {"a", "b"}  # seed 0 fails only component a's first group
    for g in {r["group"] for r in used}:
        members = [r for r in used if r["group"] == g]
        assert len({r["component"] for r in members}) == 1  # a group never mixes rules
        assert sorted((r["reward"], r["advantage"]) for r in members) == [(0.0, -0.5), (1.0, 0.5)]
    [batch] = trainer_client.batches
    submitted = sorted(max(d.loss_fn_inputs["advantages"].to_torch().tolist(), key=abs) for d in batch)
    assert len(batch) == len(used) and submitted == sorted(r["advantage"] for r in used)
