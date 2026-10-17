"""Chen et al.'s CoT-Control-QA ("Reasoning Models Struggle to Control their Chains of Thought", arXiv 2603.05706;
github.com/YuehHanChen/CoTControl) on a Tinker base model or checkpoint: the nine controllability modes behind the
paper's headline score (compliance averaged over every (dataset, mode) pair); the unconstrained baseline is not run.

Everything task-side is the repo's own code, imported from a local clone: datasets, `create_prompt_for_sample`,
answer extraction and `run_evaluation` (which also writes the repo's results CSV, graded later by its
grade_compliance_csv.py). Only the model is ours: a `BaseLLM` that renders the messages with the same Tinker renderer
as training (Qwen3.8's injects its reasoning-effort system sentence) and samples through src/backends/tinker, with the
file cache: to resume an interrupted run, start it again with a fresh out_dir and the same cache_dir, and the samples
already drawn are reused at no cost. run_evaluation fixes temperature 0.0 (its default; the paper's main setting is
unstated) unless temperature=... overrides it here. Each sample is also saved to samples.jsonl
as it arrives, with its prompt, raw completion, token counts and stop reason.

    python scripts/experiments/cotcontrol_qa.py label=base out_dir=results/cotcontrol/<run> [n=20]
"""

from __future__ import annotations

import asyncio
import json
import signal
import sys
import time
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path

import chz
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments.cotcontrol_checks import DATASETS, MODES, check_clone
from src.backends.tinker.clients import close_service_client, get_tokenizer_and_renderer
from src.backends.tinker.sampling import Sampler
from src.utils.cache import FileCache
from src.utils.costs import CostTracker


@chz.chz
class Config:
    label: str  # e.g. base, repeat_seed1_step34
    out_dir: str
    model: str = "Qwen/Qwen3.8-27B"
    model_path: str | None = None  # tinker:// sampler weights; None = the untrained base model
    modes: str = ",".join(MODES)  # comma-separated
    datasets: str = ",".join(DATASETS)
    n: int | None = None  # first n questions per dataset (pilot); None = all
    temperature: float | None = None  # None = run_evaluation's own 0.0
    max_tokens: int = 25000  # the repo's DEFAULT_MAX_TOKENS / reasoning budget
    max_concurrency: int = 256
    # CoT-Control-QA inside a clone of github.com/YuehHanChen/CoTControl at cotcontrol_checks.PINNED_COMMIT
    cotcontrol_dir: str = str(ROOT / "external/CoTControl/CoT-Control-QA")
    cache_dir: str = str(ROOT / "file_cache")
    cost_file: str = str(ROOT / "costs/total_cost.jsonl")


def split_reasoning(sample) -> tuple[str, str]:
    """(reasoning, final response). A completion cut off by the token limit has no clean parse: everything before
    </think> is reasoning and the response is empty, as an API returns a truncated reasoning trace."""
    if sample.is_clean:
        return sample.thinking_text, sample.answer_text
    raw = sample.raw_text.replace("<|im_end|>", "")
    head, sep, tail = raw.partition("</think>")
    return head.replace("<think>", "").strip(), tail.strip() if sep else ""


_FRESH: ContextVar[list[bool] | None] = ContextVar("fresh", default=None)


class RecordingCache(FileCache):
    """A FileCache that tells the calling sample whether it was computed (paid for) or read back."""

    async def aget_or_compute_set(self, key, compute_fn):
        async def compute():
            if (flag := _FRESH.get()) is not None:
                flag[0] = True
            return await compute_fn()

        return await super().aget_or_compute_set(key, compute)


def save(path: Path, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def make_llm(config: Config, sampler: Sampler, samples_path: Path):
    from llm import BaseLLM, LLMResponse

    _, renderer = get_tokenizer_and_renderer(sampler.model, sampler.renderer_name)

    class TinkerLLM(BaseLLM):
        def generate(self, messages, max_tokens=4096, temperature=0.0, **kwargs):
            raise NotImplementedError("async only")

        async def generate_async(self, messages, max_tokens=4096, temperature=0.0, **kwargs):
            temp = config.temperature if config.temperature is not None else temperature
            prompt = renderer.build_generation_prompt(messages).to_ints()
            start, fresh = time.time(), [False]
            _FRESH.set(fresh)  # this task's own flag: the cache marks it if the sample is computed
            try:
                s = await sampler.sample_tokens(prompt, sample_idx=0, max_tokens=config.max_tokens, temperature=temp)
            except Exception as e:  # noqa: BLE001 - run_evaluation records it as an errored row
                return LLMResponse(content="", error=f"{type(e).__name__}: {e}")
            reasoning, content = split_reasoning(s)
            save(samples_path, {"messages": messages, "prompt_text": s.prompt_text, "raw_text": s.raw_text,
                            "reasoning": reasoning, "response": content, "is_clean": s.is_clean,
                            "stop_reason": s.stop_reason, "n_prompt_tokens": s.n_prompt_tokens,
                            "n_sample_tokens": s.n_sample_tokens, "temperature": temp,
                            "seconds": round(time.time() - start, 2), "from_cache": not fresh[0]})
            return LLMResponse(content=content, reasoning=reasoning, model=config.model_path or config.model,
                               usage={"prompt_tokens": s.n_prompt_tokens, "completion_tokens": s.n_sample_tokens},
                               latency_ms=1000 * (time.time() - start))

        async def generate_batch_async(self, batch_messages, max_tokens=4096, temperature=0.0, max_concurrency=50,
                                       batch_size=8, show_progress=True, **kwargs):
            # concurrency is the Sampler's semaphore, shared by every (mode, dataset) run in this process
            return await asyncio.gather(*(self.generate_async(m, max_tokens, temperature) for m in batch_messages))

    return TinkerLLM()


async def main(config: Config, out: Path, commit: str) -> None:
    import run_cceval

    tracker = CostTracker(Path(config.cost_file), run_description=f"cotcontrol_qa {config.label}")
    sampler = Sampler(config.model, config.model_path, cache=RecordingCache(config.cache_dir), tracker=tracker,
                      max_concurrency=config.max_concurrency)
    manifest = {"label": config.label, "config": chz.asdict(config),
                "cotcontrol_commit": commit, "renderer": sampler.renderer_name, "started_at": datetime.now(UTC).isoformat(),
                "status": "running"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    llm = make_llm(config, sampler, out / "samples.jsonl")
    if config.n is not None:  # pilot: the first n questions per dataset (the repo's own test mode is fixed at 3)
        original = run_cceval.load_evaluation_data
        run_cceval.load_evaluation_data = lambda f, test_mode=False, mode="word_suppression": original(f, False, mode)[: config.n]
    jobs = [run_cceval.run_evaluation(model_name=config.label, data_file=Path(config.cotcontrol_dir) / "datasets" / f"{d}.csv",
                                      mode=m, log_dir=str(out / "csv"), max_tokens=config.max_tokens, llm=llm)
            for d in config.datasets.split(",") for m in config.modes.split(",")]
    try:
        files = await asyncio.gather(*jobs)
        manifest.update(status="complete", csv_files=[str(f) for f in files])
    except BaseException as e:
        manifest.update(status="failed", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        manifest.update(finished_at=datetime.now(UTC).isoformat(), cost=tracker.summary())
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
        print(f"cost {tracker.summary()}", flush=True)


def entry(config: Config) -> None:
    load_dotenv(ROOT / ".env")
    sys.path.insert(0, config.cotcontrol_dir)
    out = Path(config.out_dir).resolve()
    if (out / "manifest.json").exists():
        raise SystemExit(f"{out} already holds an attempt; use a fresh out_dir (cached samples are reused at no cost)")
    out.mkdir(parents=True, exist_ok=True)
    commit = check_clone(config.cotcontrol_dir)

    def stop(signum, frame):  # SIGTERM (e.g. the runner stopping the scope): unwind so manifest and cost are saved
        raise KeyboardInterrupt(signal.Signals(signum).name)

    signal.signal(signal.SIGTERM, stop)
    status = "errored"
    try:
        asyncio.run(main(config, out, commit))
        status = "success"
    finally:
        close_service_client(status)


if __name__ == "__main__":
    chz.nested_entrypoint(entry)
