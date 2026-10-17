import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import chz
from chz.blueprint._entrypoint import EntrypointException, EntrypointHelpException
from chz.validators import ge, gt, typecheck
from dotenv import load_dotenv

# Defaults are anchored at the repo root so outputs land in the same place from any cwd.
ROOT = Path(__file__).resolve().parent


@chz.chz
class MakePoolsConfig:
    pool: str  # a key of src.dataset.pools.POOLS


@chz.chz
class RolloutsConfig:
    pool: str = "gsm8k_gate"
    split: str = "val"  # math pools: train/val/test (never sample the sealed test split); repeat_direct_correct: train/eval
    n_problems: int = chz.field(default=5, validator=gt(0))  # small while iterating on prompts
    problem_offset: int = chz.field(default=0, validator=ge(0))
    samples_per_problem: int = chz.field(default=1, validator=gt(0))
    seed_offset: int = chz.field(default=0, validator=ge(0))  # sample i is seeded seed_offset + i
    # JSON merged into the pool task's spec, e.g. warm-ups: '{"warmup": "v1", "warmup_k": 1}'
    task_options: str = ""

    model_name: str = "Qwen/Qwen3.8-27B"
    model_path: str | None = None
    renderer_name: str | None = None
    max_tokens: int = chz.field(default=32768, validator=gt(0))  # per turn; 8192 cut off long turn-2 thinking
    temperature: float = 1.0
    top_p: float = 1.0
    max_concurrency: int = chz.field(default=32, validator=gt(0))

    use_cache: bool = True
    cache_dir: str = str(ROOT / "file_cache")
    cost_file: str = str(ROOT / "costs/total_cost.jsonl")
    output_dir: str = str(ROOT / "results/rollouts")
    run_name: str | None = None  # default: <pool>_<split>_<utc timestamp>


@chz.chz
class TrainConfig:
    """GRPO as in reward laundering (B=8 problems x G=16 samples, lr 1e-4 decaying linearly, LoRA rank 32), with evals
    sized for an approximate +/-2-point margin on the fixed eval questions."""

    pool: str = "gsm8k_gate"  # a key of src.dataset.pools.POOLS
    # a named mix of rules (src/training/curricula.py) trained together instead of `pool`: each step draws
    # batch_size / len(components) questions per rule; evals cover every rule's eval split; rows carry "component"
    curriculum: str | None = None
    # Update s (0..n_steps-1) uses learning_rate x (1 - s / horizon) under the linear schedule, horizon =
    # lr_schedule_steps or n_steps. The post's runs used n_steps=34 lr_schedule_steps=68, so their last update ran at
    # about half the peak (5.15e-5); configs/ records each figure's exact settings.
    n_steps: int = chz.field(default=34, validator=gt(0))
    batch_size: int = chz.field(default=8, validator=gt(0))  # problems per step
    group_size: int = chz.field(default=16, validator=gt(1))  # samples per problem; advantages need G > 1
    learning_rate: float = chz.field(default=1e-4, validator=gt(0))  # the original reward-laundering experiment's value
    lr_schedule: Literal["linear", "cosine", "constant"] = chz.field(default="linear", validator=typecheck)
    lr_schedule_steps: int | None = None  # the schedule's horizon (default n_steps); never shorter than n_steps
    # grpo: GRPO on the task's reward. grpo_shuffled: the same, but each group's rewards are permuted among its episodes
    # before advantages are computed (the same groups update, with the same advantage values, credited to random
    # episodes). The no-reward ablations fine-tune (cross-entropy) on every episode of the step's groups, sampled from
    # the current policy (sft_on_policy) or from the base model (sft_base_rollouts); problems, seeds and evals are as
    # in grpo.
    objective: Literal["grpo", "grpo_shuffled", "sft_on_policy", "sft_base_rollouts"] = chz.field(
        default="grpo", validator=typecheck)
    shuffle_seed: int = 0  # grpo_shuffled: the permutation of group g at step s is seeded by (shuffle_seed, s, g)
    # SFT token weights: sum = each sampled token weight 1 (under ideal exact on-policy sampling its expected raw
    # gradient, before Adam, is zero; the post's SFT ablations); mean = each episode's weights sum to 1, which on
    # regenerated data acts as a reward of 1/length
    sft_reduction: Literal["sum", "mean"] = chz.field(default="sum", validator=typecheck)
    # JSON merged into the pool task's spec (as in `rollouts`), e.g. '{"warmup": "v1", "warmup_k": 2}'; recorded in
    # the manifest's task spec, so evals, checkpoints and Docent see the same prompt
    task_options: str = ""
    lora_rank: int = chz.field(default=32, validator=gt(0))
    seed: int = 0  # training question order only (sampling seeds and LoRA init do not depend on it)
    lora_seed: int = 0  # initialization of the fresh LoRA

    eval_every: int = chz.field(default=17, validator=gt(0))  # also at step 0 and after the last step: 0/17/34
    eval_n_problems: int = chz.field(default=100, validator=ge(0))  # 0 disables eval
    eval_split: str = "val"  # the pool split evals sample: "val" for the GSM8K pools (never "test"), "eval" for direct
    eval_margin: float = chz.field(default=0.02, validator=gt(0))  # 95% half-width at worst case: sets samples/question

    model_name: str = "Qwen/Qwen3.8-27B"
    renderer_name: str | None = None
    max_tokens: int = chz.field(default=32768, validator=gt(0))
    temperature: float = 1.0
    top_p: float = 1.0
    max_concurrency: int = chz.field(default=512, validator=gt(0))

    wandb_project: str | None = None  # a W&B project to chart the run in (opt-in)
    docent: bool = False  # stream train/eval transcripts to Docent while training (opt-in; needs DOCENT_API_KEY)
    cost_file: str = str(ROOT / "costs/total_cost.jsonl")
    output_dir: str = str(ROOT / "results/train")
    checkpoint_index: str = str(ROOT / "results/checkpoints.jsonl")
    run_name: str | None = None  # default: <pool>_<objective>_<utc timestamp>


@chz.chz
class GradeConfig:
    run_dir: str  # a finished `rollouts` run directory


async def _run_rollouts(config: RolloutsConfig) -> None:
    from src.rollouts.run import run

    await run(config)


async def _run_grade(config: GradeConfig) -> None:
    from src.rollouts.grading import grade_run

    print(json.dumps(grade_run(Path(config.run_dir)), indent=2))


async def _run_train(config: TrainConfig) -> None:
    import subprocess
    import threading

    from src.training.grpo import run

    if config.curriculum and config.docent:  # before the streamer starts
        raise ValueError("curriculum runs are not exported to Docent (scripts/docent reads one task per run); pass docent=false")
    run_name = config.run_name or f"{config.curriculum or config.pool}_{config.objective}_{datetime.now(UTC):%Y%m%d_%H%M%S}"
    config = chz.replace(config, run_name=run_name)
    run_dir = (Path(config.output_dir) / run_name).resolve()
    if run_dir.exists():
        raise ValueError(f"{run_dir} exists; runs are never overwritten")
    catch_up = f"uv run python -m scripts.docent.stream run_dir={run_dir}"
    # Docent stays out of src/: a separate process follows the run's sealed step files and exits once it is finished.
    streamer, training = None, threading.Event()
    if config.docent:
        streamer = subprocess.Popen(  # noqa: ASYNC220
            [sys.executable, "-m", "scripts.docent.stream", f"run_dir={run_dir}"], cwd=ROOT)

        def warn_if_streamer_dies() -> None:  # tell the user at once, not hours later when training ends
            if streamer.wait() != 0 and training.is_set():
                print(f"WARNING: the Docent streamer exited ({streamer.returncode}) while training continues; local "
                      f"files are unaffected. Catch up with: {catch_up}", file=sys.stderr, flush=True)

        training.set()
        threading.Thread(target=warn_if_streamer_dies, daemon=True).start()
    try:
        await run(config)
    finally:
        training.clear()  # from here on, an exit is expected or reported below
        if streamer is not None:
            # A run that failed before writing its manifest has nothing to upload; otherwise the manifest's final
            # status tells the streamer to upload what is sealed and exit.
            if not (run_dir / "manifest.json").exists():
                streamer.terminate()
            print(f"waiting for the Docent streamer to finish uploading {run_dir} ...")
            try:
                code = streamer.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                streamer.terminate()
                code = "timeout"
            if code != 0 and (run_dir / "manifest.json").exists():
                print(f"The local run ended (see {run_dir / 'manifest.json'} for its status), but the Docent export "
                      f"did not finish ({code}). "
                      f"Catch up with: {catch_up}", file=sys.stderr)


async def _run_make_pools(config: MakePoolsConfig) -> None:
    from src.dataset.pools import write_pool

    for path in write_pool(config.pool):
        print(f"wrote {path}")


COMMANDS = {
    "make-pools": (MakePoolsConfig, _run_make_pools),
    "rollouts": (RolloutsConfig, _run_rollouts),
    "grade": (GradeConfig, _run_grade),
    "train": (TrainConfig, _run_train),
}
USAGE = f"Usage: rl <command> [key=value ...]\nCommands: {', '.join(COMMANDS)}\nRun `rl <command> --help` for options."


def _parse_config(config_cls: type, args: list[str]):
    try:
        return chz.Blueprint(config_cls).make_from_argv(args)
    except EntrypointHelpException as exc:
        print(exc, end="" if str(exc).endswith("\n") else "\n")
        raise SystemExit(0) from None
    except (EntrypointException, ValueError) as exc:
        print("Error:", file=sys.stderr)
        print(exc, end="" if str(exc).endswith("\n") else "\n", file=sys.stderr)
        raise SystemExit(1) from None


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(USAGE)
        return
    if args[0] not in COMMANDS:
        raise SystemExit(f"Unknown command: {args[0]}\n{USAGE}")
    config_cls, runner = COMMANDS[args[0]]
    config = _parse_config(config_cls, args[1:])
    from src.backends.tinker.clients import close_service_client

    status = "errored"
    try:
        asyncio.run(runner(config))
        status = "success"
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    finally:
        close_service_client(status)


if __name__ == "__main__":
    main()
