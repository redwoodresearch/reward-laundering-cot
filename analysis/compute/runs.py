"""The runs behind the post's figures, by role: directory names under a results root (`train/<run>` for training runs,
`rollouts/<run>` for separate evaluation runs, `cotcontrol/<run>` for CoT-Control-QA runs). Every compute script takes
that root as `--results`; `check_steps` is the only thing here that reads it."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

TRAIN_STEPS = tuple(range(34))  # training batches of every post run (steps 0-33)
EVAL_STEPS = (*range(0, 34, 3), 34)  # single-rule runs: evals at step 0, every 3 steps and after the last update
CURRICULUM_EVAL_STEPS = (*range(0, 34, 6), 34)
EVAL_EPISODES = 2500  # 100 validation questions x 25 samples, per rule

# rule key (CoT-Control mode) -> the name the figures show
RULES = {"repeat_sentences": "Repeat sentences", "lowercase_thinking": "Lowercase",
         "ok_start_sentence": '"OK" at sentence start', "word_suppression": "Word suppression",
         "meow_between_words": "Meow between words"}

# GRPO on each rule, seeds 0-3 (Figures 1, 2, 2b, 5's grey lines, 6)
GRPO = {"repeat_sentences": ["gsm8k_gate_grpo_seed0_e3_20261008_184834", *(f"twoturn_gsm8k_grpo_e3_seed{s}" for s in (1, 2, 3))],
        **{rule: [f"task33_{rule}_k{k}_grpo_seed{s}" for s in range(4)]
           for rule, k in (("lowercase_thinking", 1), ("ok_start_sentence", 1), ("word_suppression", 2),
                           ("meow_between_words", 1))}}

# repeat-sentences ablations, seeds 0-3 (Figure 3)
SHUFFLED = [f"twoturn_gsm8k_grpo_shuffled_e3_seed{s}" for s in range(4)]
SFT_ON_POLICY = [f"twoturn_gsm8k_sft_on_policy_sum_e3_seed{s}" for s in range(4)]
SFT_BASE_ROLLOUTS = [f"twoturn_gsm8k_sft_base_rollouts_sum_e3_seed{s}" for s in range(4)]

# the direct-reward control (Figure 3): four training runs (their own evals at steps 0, 17, 34) plus separate evals of
# their checkpoints every 3 steps; direct_grpo_r2 trained 68 steps, the figure shows steps 0-34
DIRECT = ["direct_grpo_r2", *(f"task40_direct_grpo_seed{s}" for s in (1, 2, 3))]
DIRECT_EVAL_STEPS = tuple(range(3, 34, 3))


def direct_eval(run: str, step: int) -> str:
    """The separate eval run used for `run` at `step`: the first attempt, except direct_grpo_r2 step 3, whose first
    attempt was stopped incomplete and rerun."""
    attempt = 2 if (run, step) == ("direct_grpo_r2", 3) else 1
    return f"task41_eval_{run}_step{step:04d}_retry{attempt}"


# all five rules trained together, seeds 0-3 (Figure 5)
CURRICULUM = [f"task39_curriculum_five_rules_seed{s}" for s in range(4)]

# step-34 checkpoints of repeat-sentences seeds 1 and 2 evaluated on the four other rules (Figure 4); the untrained
# model is the four rules' GRPO runs at step 0
TRANSFER_RULES = ("lowercase_thinking", "ok_start_sentence", "word_suppression", "meow_between_words")


def transfer(seed: int, rule: str) -> str:
    return f"task38_transfer_rs_seed{seed}_on_{rule}"


# Chen et al.'s CoT-Control-QA (Figure 4b): figure label -> cotcontrol run
COTCONTROL = {"untrained": "full2_base", "rs seed 1": "full2_repeat_seed1_step34", "rs seed 2": "full2_repeat_seed2_step34"}


def check_steps(results: Path, kind: str, run_names: Iterable[str], steps: Iterable[int]) -> None:
    """Refuses any run whose `train/<run>/<kind>/step_<k>.jsonl` files are not exactly `steps`, or include an empty one,
    so a partial (or longer) rerun cannot silently replace a full curve."""
    for run in run_names:
        files = {int(p.stem.removeprefix("step_")): p for p in (results / "train" / run / kind).glob("step_*.jsonl")}
        if sorted(files) != sorted(steps):
            raise SystemExit(f"{results / 'train' / run / kind}: steps {sorted(files)}, expected {sorted(steps)}")
        if empty := [s for s, p in sorted(files.items()) if p.stat().st_size == 0]:
            raise SystemExit(f"{results / 'train' / run / kind}: empty step files {empty}")
