# Analysis: from run outputs to the post's figures

Three steps, each runnable on its own from the repository root:

1. **Render** the figure page from the committed figure data (no results needed). Every figure is drawn as an SVG
   and turned into a PNG (right-click to copy); `tests/test_figures.py` checks the page against the published one.

       uv run python analysis/build_figures.py --out figures.html

2. **Stage 2**: the figure data (`figures/data/<figure>.json`) from the committed intermediates (`data/intermediate/`).
   It only selects, averages and rounds.

       uv run python -m analysis.compute.figures            # --check compares instead of writing

3. **Stage 1**: the intermediates from the raw run outputs (not published), under a results root laid out as
   `results/`. Each module states its definitions in its docstring:

   | Intermediate | `uv run python -m analysis.compute.<module> --results <root> --out <file>` | Figures |
   |---|---|---|
   | `cond_leakage.json`: eval counts per run and step | `leakage` | 1, 2b, 3, 5, 6 |
   | `grid.json`: training reward gap split by 3-step window | `grid` (also `--leakage <cond_leakage.json>`) | 1 |
   | `maths_t1.json`: turn-1 maths right, read from turn 1 only | `maths` | 1 |
   | `exposure.json`: training episodes pushed up or down | `exposure` | 2, 2b |
   | `ablations.json`: direct-reward eval compliance | `ablations` | 3 |
   | `transfer.json`: transfer evals with question-bootstrap intervals | `transfer` | 4 |
   | `cotcontrol.json`: CoT-Control-QA compliance with intervals | `cotcontrol` | 4b |
   | `curriculum.json`: five-rule run evals per rule and step | `curriculum` | 5 |
   | `length.json`: word-suppression eval length at steps 6 and 34 | `length` | 6 |

   Build `cond_leakage.json` first: exposure and curriculum read it from `--data` (default `data/intermediate/`).
   Leakage, grid, exposure, curriculum and maths refuse partial runs (missing or empty step files, short evals). The
   maths step makes no API calls: it reads cached Claude Haiku 5.5 extractions (with a Claude Sonnet 5.5 check on a
   subset) and the extractor's saved inputs (`--inputs`), and a cached reply counts only if the episode's text is
   that saved input and contains its evidence quote. The CoT-Control step reads only runs that pass the grading
   checks (`scripts/experiments/cotcontrol_checks.py`), which need the pinned CoT-Control clone.

## Runs behind each figure

Directories under the results root (`train/<run>`, `rollouts/<run>`, `cotcontrol/<run>`), also listed in
`compute/runs.py`; `configs/post_runs.json` has each one's command. All trained 34 steps except `direct_grpo_r2`
(68; the figures show steps 0-34, over which its learning rates were the same).

| Figure | Runs |
|---|---|
| 1, 2, 2b | GRPO, 4 seeds per rule: `gsm8k_gate_grpo_seed0_e3_20261008_184834` and `twoturn_gsm8k_grpo_e3_seed{1,2,3}` (repeat sentences); `task33_{lowercase_thinking_k1,ok_start_sentence_k1,word_suppression_k2,meow_between_words_k1}_grpo_seed{0..3}` |
| 3 | GRPO repeat sentences as above; `twoturn_gsm8k_grpo_shuffled_e3_seed{0..3}`; `twoturn_gsm8k_sft_on_policy_sum_e3_seed{0..3}`; `twoturn_gsm8k_sft_base_rollouts_sum_e3_seed{0..3}`; direct reward `direct_grpo_r2` and `task40_direct_grpo_seed{1,2,3}` (own evals at steps 0/17/34, plus `rollouts/task41_eval_<run>_step<k>_retry1` every 3 steps; `direct_grpo_r2` step 3 uses `_retry2`, as `_retry1` stopped incomplete; seed 2's step 30 and seed 3's step 24 used the step-29 and step-23 checkpoints) |
| 4 | `rollouts/task38_transfer_rs_seed{1,2}_on_<rule>` (step-34 checkpoints of repeat-sentences seeds 1 and 2); untrained = the step-0 evals of each rule's GRPO runs |
| 4b | `cotcontrol/full2_{base,repeat_seed1_step34,repeat_seed2_step34}` |
| 5 | `task39_curriculum_five_rules_seed{0..3}`; grey lines = Figure 1's single-rule means |
| 6 | `task33_word_suppression_k2_grpo_seed{0..3}`, evals at steps 6 and 34 |
