# Data

Everything the post's runs trained and evaluated on, committed so the experiments do not depend on fresh downloads or
paid generations. Files are never rewritten in place: builders refuse to overwrite them, and a changed file gets a new
name. Provenance fields inside the files (e.g. `annotation_file`, `source.run`) name the research run directories they
were built from; those raw runs are not published.

## sources/
- `math_gsm8k.jsonl`: GSM8K (openai/gsm8k, MIT) at a pinned revision, integer answers only; the official test split is
  `test`, `val` is a hash-chosen 10% of the official train split, train copies of test questions are dropped. Built by
  `scripts/data/build_math_sources.py` (pins and counts in `curation/math_sources_report.json`).
- `side_mmlu_pro_mini.jsonl`: the 300 MMLU-Pro questions of CoT-Control's `mmlu_pro_mini_w_keyword.csv`
  (github.com/YuehHanChen/CoTControl @ 5d78aef, MIT), for the direct-reward control. Built by
  `scripts/data/build_side_questions.py`.

## pools/
The problems each run sampled, as `rl make-pools` wrote them from `src/dataset/pools.py`'s recipes (every load re-checks
each stored answer against its task). `gsm8k_gate_*` and `gsm8k_cot_warmup_*` are symlinks to `math_gsm8k_*` (same rows).
The `*_rl_gate` pools leave out each rule's warm-up source questions and calibration questions
(`curation/rl_exclusions_<rule>.json`); word suppression keeps only questions with a selected keyword.

## keywords/
Word-suppression keywords: per question, the word to avoid and its banned forms. Proposed by a fixed annotation prompt
(`scripts/data/annotate_keywords.py`), chosen as the highest-ranked candidate that two plain solutions of the question
use at least once, else the top candidate (`scripts/data/select_keywords.py`), then corrected
(`scripts/data/correct_keywords.py`). The plain solutions come from the plain maths pool, on the same split, offset and
count as the annotation, with exactly two samples per question:

    uv run python -m scripts.data.annotate_keywords split=train offset=0 n=1000 out=<annotations>.jsonl
    uv run rl rollouts pool=math_gsm8k split=train problem_offset=0 n_problems=1000 samples_per_problem=2 run_name=<run>
    uv run python -m scripts.data.select_keywords annotations=<annotations>.jsonl usage_run=results/rollouts/<run> \
        out=<name>.jsonl

The pools use the v2 files; they rebuild byte for byte from the committed earlier files and correction records:

    uv run python -m scripts.data.correct_keywords sources=word_suppression_keywords_train_0_1000.jsonl \
        corrections=data/curation/keyword_corrections_v2_train_0_1000.json out=<name>.jsonl
    uv run python -m scripts.data.correct_keywords \
        sources=word_suppression_keywords_pilot.jsonl,word_suppression_keywords_val_10_130.jsonl \
        corrections=data/curation/keyword_corrections_v2_val_0_130.json out=<name>.jsonl

## warmups/
The warm-up exchanges shown before the gate for lowercase, "OK" at sentence start, meow between words (`k=1`) and word
suppression (`k=2`): the model's own traces under the warm-up message, made compliant by a deterministic transform or,
for word suppression, a teacher's rewrite. The traces come from the warm-up message alone, on train questions:

    # lowercase, "OK" at sentence start, meow between words: a deterministic transform
    uv run rl rollouts pool=gsm8k_cot_warmup split=train task_options='{"constraint": "lowercase_thinking"}' \
        n_problems=<n> run_name=<run>
    uv run python -m scripts.data.build_warmup_pool source=results/rollouts/<run> constraint=lowercase_thinking version=<v>
    # word suppression: its own pool (questions with keywords, in their original forms), then a teacher's rewrites
    uv run rl rollouts pool=gsm8k_word_suppression_train_warmup split=train n_problems=<n> run_name=<run>
    uv run python -m scripts.data.rewrite_warmup source=results/rollouts/<run> constraint=word_suppression \
        teacher=Qwen/Qwen3.8-27B n=<n> out=results/warmup_rewrites/<name>
    uv run python -m scripts.data.build_warmup_pool source=results/rollouts/<run> constraint=word_suppression \
        rewrites=results/warmup_rewrites/<name>/rewrites.jsonl version=<v>

A pool keeps 16 warm-ups (build_warmup_pool's `m`), so rewrite more traces than that (`n`, default 10), since some
rewrites fail the grader.

**Do not edit these files, not even whitespace**: each question's warm-ups are drawn by a permutation keyed by the
sha256 of the file's bytes (`src/tasks/warmups.py`). `calibration_v1.json` is the historical record of how the number
of warm-ups was chosen (its protocol and amendments, with the paths of the time and rules later dropped); the
published setup is the four `.v1.jsonl` files above with the warm-up counts in `configs/post_runs.json`.

## curation/
Audited exclusions (`math_exclusions.jsonl`), the GSM8K build report, the per-rule RL exclusions, and the keyword
correction records.
