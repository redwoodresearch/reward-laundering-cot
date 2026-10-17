"""Named multi-rule training mixes for `train curriculum=<name>`: (key, pool, task_options) per component, each with
the prompt its single-rule runs used. Every step draws batch_size / len(components) questions from each component."""

CURRICULA: dict[str, tuple[tuple[str, str, str], ...]] = {
    # repeat sentences + the four other rules, each with the warm-ups of its single-rule runs (meow graded with v2)
    "five_rules_v1": (
        ("repeat_sentences", "gsm8k_gate", ""),
        ("lowercase", "gsm8k_lowercase_thinking_rl_gate", '{"warmup": "v1", "warmup_k": 1}'),
        ("ok_start", "gsm8k_ok_start_sentence_rl_gate", '{"warmup": "v1", "warmup_k": 1}'),
        ("word_suppression", "gsm8k_word_suppression_rl_gate", '{"warmup": "v1", "warmup_k": 2}'),
        ("meow", "gsm8k_meow_between_words_rl_gate", '{"warmup": "v1", "warmup_k": 1}'),
    ),
}
