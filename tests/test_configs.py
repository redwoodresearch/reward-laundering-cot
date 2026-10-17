"""configs/post_runs.json: every recorded command parses and sets up offline, as the run it names."""

import json
from pathlib import Path

import chz
import pytest

import main
from src.dataset.pools import load_pool, pool_task
from src.training import grpo

RECIPES = json.loads((Path(__file__).resolve().parents[1] / "configs/post_runs.json").read_text())


@pytest.mark.parametrize("recipe", RECIPES, ids=[r["run"] for r in RECIPES])
def test_every_recipe_sets_up_as_the_run_it_names(recipe, tmp_path, monkeypatch):
    argv = recipe["argv"]
    if argv[0] == "scripts/experiments/cotcontrol_qa.py":
        from scripts.experiments.cotcontrol_qa import Config

        config = chz.Blueprint(Config).apply_from_argv(argv[1:]).make()
        assert Path(config.out_dir).name == recipe["run"]
        return
    kind, args = argv[1], argv[2:] + (["model_path=tinker://checkpoint"] if "checkpoint" in recipe else [])
    config = main._parse_config(main.COMMANDS[kind][0], args)
    assert config.run_name == recipe["run"]
    if kind == "train":  # the trainer's own guards, tasks and pools; tokenizing every prompt is left to tests/golden
        monkeypatch.setattr(grpo, "get_tokenizer_and_renderer", lambda model, renderer: (None, None))
        monkeypatch.setattr(grpo, "plan_episode", lambda task, problem, tokenizer, renderer: None)
        grpo.Trainer(chz.replace(config, cost_file=str(tmp_path / "costs.jsonl")), tmp_path / "run")
    else:
        pool_task(config.pool, config.task_options)
        assert len(load_pool(config.pool, config.split)) >= config.problem_offset + config.n_problems
        assert config.use_cache is False and config.model_path  # every recorded evaluation sampled fresh, a checkpoint
