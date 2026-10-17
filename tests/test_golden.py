"""The code reproduces what the published experiments saw (tests/golden/freeze.py; values frozen in golden.json by
the research code before the public-release refactor)."""

import json
from pathlib import Path

import pytest

from tests.golden import freeze

GOLDEN = json.loads((Path(__file__).parent / "golden" / "golden.json").read_text())


@pytest.mark.parametrize("label", [s[0] for s in freeze.SETUPS])
def test_prompts_and_pool_sizes_are_unchanged(label):
    assert freeze.prompts()[label] == GOLDEN["prompts"][label]


def test_question_order_and_curriculum_are_unchanged():
    assert freeze.orders() == GOLDEN["orders"]
    assert freeze.curriculum_definition() == GOLDEN["curriculum_definition"]


def test_shuffled_reward_permutations_are_unchanged():
    assert freeze.shuffles() == GOLDEN["shuffles"]


def test_the_trainers_learning_rates_are_the_published_schedule():
    """The research trainer computed learning_rate x multiplier; the multipliers were frozen."""
    assert freeze.learning_rates() == [1e-4 * m for m in GOLDEN["lr_multipliers"]]


def test_grader_verdicts_on_saved_episodes_are_unchanged():
    assert freeze.grades(GOLDEN["cases"]) == GOLDEN["grades"]
