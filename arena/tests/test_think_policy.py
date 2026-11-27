"""Spec v7 6.5: the arena's think-policy budget functions."""

import pytest

from arena import think_policy


def test_scale_clamps_and_is_one_at_a_fresh_head():
    assert think_policy.scale(0.0) == 0.5
    assert think_policy.scale(1.0) == 2.0
    assert think_policy.scale(-3.0) == 0.5
    assert think_policy.scale(5.0) == 2.0
    assert think_policy.scale(1 / 3) == pytest.approx(1.0)
    assert think_policy.scale(0.5) == pytest.approx(1.25)


def test_budget_without_a_head_is_the_clock_budget():
    assert think_policy.budget(800, None, 100, 400) == 800


def test_budget_scales_and_pays_the_quick_look():
    q = think_policy.QUICK_LOOK_NODES
    assert q == 16
    assert think_policy.budget(800, 1 / 3, 1, 10_000) == 800 - q
    assert think_policy.budget(800, 1.0, 1, 10_000) == 1600 - q
    assert think_policy.budget(800, 0.0, 1, 10_000) == 400 - q
    assert think_policy.budget(101, 0.5, 1, 10_000) == round(101 * 1.25) - q


def test_budget_clamps():
    assert think_policy.budget(800, 1.0, 1, 1000) == 1000
    assert think_policy.budget(20, 0.0, 50, 1000) == 50
    assert isinstance(think_policy.budget(800, 0.4, 1, 10_000), int)
