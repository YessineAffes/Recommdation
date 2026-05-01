from __future__ import annotations

from core.state_machine import get_next_action, should_skip_action


def test_age_gt_60_skips_declared_questions():
    state = {"Q1_age": 65}
    for action in [6, 10, 11, 16, 17, 21]:
        assert should_skip_action(action, state)


def test_age_gt_60_from_action5_jumps_after_q7():
    assert get_next_action(5, {"Q1_age": 65}) == 8


def test_q7_only_when_q6_tres_souvent():
    assert get_next_action(6, {"Q1_age": 45, "Q4_usage_ordinateur": "tres_souvent"}) == 7
    assert get_next_action(6, {"Q1_age": 45, "Q4_usage_ordinateur": "parfois"}) == 9
    assert should_skip_action(7, {"Q1_age": 45, "Q4_usage_ordinateur": "parfois"})


def test_action3_add_shortcuts():
    assert get_next_action(3, {"Q1_age": 45, "Q3_add": 2.75}) == 9
    assert get_next_action(3, {"Q1_age": 45, "Q3_add": 2.25}) == 8
    assert get_next_action(3, {"Q1_age": 45, "Q3_add": 1.75}) == 4
