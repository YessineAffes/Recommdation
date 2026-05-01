from __future__ import annotations

from core.state_machine import get_next_action, should_skip_action


def test_age_gt_60_skips_declared_questions():
    state = {"Q1_age": 65}
    for action in [6, 10, 11, 16, 17, 21]:
        assert should_skip_action(action, state)


def test_age_lt_42_skips_varilux_design_questions():
    state = {"Q1_age": 36}
    for action in [4, 5, 6, 7, 8, 9, 10]:
        assert should_skip_action(action, state)
    assert get_next_action(3, {"Q1_age": 36, "Q3_add": 1.75}) == 11


def test_duplicate_ai_question_is_skipped():
    assert should_skip_action(10, {"Q1_age": 45})
    assert get_next_action(9, {"Q1_age": 45, "Q4_innovation": False}) == 11


def test_age_gt_60_from_action5_jumps_after_q7():
    assert get_next_action(5, {"Q1_age": 65}) == 8


def test_q7_only_when_q6_tres_souvent():
    assert get_next_action(6, {"Q1_age": 45, "Q4_usage_ordinateur": "tres_souvent"}) == 7
    assert get_next_action(6, {"Q1_age": 45, "Q4_usage_ordinateur": "parfois"}) == 9
    assert should_skip_action(7, {"Q1_age": 45, "Q4_usage_ordinateur": "parfois"})


def test_solar_followups_are_skipped_after_strong_sun_signal():
    state = {"Q1_age": 45, "Q13_paire_soleil": "oui_tres_expose"}
    assert should_skip_action(14, state)
    assert should_skip_action(15, state)
    assert get_next_action(13, state) == 16


def test_full_sun_question_skipped_after_reflets():
    state = {"Q1_age": 45, "Q7_reflets_genants": True}
    assert should_skip_action(15, state)
    assert get_next_action(14, state) == 16


def test_action3_add_shortcuts():
    assert get_next_action(3, {"Q1_age": 45, "Q3_add": 2.75}) == 9
    assert get_next_action(3, {"Q1_age": 45, "Q3_add": 2.25}) == 8
    assert get_next_action(3, {"Q1_age": 45, "Q3_add": 1.75}) == 4
