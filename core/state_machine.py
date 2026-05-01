from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any

DYNAMIC_RULES_PATH = Path(__file__).resolve().parent / "dynamic_rules.json"


def load_rules(path: str | Path = DYNAMIC_RULES_PATH) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _field_map(rules: dict[str, Any]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for q in rules.get("questions", []):
        state_key = q.get("state_key")
        for key in (q.get("key"), q.get("state_key"), q.get("qid")):
            if key and state_key:
                aliases[str(key)] = str(state_key)
        for sub in q.get("sub_questions", []) or []:
            sub_state = sub.get("state_key")
            for key in (sub.get("key"), sub_state):
                if key and sub_state:
                    aliases[str(key)] = str(sub_state)
    return aliases


def _norm(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = unicodedata.normalize("NFKD", value.strip().lower())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def _number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", "."))
    except ValueError:
        return None


def _value(state: dict[str, Any], field: str, aliases: dict[str, str]) -> Any:
    state_key = aliases.get(field, field)
    return state.get(state_key, state.get(field))


def condition_matches(state: dict[str, Any], condition: dict[str, Any] | None, rules: dict[str, Any] | None = None) -> bool:
    if not condition:
        return True
    rules = rules or load_rules()
    aliases = _field_map(rules)
    if "all" in condition:
        return all(condition_matches(state, c, rules) for c in condition["all"])
    if "any" in condition:
        return any(condition_matches(state, c, rules) for c in condition["any"])
    if "not" in condition:
        return not condition_matches(state, condition["not"], rules)

    field = condition.get("field")
    op = condition.get("operator", "==")
    expected = condition.get("value")
    actual = _value(state, str(field), aliases) if field else None

    if op in (">", ">=", "<", "<="):
        a = _number(actual)
        b = _number(expected)
        if a is None or b is None:
            return False
        if op == ">":
            return a > b
        if op == ">=":
            return a >= b
        if op == "<":
            return a < b
        return a <= b
    if op in ("==", "equals"):
        return _norm(actual) == _norm(expected)
    if op in ("!=", "neq"):
        return _norm(actual) != _norm(expected)
    if op == "in":
        values = expected if isinstance(expected, list) else [expected]
        return _norm(actual) in {_norm(v) for v in values}
    return False


def should_skip_action(action: int, state: dict[str, Any], rules: dict[str, Any] | None = None) -> bool:
    rules = rules or load_rules()
    for skip_rule in rules.get("skip_rules", []):
        if int(action) in {int(x) for x in skip_rule.get("skip_questions", [])}:
            if condition_matches(state, skip_rule.get("condition"), rules):
                return True
    return False


def _next_candidate(current: int, state: dict[str, Any], rules: dict[str, Any]) -> int:
    default_next = current + 1
    for rule in rules.get("navigation_rules", []):
        if int(rule.get("from", -1)) != int(current):
            continue
        if "condition" in rule and condition_matches(state, rule.get("condition"), rules):
            return int(rule["next"])
        if "default_next" in rule:
            default_next = int(rule["default_next"])
    return default_next


def get_next_action(current: int, state: dict[str, Any], rules_path: str | Path = DYNAMIC_RULES_PATH) -> int:
    """Navigation des 22 actions pilotee par dynamic_rules.json."""
    rules = load_rules(rules_path)
    if current >= 22:
        return 23
    nxt = _next_candidate(current, state, rules)
    while nxt <= 22 and should_skip_action(nxt, state, rules):
        nxt += 1
    return nxt


def get_question(action: int, rules_path: str | Path = DYNAMIC_RULES_PATH) -> dict[str, Any] | None:
    rules = load_rules(rules_path)
    for q in rules.get("questions", []):
        if int(q.get("id", -1)) == int(action):
            return q
    return None


def get_question_flow(rules_path: str | Path = DYNAMIC_RULES_PATH) -> list[dict[str, Any]]:
    rules = load_rules(rules_path)
    return sorted(rules.get("questions", []), key=lambda q: int(q.get("id", 999)))
