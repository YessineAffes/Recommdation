from __future__ import annotations

import json
import unicodedata
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

ROOT = Path(__file__).resolve().parent.parent
DYNAMIC_RULES_PATH = Path(__file__).resolve().parent / "dynamic_rules.json"

_NUMERIC_BUCKETS = {
    "<200": 199.0,
    "225-400": 400.0,
    "425-600": 600.0,
    ">625": 626.0,
    ">=625": 626.0,
    ">800": 801.0,
    ">=800": 801.0,
}


def _rank_key(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


VARILUX_RANKS = {
    _rank_key("Varilux Liberty 3.0"): 1,
    _rank_key("Varilux Comfort 3.0"): 2,
    _rank_key("Varilux Comfort Max"): 2,
    _rank_key("Varilux Physio 3.0"): 3,
    _rank_key("Varilux S Design"): 4,
    _rank_key("Varilux X Design"): 5,
    _rank_key("Varilux XR"): 6,
}


CRIZAL_RANKS = {
    _rank_key("Crizal Easy Pro"): 1,
    _rank_key("Crizal Rock"): 2,
    _rank_key("Crizal Sapphire HR"): 3,
    _rank_key("Crizal Prevencia"): 3,
    _rank_key("Crizal Drive"): 3,
}


def keep_highest_ranked(
    current: str | None,
    candidate: Any,
    ranks: dict[str, int],
    current_rank: int = 0,
    *,
    allow_equal_override: bool = False,
) -> tuple[str | None, int, bool]:
    """Return current or candidate, keeping only the highest ranked value."""
    if candidate is None:
        return current, current_rank, False

    candidate_value = str(candidate)
    candidate_rank = ranks.get(_rank_key(candidate_value), 0)

    if not current:
        return candidate_value, candidate_rank, True

    if candidate_rank == 0 and current_rank == 0:
        return candidate_value, candidate_rank, True

    if candidate_rank > current_rank:
        return candidate_value, candidate_rank, True

    if allow_equal_override and candidate_rank == current_rank and candidate_rank > 0:
        return candidate_value, candidate_rank, True

    return current, current_rank, False


class DynamicRulesConfig(BaseModel):
    version: str
    metadata: dict[str, Any]
    priority_rules: list[dict[str, Any]]
    questions: list[dict[str, Any]]
    skip_rules: list[dict[str, Any]] = Field(default_factory=list)
    navigation_rules: list[dict[str, Any]] = Field(default_factory=list)
    feedback_messages: dict[str, Any] = Field(default_factory=dict)
    engine: dict[str, Any] = Field(default_factory=dict)
    product_badges: dict[str, Any] = Field(default_factory=dict)
    rag_tech_messages: dict[str, str] = Field(default_factory=dict)
    rules: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("questions")
    @classmethod
    def _must_have_22_actions(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        action_ids = {int(q["id"]) for q in value if "id" in q}
        missing = set(range(1, 23)) - action_ids
        if missing:
            raise ValueError(f"Questions manquantes: {sorted(missing)}")
        return value


@dataclass
class Recommendation:
    type_verre: str
    indice: str
    design_varilux: str | None
    traitement: str
    couleur: str
    corridor_type: str | None = None
    extras: list[str] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type_verre": self.type_verre,
            "indice": self.indice,
            "design_varilux": self.design_varilux,
            "traitement": self.traitement,
            "couleur": self.couleur,
            "corridor_type": self.corridor_type,
            "extras": list(self.extras),
            "trace": list(self.trace),
        }


@dataclass
class _DraftRecommendation:
    type_verre: str
    indice: str
    design_varilux: str | None
    traitement: str
    couleur: str
    corridor_type: str | None = None
    extras: list[str] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    design_rank: int = 0
    traitement_rank: int = 0


class DecisionEngine:
    """Moteur deterministe: 0 LLM, 100% pilote par core/dynamic_rules.json."""

    def __init__(
        self,
        rules_path: str | Path | None = None,
        dynamic_rules_path: str | Path | None = None,
        corrections_store=None,
    ):
        selected = dynamic_rules_path
        if selected is None and rules_path is not None:
            candidate = Path(rules_path)
            if candidate.name == "dynamic_rules.json":
                selected = candidate
        self.dynamic_rules_path = Path(selected) if selected else DYNAMIC_RULES_PATH
        self.corrections_store = corrections_store
        self.config = self._load_config()
        self.rules = self.config.model_dump()
        self._rebuild_indexes()

    def _load_raw_json(self, path: Path) -> dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"{path} doit contenir un objet JSON")
        return data

    def _load_config(self) -> DynamicRulesConfig:
        base = self._load_raw_json(DYNAMIC_RULES_PATH)
        path = self.dynamic_rules_path
        if path != DYNAMIC_RULES_PATH and path.exists():
            override = self._load_raw_json(path)
            if "priority_rules" in override and "engine" in override:
                data = override
            else:
                data = deepcopy(base)
                data["rules"] = override.get("rules", [])
        else:
            data = base
        try:
            return DynamicRulesConfig.model_validate(data)
        except ValidationError as exc:
            raise ValueError(f"dynamic_rules.json invalide: {exc}") from exc

    def _reload_config(self) -> None:
        self.config = self._load_config()
        self.rules = self.config.model_dump()
        self._rebuild_indexes()

    def _rebuild_indexes(self) -> None:
        self.questions_by_action = {int(q["id"]): q for q in self.config.questions}
        self.field_aliases: dict[str, str] = {}
        for q in self.config.questions:
            state_key = q.get("state_key")
            for key in (q.get("key"), q.get("state_key"), q.get("qid")):
                if key and state_key:
                    self.field_aliases[str(key)] = str(state_key)
            for sub in q.get("sub_questions", []) or []:
                sub_state = sub.get("state_key")
                for key in (sub.get("key"), sub_state):
                    if key and sub_state:
                        self.field_aliases[str(key)] = str(sub_state)

    def _load_dynamic_rules(self) -> list[dict[str, Any]]:
        self._reload_config()
        return sorted(self.config.rules, key=lambda r: r.get("priority", 99))

    def _check_corrections(self, state: dict[str, Any]) -> Recommendation | None:
        if self.corrections_store is None:
            return None
        try:
            override = self.corrections_store.find_override(state)
        except Exception:
            return None
        if not override:
            return None
        final_type = override.get("design_varilux") or override.get("type_verre", "")
        trace = list(override.get("trace", []))
        trace.append("OVERRIDE expert applique (lookup corrections.db)")
        return Recommendation(
            type_verre=final_type,
            indice=override.get("indice", ""),
            design_varilux=final_type,
            traitement=override.get("traitement", ""),
            couleur=override.get("couleur", ""),
            corridor_type=override.get("corridor_type"),
            extras=list(override.get("extras", [])) + ["source=expert_override"],
            trace=trace,
        )

    def _norm(self, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        text = unicodedata.normalize("NFKD", value.strip().lower())
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        return text.replace("é", "e").replace("è", "e")

    def _number(self, value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip()
        if text in _NUMERIC_BUCKETS:
            return _NUMERIC_BUCKETS[text]
        try:
            return float(text.replace(",", "."))
        except ValueError:
            return None

    def _state_value(
        self,
        state: dict[str, Any],
        field: str,
        rec: _DraftRecommendation | None = None,
    ) -> Any:
        if rec is not None and hasattr(rec, field):
            return getattr(rec, field)
        state_key = self.field_aliases.get(field, field)
        if state_key in state:
            return state[state_key]
        if field in state:
            return state[field]
        return None

    def _compare(self, actual: Any, op: str, expected: Any) -> bool:
        if op in (">", ">=", "<", "<="):
            actual_num = self._number(actual)
            expected_num = self._number(expected)
            if actual_num is None or expected_num is None:
                return False
            if op == ">":
                return actual_num > expected_num
            if op == ">=":
                return actual_num >= expected_num
            if op == "<":
                return actual_num < expected_num
            return actual_num <= expected_num

        if op in ("==", "equals"):
            return self._norm(actual) == self._norm(expected)
        if op in ("!=", "neq"):
            return self._norm(actual) != self._norm(expected)
        if op == "in":
            values = expected if isinstance(expected, list) else [expected]
            return self._norm(actual) in {self._norm(v) for v in values}
        if op == "exists":
            return actual is not None
        return False

    def _match_condition(
        self,
        state: dict[str, Any],
        condition: dict[str, Any] | None,
        rec: _DraftRecommendation | None = None,
    ) -> bool:
        if not condition:
            return True
        if "all" in condition:
            return all(self._match_condition(state, c, rec) for c in condition["all"])
        if "any" in condition:
            return any(self._match_condition(state, c, rec) for c in condition["any"])
        if "not" in condition:
            return not self._match_condition(state, condition["not"], rec)
        field = condition.get("field")
        op = condition.get("operator", "==")
        if field is None:
            return False
        actual = self._state_value(state, str(field), rec)
        return self._compare(actual, str(op), condition.get("value"))

    def _resolve_type(self, state: dict[str, Any], trace: list[str]) -> str:
        rule = next((r for r in self.config.priority_rules if r.get("param") == "type_verre"), None)
        if rule is None:
            raise ValueError("Regle prioritaire type_verre manquante")
        age = self._number(self._state_value(state, rule.get("state_key", "age")))
        if age is None:
            raise ValueError("Age manquant: impossible de determiner type_verre")
        for item in rule.get("ranges", []):
            min_ok = "min" not in item or age >= float(item["min"])
            max_ok = "max" not in item or age <= float(item["max"])
            if min_ok and max_ok:
                value = str(item["value"])
                trace.append(f"P1 type_verre={value} (age={int(age)})")
                return value
        raise ValueError(f"Aucune plage type_verre pour age={age}")

    def _resolve_index(self, state: dict[str, Any], trace: list[str]) -> str:
        defaults = self.config.engine.get("defaults", {})
        indice = str(defaults.get("indice", "1.50"))
        for rule in self.config.engine.get("index_rules", []):
            if self._match_condition(state, rule.get("condition")):
                indice = str(rule["value"])
                trace.append(f"Indice {indice}: {rule.get('reason', 'regle dynamique')}")
                break
        return indice

    def _apply_priority_index_floor(self, state: dict[str, Any], draft: _DraftRecommendation) -> None:
        for rule in self.config.priority_rules:
            if rule.get("param") != "indice":
                continue
            if not self._match_condition(state, rule.get("condition"), draft):
                continue
            target = str(rule.get("value"))
            if self._number(draft.indice) is None or self._number(target) is None:
                continue
            if float(self._number(draft.indice)) < float(self._number(target)):
                draft.indice = target
                draft.trace.append(f"P2 indice={target}: {rule.get('reason', 'minimum absolu')}")

    def _apply_priority_design(self, state: dict[str, Any], draft: _DraftRecommendation) -> None:
        if draft.type_verre != "Varilux":
            return
        rules = sorted(self.config.engine.get("priority_design_rules", []), key=lambda r: r.get("priority", 99))
        for rule in rules:
            if self._match_condition(state, rule.get("condition"), draft):
                self._keep_ranked_recommendation(
                    draft,
                    attr="design_varilux",
                    rank_attr="design_rank",
                    value=rule.get("value"),
                    ranks=VARILUX_RANKS,
                    label=f"P{rule.get('priority')} design",
                    reason=str(rule.get("reason", "priorite Varilux")),
                )

    def _keep_ranked_recommendation(
        self,
        draft: _DraftRecommendation,
        *,
        attr: str,
        rank_attr: str,
        value: Any,
        ranks: dict[str, int],
        label: str,
        reason: str,
        allow_equal_override: bool = False,
    ) -> None:
        current = getattr(draft, attr)
        current_rank = int(getattr(draft, rank_attr, 0))
        selected, selected_rank, changed = keep_highest_ranked(
            current,
            value,
            ranks,
            current_rank,
            allow_equal_override=allow_equal_override,
        )
        if changed:
            setattr(draft, attr, selected)
            setattr(draft, rank_attr, selected_rank)
            draft.trace.append(f"{label}={selected}: {reason}")
            return
        if value is not None and str(value) != str(current):
            draft.trace.append(f"{label} conserve {current}: {value} non prioritaire ({reason})")

    def _apply_effect(self, state: dict[str, Any], draft: _DraftRecommendation, effect: dict[str, Any]) -> None:
        if not self._match_condition(state, effect.get("condition"), draft):
            return
        only_if_type = effect.get("only_if_type")
        if only_if_type and draft.type_verre != only_if_type:
            return
        param = effect.get("param")
        value = effect.get("value")
        reason = effect.get("reason", "regle dynamique")

        if param == "type_verre":
            draft.trace.append("Regle ignoree: type_verre blinde par age")
            return

        preserve = effect.get("preserve_if")
        if preserve:
            current = getattr(draft, preserve.get("param"), None)
            if "equals" in preserve and self._compare(current, "==", preserve.get("equals")):
                return
            if "not_equals" in preserve and self._compare(current, "!=", preserve.get("not_equals")):
                return

        if param == "indice":
            if effect.get("mode") == "min":
                if self._number(draft.indice) is not None and self._number(value) is not None:
                    if float(self._number(value)) > float(self._number(draft.indice)):
                        draft.indice = str(value)
                        draft.trace.append(f"Indice {value}: {reason}")
                return
            draft.indice = str(value)
            draft.trace.append(f"Indice {value}: {reason}")
            return

        if param == "design_varilux":
            self._keep_ranked_recommendation(
                draft,
                attr="design_varilux",
                rank_attr="design_rank",
                value=value,
                ranks=VARILUX_RANKS,
                label="Design",
                reason=reason,
            )
            return

        if param == "traitement":
            self._keep_ranked_recommendation(
                draft,
                attr="traitement",
                rank_attr="traitement_rank",
                value=value,
                ranks=CRIZAL_RANKS,
                label="Traitement",
                reason=reason,
                allow_equal_override=bool(effect.get("override_same_rank", False)),
            )
            return

        if param == "couleur":
            draft.couleur = str(value)
            draft.trace.append(f"Couleur {value}: {reason}")
            return

        if param == "corridor_type":
            draft.corridor_type = str(value)
            draft.trace.append(f"Corridor {value}: {reason}")
            return

        if param == "extras":
            if value and str(value) not in draft.extras:
                draft.extras.append(str(value))
            return

    def _apply_engine_actions(self, state: dict[str, Any], draft: _DraftRecommendation) -> None:
        for action in sorted(self.config.engine.get("actions", []), key=lambda a: int(a.get("id", 999))):
            if self._should_skip_action(int(action.get("id", 0)), state, draft):
                continue
            for effect in action.get("effects", []):
                self._apply_effect(state, draft, effect)
        self._apply_priority_index_floor(state, draft)

    def _should_skip_action(self, action_id: int, state: dict[str, Any], draft: _DraftRecommendation) -> bool:
        for skip_rule in self.config.skip_rules:
            if action_id not in {int(x) for x in skip_rule.get("skip_questions", [])}:
                continue
            if self._match_condition(state, skip_rule.get("condition"), draft):
                draft.trace.append(f"Action {action_id} skippee par dynamic_rules.json")
                return True
        return False

    def _apply_user_rules(self, state: dict[str, Any], draft: _DraftRecommendation) -> None:
        for rule in sorted(self.config.rules, key=lambda r: r.get("priority", 99)):
            rid = rule.get("id", "DYN_?")
            condition = rule.get("condition", {})
            result = rule.get("result", {})
            if not self._match_condition(state, condition, draft):
                continue
            param = result.get("parameter")
            value = result.get("value")
            if param == "type_verre":
                draft.trace.append(f"{rid} ignoree: type_verre blinde")
                continue
            if param == "indice" and self._match_condition(state, {"field": "Q2_montage", "operator": "==", "value": "perce"}, draft):
                if self._number(value) is None or self._number(value) < 1.60:
                    draft.trace.append(f"{rid} ignoree: indice<1.60 interdit si perce")
                    continue
            effect_param = "design_varilux" if param == "design" else param
            self._apply_effect(state, draft, {"param": effect_param, "value": value, "condition": {}, "ignore_design_lock": True, "reason": rid})
            draft.trace.append(f"{rid} appliquee: {param}={value}")

    def decide(self, state: dict[str, Any]) -> Recommendation:
        self._reload_config()
        override = self._check_corrections(state)
        if override is not None:
            return override

        trace: list[str] = []
        type_verre = self._resolve_type(state, trace)
        defaults = self.config.engine.get("defaults", {})
        draft = _DraftRecommendation(
            type_verre=type_verre,
            indice=self._resolve_index(state, trace),
            design_varilux=None,
            traitement=str(defaults.get("traitement", "Crizal Sapphire HR")),
            couleur=str(defaults.get("couleur", "Blanc")),
            trace=trace,
        )

        self._apply_priority_index_floor(state, draft)
        self._apply_priority_design(state, draft)
        self._apply_engine_actions(state, draft)
        self._apply_user_rules(state, draft)

        # Regle projet: le design affiche est le type final recommande.
        # L'age determine la famille de depart; la sortie expose une seule valeur metier.
        final_type = draft.design_varilux or draft.type_verre

        return Recommendation(
            type_verre=final_type,
            indice=draft.indice,
            design_varilux=final_type,
            traitement=draft.traitement,
            couleur=draft.couleur,
            corridor_type=draft.corridor_type,
            extras=draft.extras,
            trace=draft.trace,
        )
