from __future__ import annotations

import json
from copy import deepcopy

from core.decision_engine import CRIZAL_RANKS, VARILUX_RANKS, DecisionEngine, DynamicRulesConfig, keep_highest_ranked


def _state(**overrides):
    state = {
        "Q1_age": 50,
        "Q2_montage": "plastique",
        "Q3_correction_total": "225-400",
        "Q3_add": 0.0,
        "Q3_diff_od_og": "<=200",
        "Q4_vision_laterale": "mixte",
        "Q4_conduite_nuit": False,
        "Q4_usage_ordinateur": "jamais",
        "Q4_adaptation_facile": False,
        "Q4_innovation": False,
        "Q10_solution_ia": False,
        "Q5_besoin_principal": "transparence",
        "Q6_sante_oculaire": "ras",
        "Q13_paire_soleil": "non",
        "Q7_reflets_genants": False,
        "Q7_mal_voir_soleil": False,
        "Q7_gene_lumiere": "faible",
        "Q7_environnement": "interieur",
        "Q18_preference_verre": "aucune",
        "Q19_antecedents_familiaux": False,
        "Q20_operation_yeux": False,
        "Q21_fatigue_oculaire": False,
        "Q22_style_vie_actif": False,
    }
    state.update(overrides)
    return state


def test_dynamic_rules_schema_has_22_actions():
    data = json.load(open("core/dynamic_rules.json", encoding="utf-8"))
    cfg = DynamicRulesConfig.model_validate(data)
    assert cfg.version == "2.0"
    assert cfg.metadata["total_rules"] == 22
    assert {q["id"] for q in cfg.questions} == set(range(1, 23))


def test_engine_reflects_dynamic_json_change_without_restart(tmp_path):
    data = json.load(open("core/dynamic_rules.json", encoding="utf-8"))
    modified = deepcopy(data)
    for rule in modified["engine"]["index_rules"]:
        if rule["condition"].get("value") == "225-400":
            rule["value"] = "1.60"
            rule["reason"] = "test runtime change"
            break
    p = tmp_path / "dynamic_rules.json"
    p.write_text(json.dumps(modified), encoding="utf-8")

    engine = DecisionEngine(dynamic_rules_path=p)
    assert engine.decide(_state()).indice == "1.60"

    for rule in modified["engine"]["index_rules"]:
        if rule["condition"].get("value") == "225-400":
            rule["value"] = "1.56"
            break
    p.write_text(json.dumps(modified), encoding="utf-8")

    assert engine.decide(_state()).indice == "1.56"


def test_user_rule_can_change_non_blinded_behavior(tmp_path):
    data = json.load(open("core/dynamic_rules.json", encoding="utf-8"))
    data["rules"] = [
        {
            "id": "DYN_COLOR",
            "priority": 1,
            "condition": {"field": "Q1_age", "operator": ">=", "value": 18},
            "result": {"parameter": "couleur", "value": "Test Blue"},
        }
    ]
    p = tmp_path / "dynamic_rules.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    rec = DecisionEngine(dynamic_rules_path=p).decide(_state())
    assert rec.couleur == "Test Blue"


def test_blinded_dynamic_rule_does_not_change_type_or_perce_index(tmp_path):
    data = json.load(open("core/dynamic_rules.json", encoding="utf-8"))
    data["rules"] = [
        {
            "id": "DYN_TYPE",
            "priority": 1,
            "condition": {"field": "Q1_age", "operator": ">=", "value": 18},
            "result": {"parameter": "type_verre", "value": "Illegal"},
        },
        {
            "id": "DYN_INDEX",
            "priority": 2,
            "condition": {"field": "Q2_montage", "operator": "==", "value": "perce"},
            "result": {"parameter": "indice", "value": "1.50"},
        },
    ]
    p = tmp_path / "dynamic_rules.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    rec = DecisionEngine(dynamic_rules_path=p).decide(_state(Q1_age=25, Q2_montage="perce", Q3_correction_total="<200"))
    assert rec.type_verre == "Simple foyer"
    assert rec.indice == "1.60"


def test_ranked_choice_keeps_highest_varilux():
    selected, rank, changed = keep_highest_ranked(
        "Varilux Physio 3.0",
        "Varilux Liberty 3.0",
        VARILUX_RANKS,
        current_rank=3,
    )
    assert selected == "Varilux Physio 3.0"
    assert rank == 3
    assert changed is False

    selected, rank, changed = keep_highest_ranked(
        "Varilux Physio 3.0",
        "Varilux X Design",
        VARILUX_RANKS,
        current_rank=3,
    )
    assert selected == "Varilux X Design"
    assert rank == 5
    assert changed is True


def test_ranked_choice_keeps_first_crizal_at_same_rank():
    selected, rank, changed = keep_highest_ranked(
        "Crizal Sapphire HR",
        "Crizal Prevencia",
        CRIZAL_RANKS,
        current_rank=3,
    )
    assert selected == "Crizal Sapphire HR"
    assert rank == 3
    assert changed is False
