"""Onglet Streamlit : Editeur de Regles dynamiques."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

DYN_PATH = Path(__file__).resolve().parents[2] / "core" / "dynamic_rules.json"

CONDITION_FIELDS = ["Q1_age", "Q2_montage", "Q3_correction_total", "Q18_preference_verre",
                    "Q6_sante_oculaire", "Q4_usage_ordinateur", "Q22_style_vie_actif"]
OPERATORS = ["==", "<", ">", "<=", ">=", "in"]
RESULT_PARAMS = ["type_verre", "indice", "traitement", "couleur"]


def _load() -> dict:
    if DYN_PATH.exists():
        return json.loads(DYN_PATH.read_text(encoding="utf-8"))
    return {"$schema_version": "1.0", "rules": []}


def _save(data: dict) -> None:
    DYN_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _next_id(rules: list) -> str:
    nums = []
    for r in rules:
        rid = r.get("id", "")
        if rid.startswith("DYN_"):
            try:
                nums.append(int(rid.split("_")[1]))
            except (IndexError, ValueError):
                pass
    return f"DYN_{(max(nums) + 1) if nums else 1:03d}"


def _coerce_value(raw: str, op: str):
    raw = raw.strip()
    if op == "in":
        return [v.strip() for v in raw.split(",") if v.strip()]
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def render() -> None:
    st.header("Editeur de Regles Dynamiques")
    st.caption("Ajoute des regles a appliquer apres le moteur. Blindage : type_verre + indice/perce non modifiables.")

    data = _load()
    rules = data.get("rules", [])

    with st.form("add_rule"):
        st.subheader("Nouvelle regle")
        c1, c2, c3 = st.columns(3)
        with c1:
            cond_field = st.selectbox("Condition - champ", CONDITION_FIELDS)
        with c2:
            cond_op = st.selectbox("Operateur", OPERATORS)
        with c3:
            cond_val = st.text_input("Valeur (in: separe par virgules)")

        c4, c5 = st.columns(2)
        with c4:
            res_param = st.selectbox("Resultat - parametre", RESULT_PARAMS)
        with c5:
            res_val = st.text_input("Resultat - valeur")

        priority = st.number_input("Priorite (1=max, 10=min)", 1, 10, 5)
        note = st.text_area("Note")
        submit = st.form_submit_button("Ajouter la regle")

        if submit:
            if not res_val.strip():
                st.error("Resultat - valeur requise")
            else:
                new_rule = {
                    "id": _next_id(rules),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "priority": int(priority),
                    "note": note,
                    "condition": {
                        "field": cond_field,
                        "operator": cond_op,
                        "value": _coerce_value(cond_val, cond_op),
                    },
                    "result": {
                        "parameter": res_param,
                        "value": res_val.strip(),
                    },
                }
                rules.append(new_rule)
                data["rules"] = rules
                _save(data)
                st.success(f"Regle {new_rule['id']} ajoutee")

    st.divider()
    st.subheader(f"Regles existantes ({len(rules)})")
    for r in sorted(rules, key=lambda x: x.get("priority", 99)):
        with st.expander(f"{r['id']} - prio {r.get('priority')} - {r.get('note', '')[:60]}"):
            st.json(r)
            if st.button(f"Supprimer {r['id']}", key=f"del_{r['id']}"):
                data["rules"] = [x for x in rules if x["id"] != r["id"]]
                _save(data)
                st.rerun()
