"""Onglet Streamlit : Evaluation Expert - mode chatbot conversationnel."""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from core.corrections_store import CorrectionsStore
from core.decision_engine import DecisionEngine
from core.state_machine import DYNAMIC_RULES_PATH, condition_matches

ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = DYNAMIC_RULES_PATH


def _load_question_flow() -> list[dict]:
    """Construit l'ordre des questions a poser depuis dynamic_rules.json."""
    with open(RULES_PATH, "r", encoding="utf-8") as f:
        rules = json.load(f)
    flow: list[dict] = []
    for q in sorted(rules.get("questions", []), key=lambda item: int(item["id"])):
        qtype = q.get("type", "choice")
        options = [opt["value"] for opt in q.get("options", [])]
        flow.append({
            "action_id": int(q["id"]),
            "state_key": q["state_key"],
            "type": {"number": "int", "boolean": "bool", "choice": "enum"}.get(qtype, qtype),
            "options": options,
            "label": q.get("label", q["state_key"]),
            "skip_if": q.get("skip_if"),
            "min": q.get("min"),
            "max": q.get("max"),
        })
    return flow


def _should_skip(q: dict, state: dict) -> bool:
    skip_if = q.get("skip_if")
    if not skip_if:
        return False
    for field, operators in skip_if.items():
        for op, value in operators.items():
            mapped = {"gt": ">", "lt": "<", "gte": ">=", "lte": "<=", "neq": "!="}.get(op, op)
            if condition_matches(state, {"field": field, "operator": mapped, "value": value}):
                return True
    return False


def _coerce_answer(q: dict, raw):
    t = q["type"]
    if t == "int":
        return int(raw)
    if t == "float":
        return float(raw)
    if t == "bool":
        if isinstance(raw, bool):
            return raw
        return str(raw).lower() in ("oui", "true", "1", "yes")
    return raw


def _reset_session():
    for k in ("chat_history", "chat_idx", "chat_state", "chat_done",
              "chat_rec", "chat_correcting", "chat_error"):
        st.session_state.pop(k, None)


def _ensure_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Bonjour ! Je vais vous poser quelques questions pour determiner la meilleure recommandation."}
        ]
        st.session_state.chat_idx = 0
        st.session_state.chat_state = {}
        st.session_state.chat_done = False
        st.session_state.chat_rec = None
        st.session_state.chat_correcting = False
        st.session_state.chat_error = None


def render() -> None:
    st.header("Evaluation Expert (chatbot)")
    st.caption("Repondez aux questions une par une. A la fin : Valider ou Corriger.")

    if st.button("Recommencer la conversation"):
        _reset_session()
        st.rerun()

    flow = _load_question_flow()
    store = CorrectionsStore()
    engine = DecisionEngine(corrections_store=store)

    _ensure_state()

    # Historique du chat
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Phase 1 : poser les questions
    if not st.session_state.chat_done:
        idx = st.session_state.chat_idx
        # Sauter les questions non pertinentes
        while idx < len(flow) and _should_skip(flow[idx], st.session_state.chat_state):
            idx += 1
        st.session_state.chat_idx = idx

        if idx >= len(flow):
            st.session_state.chat_done = True
            try:
                rec = engine.decide(st.session_state.chat_state)
                st.session_state.chat_rec = rec.to_dict()
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": "Merci ! Voici la recommandation calculee :",
                })
            except Exception as e:
                st.session_state.chat_error = str(e)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"Erreur lors du calcul : `{e}`",
                })
            st.rerun()
        else:
            q = flow[idx]
            with st.chat_message("assistant"):
                st.markdown(f"**Q{q['action_id']}** - {q['label']}")

                with st.form(f"q_form_{idx}", clear_on_submit=True):
                    if q["type"] == "int":
                        # Pas de max sur l'age
                        answer = st.number_input(
                            "Votre reponse",
                            min_value=int(q.get("min") or 1),
                            value=int(q.get("min") or 30),
                            step=1,
                            key=f"in_{idx}",
                        )
                    elif q["type"] == "float":
                        answer = st.number_input(
                            "Votre reponse",
                            min_value=float(q.get("min") or 0.0),
                            max_value=float(q.get("max") or 10.0),
                            value=float(q.get("min") or 0.0),
                            step=0.25,
                            key=f"in_{idx}",
                        )
                    elif q["type"] == "bool":
                        answer = st.radio(
                            "Votre reponse", ["oui", "non"],
                            horizontal=True, key=f"in_{idx}",
                        )
                    else:
                        opts = q["options"] or ["RAS"]
                        answer = st.selectbox("Votre reponse", opts, key=f"in_{idx}")

                    submitted = st.form_submit_button("Valider la reponse")

                if submitted:
                    val = _coerce_answer(q, answer)
                    st.session_state.chat_state[q["state_key"]] = val
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"**Q{q['action_id']}** - {q['label']}",
                    })
                    st.session_state.chat_history.append({
                        "role": "user", "content": f"{val}",
                    })
                    st.session_state.chat_idx = idx + 1
                    st.rerun()

    # Phase 2 : recommandation + Valider/Corriger
    if st.session_state.chat_done and st.session_state.chat_rec:
        rec_dict = st.session_state.chat_rec
        with st.chat_message("assistant"):
            st.json(rec_dict)

        st.divider()
        st.subheader("Validation expert")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Valider (CORRECT)", use_container_width=True, type="primary"):
                cid = store.add(st.session_state.chat_state, rec_dict, rec_dict, "CORRECT", None)
                st.success(f"Enregistre comme CORRECT (id={cid})")
                _reset_session()
                st.rerun()
        with c2:
            if st.button("Corriger (OVERRIDE)", use_container_width=True):
                st.session_state.chat_correcting = True

        if st.session_state.chat_correcting:
            with st.form("override_form"):
                t = st.text_input("type_verre", rec_dict.get("type_verre", ""))
                i = st.text_input("indice", rec_dict.get("indice", ""))
                tr = st.text_input("traitement", rec_dict.get("traitement", ""))
                co = st.text_input("couleur", rec_dict.get("couleur", ""))
                just = st.text_area("Justification expert", "")
                ok = st.form_submit_button("Enregistrer OVERRIDE")
                if ok:
                    expert_out = dict(rec_dict)
                    expert_out.update({
                        "type_verre": t, "indice": i,
                        "design_varilux": t,
                        "traitement": tr, "couleur": co,
                    })
                    cid = store.add(
                        st.session_state.chat_state, rec_dict, expert_out,
                        "OVERRIDE", just,
                    )
                    st.success(f"Enregistre comme OVERRIDE (id={cid})")
                    _reset_session()
                    st.rerun()

    st.divider()
    st.caption(f"Total corrections enregistrees : {store.count()}")
