from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from core.corrections_store import HISTORY_CORRECTED, HISTORY_PENDING, HISTORY_VALIDATED, CorrectionsStore
from core.decision_engine import DecisionEngine, Recommendation
from core.state_machine import DYNAMIC_RULES_PATH, get_next_action, should_skip_action
from rag import rag_builder

ROOT = Path(__file__).resolve().parent

PALETTE = {
    "bg": "#F4F8FF",
    "panel": "#FFFFFF",
    "panel2": "#EAF4FF",
    "accent": "#0EA5E9",
    "white": "#0F172A",
    "muted": "#64748B",
    "success": "#10B981",
    "warning": "#FFD166",
    "danger": "#EF476F",
}

STATUS_LABELS = {
    HISTORY_PENDING: "en attente de correction",
    HISTORY_CORRECTED: "corrigée",
    HISTORY_VALIDATED: "validée",
}


def _rules_cache_token() -> int:
    return DYNAMIC_RULES_PATH.stat().st_mtime_ns


@st.cache_data(show_spinner=False)
def _load_rules(rules_mtime_ns: int) -> dict[str, Any]:
    with open(DYNAMIC_RULES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _question_flow(rules: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(rules.get("questions", []), key=lambda q: int(q.get("id", 999)))


def _question_options(question: dict[str, Any]) -> list[dict[str, str]]:
    return list(question.get("options", []))


def _option_label(question: dict[str, Any], value: Any) -> str:
    for opt in _question_options(question):
        if opt.get("value") == value:
            return str(opt.get("label", value))
    return str(value)


def _default_answer(question: dict[str, Any]) -> Any:
    qtype = question.get("type")
    if qtype == "number":
        return float(question.get("min", 0.0)) if question.get("step") == 0.25 else int(question.get("min", 1))
    if qtype == "boolean":
        return False
    opts = _question_options(question)
    return opts[0]["value"] if opts else None


def _init_session() -> None:
    defaults = {
        "client_name": "Jean D.",
        "expert_name": "Expert",
        "authenticated": False,
        "consultation_state": {},
        "current_action": 1,
        "current_recommendation_id": None,
        "consultation_done": False,
        "last_feedback": None,
        "history": [],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _reset_consultation() -> None:
    st.session_state.consultation_state = {}
    st.session_state.current_action = 1
    st.session_state.current_recommendation_id = None
    st.session_state.consultation_done = False
    st.session_state.last_feedback = None


def _streamlit_secret(name: str) -> str | None:
    try:
        value = st.secrets.get(name)
    except Exception:
        return None
    return str(value) if value else None


def _app_password() -> str:
    return os.getenv("OPTI_RECO_PASSWORD") or _streamlit_secret("OPTI_RECO_PASSWORD") or "expert123"


def _database_url() -> str | None:
    return os.getenv("DATABASE_URL") or _streamlit_secret("DATABASE_URL")


@st.cache_resource(show_spinner=False)
def _get_store(database_url: str | None) -> CorrectionsStore:
    return CorrectionsStore(database_url=database_url)


@st.cache_resource(show_spinner=False)
def _get_engine(rules_mtime_ns: int) -> DecisionEngine:
    return DecisionEngine(reload_on_decide=False)


@st.cache_data(show_spinner=False, ttl=3600)
def _get_rag_chunks(recommendation_payload: str, n: int) -> list[str]:
    return rag_builder.get_justification(json.loads(recommendation_payload), n=n)


@st.cache_data(show_spinner=False, ttl=15)
def _list_history_cached(database_url: str, _store: CorrectionsStore, limit: int, status: str | None) -> list[dict[str, Any]]:
    return _store.list_history(limit=limit, status=status)


def _require_login() -> None:
    if st.session_state.authenticated:
        return

    st.markdown(
        """
        <div class="opti-header">
            <div>
                <div class="brand">OptiReco Pro</div>
                <div style="color:#64748B;font-weight:600;">Acces expert protege</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.form("login_form"):
        expert_name = st.text_input("Nom expert", st.session_state.expert_name)
        password = st.text_input("Mot de passe", type="password")
        submitted = st.form_submit_button("Se connecter", type="primary")
    if submitted:
        if password == _app_password():
            st.session_state.authenticated = True
            st.session_state.expert_name = expert_name.strip() or "Expert"
            st.rerun()
        st.error("Mot de passe incorrect.")
    st.stop()


def _completed_actions(flow: list[dict[str, Any]], state: dict[str, Any]) -> set[int]:
    completed: set[int] = set()
    for q in flow:
        if q.get("state_key") in state:
            completed.add(int(q["id"]))
    return completed


def _effective_total(flow: list[dict[str, Any]], state: dict[str, Any], rules: dict[str, Any]) -> int:
    skipped = {int(q["id"]) for q in flow if should_skip_action(int(q["id"]), state, rules)}
    return max(1, len(flow) - len(skipped))


def _confidence(done: int, total: int, rec: Recommendation | None) -> int:
    base = int(min(95, 45 + (done / max(1, total)) * 45))
    if rec and rec.traitement and rec.indice:
        base += 4
    return min(base, 98)


def _try_live_reco(state: dict[str, Any], engine: DecisionEngine) -> Recommendation | None:
    if "Q1_age" not in state:
        return None
    try:
        return engine.decide(state)
    except Exception:
        return None


def _feedback_for(rules: dict[str, Any], state_key: str, value: Any) -> dict[str, str] | None:
    feedback = rules.get("feedback_messages", {}).get(state_key, {})
    return feedback.get(str(value))


def _html_escape(text: Any) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _pretty_json(data: Any) -> str:
    return json.dumps(data or {}, ensure_ascii=False, indent=2, default=str)


def _status_label(status: str) -> str:
    return STATUS_LABELS.get(status, status)


def _status_class(status: str) -> str:
    return {
        HISTORY_PENDING: "status-pending",
        HISTORY_CORRECTED: "status-corrected",
        HISTORY_VALIDATED: "status-validated",
    }.get(status, "status-pending")


def _ensure_history_record(store: CorrectionsStore, state: dict[str, Any], rec: Recommendation) -> int:
    record_id = st.session_state.current_recommendation_id
    if record_id is not None and store.get_history(int(record_id)) is not None:
        return int(record_id)
    record_id = store.create_history(state, rec.to_dict(), st.session_state.expert_name)
    _list_history_cached.clear()
    st.session_state.current_recommendation_id = record_id
    return int(record_id)


def _pdf_bytes(rec: Recommendation, state: dict[str, Any]) -> bytes:
    lines = [
        "OptiReco Pro - Recommandation",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Age: {state.get('Q1_age', '-')}",
        f"Type: {rec.type_verre}",
        f"Indice: {rec.indice}",
        f"Design: {rec.type_verre}",
        f"Traitement: {rec.traitement}",
        f"Couleur: {rec.couleur}",
        f"Corridor: {rec.corridor_type or '-'}",
    ]
    escaped = []
    for line in lines:
        clean = line.encode("latin-1", "replace").decode("latin-1")
        clean = clean.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        escaped.append(f"({clean}) Tj T*")
    stream = "BT /F1 12 Tf 72 760 Td 16 TL " + " ".join(escaped) + " ET"
    objects = [
        "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj",
        "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj",
        "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj",
        "4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj",
        f"5 0 obj << /Length {len(stream.encode('latin-1'))} >> stream\n{stream}\nendstream endobj",
    ]
    pdf = "%PDF-1.4\n"
    offsets = []
    for obj in objects:
        offsets.append(len(pdf.encode("latin-1")))
        pdf += obj + "\n"
    startxref = len(pdf.encode("latin-1"))
    pdf += "xref\n0 6\n0000000000 65535 f \n"
    for offset in offsets:
        pdf += f"{offset:010d} 00000 n \n"
    pdf += f"trailer << /Size 6 /Root 1 0 R >>\nstartxref\n{startxref}\n%%EOF"
    return pdf.encode("latin-1")


def _inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
        .stApp {{ background: linear-gradient(180deg, #F8FBFF 0%, {PALETTE['bg']} 100%); color: {PALETTE['white']}; }}
        .block-container {{ max-width: 1360px; padding: 1.15rem 2rem 2rem; }}
        [data-testid="stHeader"] {{ background: transparent; }}
        [data-testid="stToolbar"], #MainMenu, footer {{ visibility: hidden; }}
        [data-testid="stSidebar"] {{ background: #FFFFFF; border-right: 1px solid #D8E6F5; }}
        [data-testid="stSidebar"] * {{ color: {PALETTE['white']}; }}
        div[data-testid="column"] {{ min-width: 0; }}
        .opti-header {{
            display:flex; align-items:center; justify-content:space-between; gap:1rem;
            padding:1rem 1.2rem; border:1px solid #CFE4F7;
            background: linear-gradient(135deg, #FFFFFF, #EAF6FF);
            border-radius:18px; box-shadow:0 14px 36px rgba(15, 23, 42, .08); margin-bottom:1rem;
        }}
        .brand {{ font-size:1.35rem; font-weight:800; color:#075985; letter-spacing:0; }}
        .client-pill {{ border:1px solid #CFE4F7; background:#FFFFFF; border-radius:999px; padding:.45rem .75rem; color:{PALETTE['white']}; font-weight:700; }}
        .panel {{
            border:1px solid #D8E6F5; background:{PALETTE['panel']};
            border-radius:18px; padding:1rem; box-shadow:0 12px 30px rgba(15, 23, 42, .07);
        }}
        .progress-panel {{ max-height: calc(100vh - 225px); overflow:auto; }}
        .preview-panel {{ position: sticky; top: 1rem; }}
        .panel h3 {{ margin:0 0 .75rem; color:#0369A1; font-size:.78rem; letter-spacing:.12em; text-transform:uppercase; }}
        .step {{ display:grid; grid-template-columns:2.45rem minmax(0,1fr); align-items:start; gap:.5rem; padding:.5rem .55rem; border-radius:12px; color:{PALETTE['muted']}; font-size:.86rem; line-height:1.25; }}
        .step span:last-child {{ min-width:0; overflow-wrap:anywhere; }}
        .step.done {{ color:#047857; background:#ECFDF5; }}
        .step.active {{ color:#075985; background:#E0F2FE; border:1px solid #BAE6FD; font-weight:700; }}
        .step-id {{ display:inline-flex; justify-content:center; align-items:center; min-width:2.25rem; height:1.45rem; border-radius:999px; background:#F1F5F9; color:#0369A1; font-weight:800; font-size:.72rem; }}
        .step.done .step-id {{ background:#D1FAE5; color:#047857; }}
        .step.active .step-id {{ background:#0EA5E9; color:#FFFFFF; }}
        .progress-shell {{ height:.65rem; background:#E2E8F0; border-radius:999px; overflow:hidden; border:1px solid #D8E6F5; }}
        .progress-fill {{ height:100%; background:linear-gradient(90deg,{PALETTE['accent']},{PALETTE['success']}); border-radius:999px; transition:width .35s ease; }}
        div[data-testid="stForm"] {{ animation:fadeIn .25s ease; border:1px solid #CFE4F7; background:#FFFFFF; border-radius:20px; padding:1.25rem; box-shadow:0 12px 30px rgba(15, 23, 42, .08); }}
        .question-title {{ font-size:1.45rem; line-height:1.25; font-weight:800; margin-bottom:.35rem; color:#0F172A; }}
        .question-meta {{ color:{PALETTE['muted']}; margin-bottom:1rem; font-weight:600; }}
        .metric-row {{ display:flex; justify-content:space-between; gap:1rem; padding:.7rem 0; border-bottom:1px solid #E2E8F0; font-size:.92rem; }}
        .metric-row span:first-child {{ color:{PALETTE['muted']}; }}
        .metric-row span:last-child {{ color:{PALETTE['white']}; font-weight:800; text-align:right; overflow-wrap:anywhere; }}
        .feedback {{ border-radius:14px; padding:.85rem 1rem; margin-top:1rem; border:1px solid #D8E6F5; }}
        .feedback.success {{ background:#ECFDF5; color:#047857; }}
        .feedback.warning {{ background:#FFFBEB; color:#92400E; }}
        .result-card {{ animation:pulseIn .42s ease; border:1px solid #A7F3D0; background:linear-gradient(135deg, #F0FDF4, #EFF6FF); border-radius:22px; padding:1.2rem; box-shadow:0 14px 36px rgba(15, 23, 42, .08); }}
        .result-title {{ margin:.9rem 0 .7rem; color:#0F172A; font-size:1.4rem; font-weight:800; }}
        .result-grid {{ display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:.65rem; }}
        .result-item {{ background:#FFFFFF; border:1px solid #D8E6F5; border-radius:14px; padding:.75rem .85rem; }}
        .result-label {{ color:{PALETTE['muted']}; font-size:.78rem; font-weight:700; text-transform:uppercase; letter-spacing:.08em; }}
        .result-value {{ color:#0F172A; font-weight:800; margin-top:.25rem; overflow-wrap:anywhere; }}
        .tech-note {{ margin-top:.85rem; padding:.85rem 1rem; border-radius:14px; background:#E0F2FE; color:#075985; border:1px solid #BAE6FD; }}
        .badge {{ display:inline-flex; align-items:center; gap:.45rem; padding:.42rem .75rem; border-radius:999px; font-weight:800; color:#08111A; }}
        .status-pill {{ display:inline-flex; align-items:center; border-radius:999px; padding:.32rem .68rem; font-size:.78rem; font-weight:800; }}
        .status-pending {{ color:#92400E; background:#FEF3C7; border:1px solid #FDE68A; }}
        .status-corrected {{ color:#075985; background:#E0F2FE; border:1px solid #BAE6FD; }}
        .status-validated {{ color:#047857; background:#D1FAE5; border:1px solid #A7F3D0; }}
        .history-meta {{ display:flex; flex-wrap:wrap; gap:.55rem; align-items:center; margin:.35rem 0 1rem; color:#64748B; font-weight:700; }}
        .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {{
            background:{PALETTE['accent']}; color:#FFFFFF; border:none; border-radius:12px; font-weight:800; padding:.62rem 1rem;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {{ background:#0284C7; color:#FFFFFF; }}
        div[data-baseweb="select"] > div, input {{ border-radius:12px !important; background:#F8FBFF !important; border-color:#CFE4F7 !important; color:#0F172A !important; }}
        div[data-baseweb="select"] span, div[data-baseweb="select"] svg {{ color:#0F172A !important; fill:#0F172A !important; }}
        div[data-baseweb="popover"] ul {{ background:#FFFFFF !important; border:1px solid #D8E6F5 !important; border-radius:12px !important; }}
        div[data-baseweb="popover"] li {{ color:#0F172A !important; }}
        div[data-baseweb="popover"] li:hover {{ background:#E0F2FE !important; }}
        label, p, span {{ letter-spacing:0; }}
        @keyframes fadeIn {{ from {{ opacity:0; transform:translateY(8px); }} to {{ opacity:1; transform:translateY(0); }} }}
        @keyframes pulseIn {{ 0% {{ opacity:0; transform:scale(.98); }} 100% {{ opacity:1; transform:scale(1); }} }}
        @media (max-width: 1100px) {{ .step {{ grid-template-columns:2.15rem minmax(0,1fr); font-size:.82rem; }} .step-id {{ min-width:2rem; }} }}
        @media (max-width: 900px) {{ .progress-panel {{ max-height:none; }} .preview-panel {{ position:static; }} .opti-header {{ flex-direction:column; align-items:flex-start; }} .result-grid {{ grid-template-columns:1fr; }} }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_progress(flow: list[dict[str, Any]], state: dict[str, Any], current: int, rules: dict[str, Any]) -> tuple[int, int]:
    completed = _completed_actions(flow, state)
    total = _effective_total(flow, state, rules)
    done = len([action for action in completed if not should_skip_action(action, state, rules)])
    pct = int(min(100, (done / max(1, total)) * 100))
    steps = []
    for q in flow:
        action = int(q["id"])
        if should_skip_action(action, state, rules):
            continue
        cls = "done" if action in completed else "active" if action == current else ""
        label = _html_escape(q.get("label", f"Q{action}"))
        steps.append(f'<div class="step {cls}"><span class="step-id">Q{action}</span><span>{label}</span></div>')
    st.markdown(
        f"""
        <div class="panel progress-panel">
            <h3>Progression</h3>
            {''.join(steps)}
            <div style="height:1rem"></div>
            <div class="progress-shell"><div class="progress-fill" style="width:{pct}%"></div></div>
            <div style="margin-top:.55rem;color:{PALETTE['muted']};font-weight:700;">{pct}% complete</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return done, total


def _render_input(question: dict[str, Any], state: dict[str, Any]) -> dict[str, Any] | None:
    with st.form(f"question_{question['id']}"):
        st.markdown(f'<div class="question-title">{_html_escape(question["label"])}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="question-meta">Question {question["id"]} / 22</div>', unsafe_allow_html=True)
        values: dict[str, Any] = {}
        values[question["state_key"]] = _input_widget(question, state)
        for sub in question.get("sub_questions", []) or []:
            values[sub["state_key"]] = _input_widget(sub, state)
        submitted = st.form_submit_button("Valider la reponse")
    return values if submitted else None


def _input_widget(question: dict[str, Any], state: dict[str, Any]) -> Any:
    key = question["state_key"]
    qtype = question.get("type")
    current = state.get(key, _default_answer(question))
    if qtype == "number":
        raw_step = question.get("step", 1)
        is_decimal = isinstance(raw_step, float) and not raw_step.is_integer()
        if is_decimal:
            return st.number_input(
                question.get("label", key),
                min_value=float(question.get("min", 0.0)),
                value=float(current),
                step=float(raw_step),
                key=f"input_{key}",
            )
        return st.number_input(
            question.get("label", key),
            min_value=int(question.get("min", 0)),
            value=int(current),
            step=int(raw_step),
            key=f"input_{key}",
        )
    if qtype == "boolean":
        options = [False, True]
        labels = {False: "Non", True: "Oui"}
        idx = 1 if current is True else 0
        return st.radio(question.get("label", key), options, index=idx, format_func=lambda v: labels[v], horizontal=True, key=f"input_{key}")
    opts = _question_options(question)
    values = [opt["value"] for opt in opts]
    index = values.index(current) if current in values else 0
    return st.selectbox(question.get("label", key), values, index=index, format_func=lambda v: _option_label(question, v), key=f"input_{key}")


def _render_live_preview(rec: Recommendation | None, done: int, total: int) -> None:
    rows = {
        "Type": rec.type_verre if rec else "-",
        "Indice": rec.indice if rec else "-",
        "Design": rec.type_verre if rec else "-",
        "Traitement": rec.traitement if rec else "-",
        "Couleur": rec.couleur if rec else "-",
        "Corridor": rec.corridor_type if rec and rec.corridor_type else "-",
    }
    rows_html = "".join(
        f'<div class="metric-row"><span>{label}</span><span>{_html_escape(value)}</span></div>'
        for label, value in rows.items()
    )
    conf = _confidence(done, total, rec)
    st.markdown(
        f"""
        <div class="panel preview-panel">
            <h3>Apercu live</h3>
            {rows_html}
            <div style="height:1rem"></div>
            <div style="color:{PALETTE['muted']};font-weight:700;">Score confiance</div>
            <div class="progress-shell"><div class="progress-fill" style="width:{conf}%"></div></div>
            <div style="margin-top:.5rem;color:{PALETTE['white']};font-weight:800;">{conf}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_feedback(feedback: dict[str, str] | None) -> None:
    if not feedback:
        return
    level = feedback.get("level", "success")
    msg = feedback.get("message", "")
    st.markdown(f'<div class="feedback {level}">{_html_escape(msg)}</div>', unsafe_allow_html=True)


def _render_expert_validation(
    store: CorrectionsStore,
    record_id: int,
    rec: Recommendation,
    state: dict[str, Any],
) -> None:
    record = store.get_history(record_id) or {}
    status = str(record.get("status", HISTORY_PENDING))
    expert_default = record.get("recommendation_expert_data") or rec.to_dict()

    st.markdown("### Validation expert")
    st.markdown(
        f'<span class="status-pill {_status_class(status)}">{_html_escape(_status_label(status))}</span>',
        unsafe_allow_html=True,
    )
    st.caption(f"Identifiant recommandation : #{record_id}")

    with st.form(f"expert_validation_{record_id}"):
        expert_name = st.text_input("Nom expert", st.session_state.expert_name)
        type_value = st.text_input("Type / Design", str(expert_default.get("type_verre", rec.type_verre)))
        indice = st.text_input("Indice", str(expert_default.get("indice", rec.indice)))
        traitement = st.text_input("Traitement", str(expert_default.get("traitement", rec.traitement)))
        couleur = st.text_input("Couleur", str(expert_default.get("couleur", rec.couleur)))
        notes = st.text_area("Commentaire / justification expert", str(record.get("notes") or ""), height=110)

        st.caption("La validation garde la recommandation automatique. La correction enregistre les champs modifies par l'expert.")
        c1, c2 = st.columns(2)
        validated = c1.form_submit_button("Valider sans correction", use_container_width=True)
        corrected = c2.form_submit_button("Enregistrer la correction", type="primary", use_container_width=True)

    if not (validated or corrected):
        return

    st.session_state.expert_name = expert_name.strip() or "Expert"
    if validated:
        expert_output = rec.to_dict()
        store.update_history(record_id, expert_output, HISTORY_VALIDATED, st.session_state.expert_name, notes)
        store.add(state, rec.to_dict(), expert_output, "CORRECT", notes, sync_history=False)
        _list_history_cached.clear()
        st.success(f"Recommandation #{record_id} validee par l'expert.")
        st.rerun()

    expert_output = dict(rec.to_dict())
    expert_output.update({
        "type_verre": type_value,
        "design_varilux": type_value,
        "indice": indice,
        "traitement": traitement,
        "couleur": couleur,
        "corridor_type": rec.corridor_type,
    })
    store.update_history(record_id, expert_output, HISTORY_CORRECTED, st.session_state.expert_name, notes)
    store.add(state, rec.to_dict(), expert_output, "OVERRIDE", notes, sync_history=False)
    _list_history_cached.clear()
    st.success(f"Correction expert enregistree pour la recommandation #{record_id}.")
    st.rerun()


def _render_history(store: CorrectionsStore, database_url: str | None) -> None:
    st.subheader("Historique")
    st.caption("Toutes les recommandations generees sont conservees, avec leur statut de validation expert.")

    options = {
        "Tous les statuts": None,
        "En attente de correction": HISTORY_PENDING,
        "Corrigée": HISTORY_CORRECTED,
        "Validée": HISTORY_VALIDATED,
    }
    selected = st.selectbox("Filtrer", list(options.keys()), label_visibility="collapsed")
    rows = _list_history_cached(database_url or "sqlite", store, 200, options[selected])
    if not rows:
        st.info("Aucune recommandation enregistree pour le moment.")
        return

    for row in rows:
        created = str(row.get("created_at", "")).replace("T", " ")[:19]
        updated = str(row.get("updated_at", "")).replace("T", " ")[:19]
        status = str(row.get("status", HISTORY_PENDING))
        title = f"#{row['id']} - {created} - {_status_label(status)}"
        with st.expander(title, expanded=False):
            st.markdown(
                f"""
                <div class="history-meta">
                    <span>Identifiant: #{row['id']}</span>
                    <span>Cree: {created}</span>
                    <span>Mis a jour: {updated}</span>
                    <span>Expert: {_html_escape(row.get('expert_name') or '-')}</span>
                    <span class="status-pill {_status_class(status)}">{_html_escape(_status_label(status))}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            input_col, auto_col, expert_col = st.columns(3)
            with input_col:
                st.markdown("**Donnees d'entree**")
                st.json(row.get("input_data") or {})
            with auto_col:
                st.markdown("**Recommandation automatique**")
                st.json(row.get("recommendation_auto_data") or {})
            with expert_col:
                st.markdown("**Recommandation expert**")
                if row.get("recommendation_expert_data"):
                    st.json(row["recommendation_expert_data"])
                else:
                    st.info("Aucune correction enregistree.")
            if row.get("notes"):
                st.markdown("**Commentaire expert**")
                st.write(row["notes"])


def _render_final(
    rec: Recommendation,
    state: dict[str, Any],
    rules: dict[str, Any],
    store: CorrectionsStore,
    record_id: int,
) -> None:
    badges = rules.get("product_badges", {})
    badge_key = rec.design_varilux or rec.type_verre
    badge = badges.get(badge_key, badges.get(rec.type_verre, {"color": PALETTE["accent"], "label": "Recommandation"}))
    tech = rules.get("rag_tech_messages", {}).get(badge_key, "")

    result_rows = {
        "Type": rec.type_verre,
        "Indice": rec.indice,
        "Design": rec.type_verre,
        "Traitement": rec.traitement,
        "Couleur": rec.couleur,
        "Corridor": rec.corridor_type or "-",
    }
    result_html = "".join(
        f'<div class="result-item"><div class="result-label">{label}</div><div class="result-value">{_html_escape(value)}</div></div>'
        for label, value in result_rows.items()
    )
    tech_html = f'<div class="tech-note">{_html_escape(tech)}</div>' if tech else ""
    st.markdown(
        f"""
        <div class="result-card">
            <span class="badge" style="background:{badge.get('color', PALETTE['accent'])};">{_html_escape(badge_key)} - {_html_escape(badge.get('label', ''))}</span>
            <div class="result-title">Recommandation finale</div>
            <div class="result-grid">{result_html}</div>
            {tech_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Pourquoi cette recommandation ?", expanded=False):
        for line in rec.trace:
            st.write(line)
        try:
            recommendation_payload = json.dumps(rec.to_dict(), sort_keys=True, ensure_ascii=False, default=str)
            chunks = _get_rag_chunks(recommendation_payload, n=2)
        except Exception:
            chunks = []
        if chunks:
            st.divider()
            st.write("Extraits RAG :")
            for chunk in chunks:
                st.write(chunk)
        elif not tech:
            st.write("Aucun extrait RAG disponible pour cette combinaison.")

    _render_expert_validation(store, record_id, rec, state)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Nouvelle consultation", use_container_width=True):
            st.session_state.history.append({
                "date": datetime.now().strftime("%H:%M"),
                "client": st.session_state.client_name,
                "rec": rec.to_dict(),
            })
            _reset_consultation()
            st.rerun()
    with c2:
        st.download_button("Exporter PDF", data=_pdf_bytes(rec, state), file_name="opti_reco.pdf", mime="application/pdf", use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="OptiReco Pro", page_icon="OR", layout="wide", initial_sidebar_state="collapsed")
    _inject_css()
    _init_session()
    _require_login()

    database_url = _database_url()
    rules_token = _rules_cache_token()
    store = _get_store(database_url)
    engine = _get_engine(rules_token)
    rules = _load_rules(rules_token)
    flow = _question_flow(rules)
    state = st.session_state.consultation_state
    current = int(st.session_state.current_action)

    header_left = '<div class="brand">OptiReco Pro</div><div style="color:#64748B;font-weight:600;">Recommandation verres optiques deterministe + RAG explicatif</div>'
    header_right = f'<div class="client-pill">Expert: {_html_escape(st.session_state.expert_name)}</div>'
    st.markdown(f'<div class="opti-header"><div>{header_left}</div><div>{header_right}</div></div>', unsafe_allow_html=True)

    tab_new, tab_history = st.tabs(["Nouvelle recommandation", "Historique"])

    with tab_new:
        top1, top2, top3 = st.columns([2.3, 1, 1])
        with top1:
            st.session_state.client_name = st.text_input("Client", st.session_state.client_name, label_visibility="collapsed")
        with top2:
            st.caption(f"Expert connecte : {st.session_state.expert_name}")
        with top3:
            if st.button("Reset", use_container_width=True):
                _reset_consultation()
                st.rerun()

        rec = _try_live_reco(state, engine)
        col1, col2, col3 = st.columns([1.35, 2.45, 1.25], gap="medium")
        with col1:
            done, total = _render_progress(flow, state, current, rules)
        with col2:
            if current > 22:
                final_rec = engine.decide(state)
                record_id = _ensure_history_record(store, state, final_rec)
                _render_final(final_rec, state, rules, store, record_id)
            else:
                question = next((q for q in flow if int(q["id"]) == current), None)
                if question is None:
                    st.error("Question introuvable dans dynamic_rules.json")
                else:
                    values = _render_input(question, state)
                    _render_feedback(st.session_state.last_feedback)
                    if values is not None:
                        state.update(values)
                        main_value = values.get(question["state_key"])
                        st.session_state.last_feedback = _feedback_for(rules, question["state_key"], main_value)
                        st.session_state.current_action = get_next_action(current, state, rules=rules)
                        st.rerun()
        with col3:
            _render_live_preview(rec, done, total)

    with tab_history:
        _render_history(store, database_url)


if __name__ == "__main__":
    main()
