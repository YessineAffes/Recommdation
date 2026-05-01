"""Application Streamlit admin (3 onglets)."""
from __future__ import annotations

import sys
from pathlib import Path

# Permet d'importer core/ et rag/ depuis n'importe ou
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from admin.tabs import expert_eval, rules_editor, product_form

st.set_page_config(
    page_title="Recommandation Optique - Admin",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Theme blanc + bleu moderne
st.markdown(
    """
    <style>
        :root {
            --primary: #2563EB;
            --primary-dark: #1E40AF;
            --primary-light: #DBEAFE;
            --bg: #FFFFFF;
            --surface: #F8FAFC;
            --border: #E2E8F0;
            --text: #0F172A;
            --text-muted: #64748B;
        }
        .stApp { background: var(--bg); color: var(--text); }
        section[data-testid="stSidebar"] { display: none; }
        h1, h2, h3, h4 { color: var(--primary-dark) !important; font-weight: 700; }
        .block-container { padding-top: 1.5rem; max-width: 1100px; }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px; background: var(--surface);
            padding: 6px; border-radius: 12px; border: 1px solid var(--border);
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent; color: var(--text-muted);
            border-radius: 8px; padding: 8px 16px; font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background: var(--primary) !important; color: white !important;
        }

        /* Buttons */
        .stButton > button, .stFormSubmitButton > button {
            background: var(--primary); color: white; border: none;
            border-radius: 8px; padding: 0.5rem 1.2rem; font-weight: 600;
            transition: all 0.15s ease;
        }
        .stButton > button:hover, .stFormSubmitButton > button:hover {
            background: var(--primary-dark); transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(37,99,235,0.25);
        }

        /* Inputs */
        input, textarea, select,
        div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
            border-radius: 8px !important; border-color: var(--border) !important;
            background: white !important;
        }
        div[data-baseweb="select"] > div:focus-within,
        div[data-baseweb="input"] > div:focus-within {
            border-color: var(--primary) !important;
            box-shadow: 0 0 0 3px var(--primary-light) !important;
        }

        /* Chat messages */
        [data-testid="stChatMessage"] {
            background: var(--surface); border-radius: 12px;
            padding: 12px 16px; margin-bottom: 8px; border: 1px solid var(--border);
        }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
            background: var(--primary-light); border-color: var(--primary);
        }

        /* Cards / expanders */
        [data-testid="stExpander"] {
            border: 1px solid var(--border); border-radius: 10px;
            background: var(--surface);
        }

        /* Alerts */
        div[data-baseweb="notification"] { border-radius: 8px; }

        /* Header banner */
        .app-header {
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
            color: white; padding: 1.5rem 2rem; border-radius: 14px;
            margin-bottom: 1.5rem; box-shadow: 0 8px 24px rgba(37,99,235,0.18);
        }
        .app-header h1 { color: white !important; margin: 0; font-size: 1.75rem; }
        .app-header p { color: rgba(255,255,255,0.85); margin: 0.4rem 0 0; }
    </style>
    <div class="app-header">
        <h1>Console Admin - Recommandation Optique</h1>
        <p>Evaluation expert, regles dynamiques et catalogue produit RAG</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs([
    "Evaluation Expert",
    "Editeur de Regles",
    "Nouveau Produit",
])

with tab1:
    expert_eval.render()

with tab2:
    rules_editor.render()

with tab3:
    product_form.render()
