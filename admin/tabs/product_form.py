"""Onglet Streamlit : Nouveau Produit (genere JSON + ingere RAG)."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from rag.rag_watcher import ingest_product_json

DOCS_PATH = Path(__file__).resolve().parents[2] / "rag" / "documents"
COLLECTIONS = ["types_verres", "indices", "traitements", "couleurs"]
FAMILLES = ["Simple foyer", "Eyezen", "Varilux", "Traitement", "Couleur"]


def _slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", s.strip()).strip("_")
    return s or "produit"


def render() -> None:
    st.header("Nouveau Produit (catalogue RAG)")
    st.caption("Cree un fichier JSON dans rag/documents/ et l'ingere immediatement dans ChromaDB.")

    with st.form("product_form"):
        nom = st.text_input("Nom du produit", placeholder="ex: Varilux XR Track")
        famille = st.selectbox("Famille", FAMILLES)
        sous_famille = st.text_input("Sous-famille")
        description = st.text_area("Description courte (2-3 phrases)")
        indications = st.text_area("Indications (pour qui ? quel usage ?)")
        contre_indications = st.text_area("Contre-indications")
        avantages = st.text_area("Avantages cles")
        comparaison = st.text_area("Comparaison produit precedent (optionnel)")
        collection_cible = st.selectbox("Collection ChromaDB cible", COLLECTIONS)
        submit = st.form_submit_button("Indexer le produit")

    if submit:
        if not nom.strip():
            st.error("Nom du produit requis")
            return

        DOCS_PATH.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"product_{_slug(nom)}_{ts}.json"
        fp = DOCS_PATH / filename

        payload = {
            "nom": nom.strip(),
            "famille": famille,
            "sous_famille": sous_famille.strip(),
            "description": description.strip(),
            "indications": indications.strip(),
            "contre_indications": contre_indications.strip(),
            "avantages": avantages.strip(),
            "comparaison_precedent": comparaison.strip(),
            "collection_cible": collection_cible,
            "source": "form",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        fp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        st.info(f"Fichier cree: {fp.name}")

        result = ingest_product_json(fp)
        if result.get("ok"):
            st.success(f"Produit indexe dans la collection [{result['collection']}]")
            st.json(result)
        else:
            st.error(f"Echec ingestion: {result.get('message')}")
