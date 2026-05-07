"""Tests : boucle expert (CORRECT, OVERRIDE prioritaire, blindage DYN)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.corrections_store import HISTORY_CORRECTED, HISTORY_PENDING, HISTORY_VALIDATED, CorrectionsStore
from core.decision_engine import DecisionEngine


BASE_STATE = {
    "Q1_age": 50,
    "Q2_montage": "perce",
    "Q3_correction_total": "425-600",
    "Q4_vision_laterale": "yeux",
    "Q4_usage_ordinateur": "souvent",
    "Q5_besoin_principal": "transparence",
    "Q6_sante_oculaire": "RAS",
    "Q7_type_ordinateur": "bureau",
    "Q7_environnement": "interieur",
    "Q7_gene_lumiere": "faible",
    "Q18_preference_verre": "RAS",
    "Q19_antecedents_familiaux": False,
    "Q20_operation_yeux": False,
    "Q21_fatigue_oculaire": False,
    "Q22_style_vie_actif": False,
}


@pytest.fixture
def tmp_store(tmp_path):
    return CorrectionsStore(tmp_path / "corrections.db")


@pytest.fixture
def empty_dyn(tmp_path):
    p = tmp_path / "dynamic_rules.json"
    p.write_text(json.dumps({"rules": []}), encoding="utf-8")
    return p


def test_recommendation_history_pending_then_corrected(tmp_store):
    auto = {
        "type_verre": "Varilux X Design",
        "design_varilux": "Varilux X Design",
        "indice": "1.60",
        "traitement": "Crizal Sapphire HR",
        "couleur": "Blanc",
    }
    record_id = tmp_store.create_history(BASE_STATE, auto, expert_name="Dr Test")
    pending = tmp_store.get_history(record_id)
    assert pending is not None
    assert pending["status"] == HISTORY_PENDING
    assert pending["recommendation_auto_data"] == auto
    assert pending["recommendation_expert_data"] is None

    expert = dict(auto)
    expert["traitement"] = "Crizal Prevencia"
    tmp_store.update_history(record_id, expert, HISTORY_CORRECTED, "Dr Test", "Protection prioritaire")
    corrected = tmp_store.get_history(record_id)
    assert corrected is not None
    assert corrected["status"] == HISTORY_CORRECTED
    assert corrected["recommendation_auto_data"] == auto
    assert corrected["recommendation_expert_data"] == expert
    assert corrected["notes"] == "Protection prioritaire"


def test_legacy_correction_is_synced_to_history(tmp_store):
    rec = DecisionEngine().decide(BASE_STATE).to_dict()
    tmp_store.add(BASE_STATE, rec, rec, "CORRECT", "ok")
    rows = tmp_store.list_history()
    assert rows[0]["status"] == HISTORY_VALIDATED
    assert rows[0]["recommendation_auto_data"] == rec


def test_correct_status_does_not_override(tmp_store, empty_dyn):
    engine = DecisionEngine(dynamic_rules_path=empty_dyn, corrections_store=tmp_store)
    rec = engine.decide(BASE_STATE)
    # Expert valide tel quel
    tmp_store.add(BASE_STATE, rec.to_dict(), rec.to_dict(), "CORRECT", None)

    # Nouveau decide -> ne doit PAS retourner d'override (status=CORRECT)
    rec2 = engine.decide(BASE_STATE)
    assert "source=expert_override" not in rec2.extras
    assert rec2.traitement == rec.traitement


def test_override_takes_priority(tmp_store, empty_dyn):
    engine = DecisionEngine(dynamic_rules_path=empty_dyn, corrections_store=tmp_store)
    rec = engine.decide(BASE_STATE)

    expert = rec.to_dict()
    expert["traitement"] = "Crizal Sapphire HR"
    expert["couleur"] = "Transitions Gen S"
    tmp_store.add(BASE_STATE, rec.to_dict(), expert, "OVERRIDE", "preference cliente")

    # Cas similaire (age +1, meme montage, meme correction) -> override applique
    similar = dict(BASE_STATE)
    similar["Q1_age"] = 51
    rec2 = engine.decide(similar)
    assert rec2.traitement == "Crizal Sapphire HR"
    assert rec2.couleur == "Transitions Gen S"
    assert "source=expert_override" in rec2.extras


def test_dyn_rule_blindage_type_verre_and_indice_perce(tmp_path, tmp_store):
    dyn = tmp_path / "dynamic_rules.json"
    dyn.write_text(json.dumps({
        "rules": [
            {
                "id": "DYN_001", "priority": 1,
                "condition": {"field": "Q1_age", "operator": ">=", "value": 18},
                "result": {"parameter": "type_verre", "value": "Hack"},
            },
            {
                "id": "DYN_002", "priority": 2,
                "condition": {"field": "Q2_montage", "operator": "==", "value": "perce"},
                "result": {"parameter": "indice", "value": "1.50"},
            },
            {
                "id": "DYN_003", "priority": 3,
                "condition": {"field": "Q1_age", "operator": ">=", "value": 18},
                "result": {"parameter": "couleur", "value": "Test Couleur"},
            },
        ]
    }), encoding="utf-8")

    engine = DecisionEngine(dynamic_rules_path=dyn, corrections_store=tmp_store)
    rec = engine.decide(BASE_STATE)

    # Blindage 1: type_verre jamais ecrase
    assert rec.type_verre != "Hack"
    # Blindage 2: indice reste >= 1.60 (perce)
    assert float(rec.indice) >= 1.60
    # Regle non blindee bien appliquee
    assert rec.couleur == "Test Couleur"
    assert any("DYN_001 ignoree" in t for t in rec.trace)
    assert any("DYN_002 ignoree" in t for t in rec.trace)
    assert any("DYN_003 appliquee" in t for t in rec.trace)


def test_ingest_product_json(tmp_path, monkeypatch):
    from rag import rag_watcher

    # Force ChromaDB dans tmp
    import rag.rag_builder as rb
    monkeypatch.setattr(rb, "DB_PATH", tmp_path / "chroma_test")

    docs = tmp_path / "documents"
    docs.mkdir()
    fp = docs / "product_test.json"
    fp.write_text(json.dumps({
        "nom": "Varilux XR Track",
        "famille": "Varilux",
        "sous_famille": "XR",
        "description": "Verre progressif sport",
        "indications": "Sportifs",
        "contre_indications": "",
        "avantages": "Vision dynamique",
        "comparaison_precedent": "",
        "collection_cible": "types_verres",
        "source": "form",
    }), encoding="utf-8")

    result = rag_watcher.ingest_product_json(fp)
    assert result["ok"] is True
    assert result["collection"] == "types_verres"


def test_ingest_product_payload_new_optical_card_schema(tmp_path, monkeypatch):
    from rag import rag_watcher

    import rag.rag_builder as rb
    monkeypatch.setattr(rb, "DB_PATH", tmp_path / "chroma_card")
    rb._client.cache_clear()

    payload = {
        "nom_produit": "Orma 1.50",
        "concept": "La lumiere sous controle",
        "plage_performance": {"min": -300, "max": 200},
        "avantages": ["Protection UV", "Protection lumiere bleue"],
        "recommande_pour": "Pour les faibles ametropes.",
        "references": [
            {
                "nom": "Crizal Alize+UV Tr Brun",
                "plage_stock": "-300 a +300",
                "cyl_200": 77,
                "cyl_100": 76,
                "spherique": 75,
            }
        ],
        "notes": [{"titre": "Disponibilite generale", "contenu": "En stock uniquement en Brun & Gris"}],
    }

    result = rag_watcher.ingest_product_payload(payload, source_name="orma_150.json", collection_cible="types_verres")
    assert result["ok"] is True
    assert result["collection"] == "types_verres"

    context = rb.get_context("Orma 1.50", "types_verres", n=1)
    assert "Orma 1.50" in context
    assert "La lumiere sous controle" in context

    rb._client.cache_clear()


def test_product_card_is_persisted_in_store(tmp_store):
    payload = {
        "nom_produit": "Orma 1.50",
        "concept": "La lumiere sous controle",
        "plage_performance": {"min": -300, "max": 200},
        "avantages": ["Protection UV"],
        "recommande_pour": "Faibles ametropes",
        "references": [],
        "notes": [],
    }
    rag_result = {"ok": True, "collection": "types_verres", "id": "product_orma_150"}

    record_id = tmp_store.add_product_card(
        payload=payload,
        collection_cible="types_verres",
        expert_name="Dr Test",
        source_file="product_orma_150.json",
        rag_result=rag_result,
    )

    row = tmp_store.get_product_card(record_id)
    assert row is not None
    assert row["product_name"] == "Orma 1.50"
    assert row["collection_cible"] == "types_verres"
    assert row["payload_data"]["nom_produit"] == "Orma 1.50"
    assert row["rag_result_data"]["ok"] is True

    rows = tmp_store.list_product_cards(limit=5)
    assert rows
    assert rows[0]["id"] == record_id
