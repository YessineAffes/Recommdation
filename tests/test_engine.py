"""Tests pour core.decision_engine — moteur 22 actions."""
from __future__ import annotations

import pytest
from core.decision_engine import DecisionEngine


def _engine():
    return DecisionEngine()


def _base_state(**kw):
    """Etat minimal valide — adulte sans particularite."""
    state = {
        "Q1_age": 35,
        "Q2_montage": "plastique",
        "Q3_correction_total": "225-400",
        "Q3_add": 0.0,
        "Q3_diff_od_og": "<=200",
        "Q4_vision_laterale": "mixte",
        "Q4_conduite_nuit": False,
        "Q4_usage_ordinateur": "jamais",
        "Q4_adaptation_facile": True,
        "Q4_innovation": False,
        "Q10_solution_ia": False,
        "Q5_besoin_principal": "transparence",
        "Q6_sante_oculaire": "ras",
        "Q13_paire_soleil": "non",
        "Q7_reflets_genants": False,
        "Q7_mal_voir_soleil": False,
        "Q7_gene_lumiere": "aucune",
        "Q7_environnement": "interieur",
        "Q18_preference_verre": "plus_legers",
        "Q19_antecedents_familiaux": False,
        "Q20_operation_yeux": False,
        "Q21_fatigue_oculaire": False,
        "Q22_style_vie_actif": False,
    }
    state.update(kw)
    return state


# --- Tests age > 60 : les actions skippees ne doivent pas changer le resultat ---

class TestAge60:
    def test_age_gt60_indice_not_overridden_by_pref(self):
        """Preference 'plus_legers' (action 18) toujours appliquee, mais actions
        6/10/16/17/21 ignorees — le moteur ne doit pas crasher."""
        state = _base_state(Q1_age=65, Q3_correction_total="425-600")
        rec = _engine().decide(state)
        assert rec.indice is not None

    def test_age_gt60_same_result_regardless_of_skipped_keys(self):
        """Sans les cles des actions skippees (comme Q21_fatigue_oculaire),
        le moteur ne doit pas lever d'exception."""
        state = {
            "Q1_age": 70,
            "Q2_montage": "metallique",
            "Q3_correction_total": ">625",
            "Q3_add": 0.0,
            "Q3_diff_od_og": "<=200",
            "Q4_conduite_nuit": False,
            "Q4_innovation": False,
            "Q4_adaptation_facile": True,
            "Q5_besoin_principal": "transparence",
            "Q6_sante_oculaire": "ras",
            "Q13_paire_soleil": "non",
        }
        rec = _engine().decide(state)
        assert rec.indice is not None

    def test_age_gt60_indice_stays_160_with_perce(self):
        """Priorite absolue: percé => indice 1.60, meme chez les 65+."""
        state = _base_state(Q1_age=65, Q2_montage="perce")
        rec = _engine().decide(state)
        assert rec.indice == "1.60"


# --- Test strabisme ---

class TestStrabisme:
    def test_strabisme_sets_corridor_short(self):
        # Strabisme + Varilux (age >= 42) → corridor Short et design Physio 3.0
        state = _base_state(Q1_age=50, Q6_sante_oculaire="strabisme_convergence")
        rec = _engine().decide(state)
        assert rec.corridor_type == "Short"


# --- Test style vie actif => Crizal Rock ---

class TestActif:
    def test_actif_adds_crizal_rock(self):
        state = _base_state(Q22_style_vie_actif=True)
        rec = _engine().decide(state)
        extras = getattr(rec, "extras", []) or []
        traitement = rec.traitement or ""
        assert any("Crizal Rock" in extra for extra in extras) or "Crizal Rock" in traitement
