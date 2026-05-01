"""Tests automatiques pour les 15 scénarios documentés."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from core.decision_engine import DecisionEngine

ENGINE = DecisionEngine()


def _state(**overrides):
    """État par défaut neutre, surchargé par les kwargs."""
    base = {
        "Q1_age": 30, "Q2_montage": "plastique",
        "Q3_correction_total": "<200", "Q3_add": 0.0, "Q3_diff_od_og": "<=200",
        "Q4_vision_laterale": "mixte", "Q4_usage_ordinateur": "jamais",
        "Q4_conduite_nuit": False, "Q4_adaptation_facile": False, "Q4_innovation": False,
        "Q5_besoin_principal": "transparence", "Q6_sante_oculaire": "ras",
        "Q7_paire_soleil": False, "Q7_environnement": "interieur",
        "Q7_gene_lumiere": "faible", "Q7_reflets_genants": False,
        "Q7_sensible_soleil": False, "Q7_mal_voir_soleil": False,
    }
    base.update(overrides)
    return base


SCENARIOS = [
    # (nom, état, attendus)
    ("S1_jeune_interieur", _state(
        Q1_age=24, Q2_montage="metallique",
        Q3_correction_total="<200", Q5_besoin_principal="lumiere_bleue",
        Q7_environnement="interieur", Q7_gene_lumiere="faible",
    ), {"type_verre": "Simple foyer", "indice": "1.50",
         "traitement": "Crizal Prevencia", "couleur": "Blanc"}),

    ("S2_eyezen_mixte", _state(
        Q1_age=36, Q3_correction_total="225-400", Q5_besoin_principal="rayures",
        Q7_environnement="mixte", Q7_gene_lumiere="moyenne",
    ), {"type_verre": "Eyezen", "indice": "1.56",
            "traitement": "Crizal Rock", "couleur": "Transitions Gen S"}),

    ("S3_eyezen_perce", _state(
        Q1_age=39, Q2_montage="perce", Q3_correction_total="225-400",
        Q5_besoin_principal="transparence", Q7_environnement="interieur",
        Q7_gene_lumiere="faible",
    ), {"type_verre": "Eyezen", "indice": "1.60",
         "traitement": "Crizal Sapphire HR", "couleur": "Blanc"}),

    ("S4_varilux_s_design", _state(
        Q1_age=52, Q3_correction_total="425-600", Q3_add=2.25,
        Q5_besoin_principal="nettoyage", Q7_environnement="mixte",
        Q7_gene_lumiere="moyenne",
        ), {"type_verre": "Varilux S Design", "design_varilux": "Varilux S Design", "indice": "1.60",
            "traitement": "Crizal Easy Pro", "couleur": "Transitions Gen S"}),

    ("S5_varilux_x_design_adaptation", _state(
        Q1_age=58, Q2_montage="nylor", Q3_correction_total="225-400", Q3_add=2.50,
        Q4_adaptation_facile=True, Q5_besoin_principal="transparence",
        Q7_environnement="interieur", Q7_gene_lumiere="faible",
    ), {"type_verre": "Varilux X Design", "design_varilux": "Varilux X Design", "indice": "1.56",
         "traitement": "Crizal Sapphire HR", "couleur": "Blanc"}),

    ("S6_varilux_xr_innovation", _state(
        Q1_age=49, Q2_montage="metallique", Q3_correction_total="425-600", Q3_add=2.75,
        Q4_innovation=True, Q5_besoin_principal="lumiere_bleue",
        Q7_environnement="mixte", Q7_gene_lumiere="forte",
        ), {"type_verre": "Varilux XR", "design_varilux": "Varilux XR", "indice": "1.60",
            "traitement": "Crizal Prevencia", "couleur": "Transitions Gen S"}),

    ("S7_varilux_physio", _state(
        Q1_age=61, Q3_correction_total="425-600", Q3_add=2.00, Q3_diff_od_og=">200",
        Q5_besoin_principal="transparence", Q7_environnement="mixte",
        Q7_gene_lumiere="moyenne",
        ), {"type_verre": "Varilux Physio 3.0", "design_varilux": "Varilux Physio 3.0", "indice": "1.60",
            "traitement": "Crizal Sapphire HR", "couleur": "Blanc"}),

    ("S8_varilux_comfort_conduite", _state(
        Q1_age=54, Q2_montage="metallique", Q3_correction_total="225-400", Q3_add=2.00,
        Q4_vision_laterale="yeux", Q4_conduite_nuit=True,
        Q5_besoin_principal="conduite_soir", Q7_environnement="exterieur",
        Q7_gene_lumiere="forte", Q7_reflets_genants=True,
            ), {"type_verre": "Varilux Physio 3.0", "design_varilux": "Varilux Physio 3.0", "indice": "1.56",
                "traitement": "Crizal Drive", "couleur": "Transitions X Polarized"}),

    ("S9_varilux_liberty_polarisant", _state(
        Q1_age=45, Q3_correction_total="<200", Q3_add=1.75,
        Q4_vision_laterale="tete", Q5_besoin_principal="transparence",
        Q7_paire_soleil=True, Q7_environnement="exterieur", Q7_gene_lumiere="tres_forte",
        Q7_mal_voir_soleil=True, Q7_reflets_genants=True,
        ), {"type_verre": "Varilux Liberty 3.0", "design_varilux": "Varilux Liberty 3.0", "indice": "1.50",
            "traitement": "Crizal Sapphire HR", "couleur": "Transitions X Polarized"}),

    ("S10_eyezen_glaucome", _state(
        Q1_age=34, Q3_correction_total="225-400", Q5_besoin_principal="nettoyage",
        Q6_sante_oculaire="glaucome", Q7_environnement="mixte", Q7_gene_lumiere="moyenne",
    ), {"type_verre": "Eyezen", "indice": "1.56",
            "traitement": "Crizal Prevencia", "couleur": "Transitions Gen S"}),

    ("S11_varilux_x_design_4250", _state(
        Q1_age=46, Q3_correction_total="225-400", Q3_add=2.25,
        Q4_adaptation_facile=True, Q5_besoin_principal="transparence",
        Q7_environnement="interieur", Q7_gene_lumiere="faible",
    ), {"type_verre": "Varilux X Design", "design_varilux": "Varilux X Design", "indice": "1.56",
         "traitement": "Crizal Sapphire HR", "couleur": "Blanc"}),

    ("S12_varilux_xr_57", _state(
        Q1_age=57, Q2_montage="metallique", Q3_correction_total="425-600", Q3_add=2.50,
        Q4_innovation=True, Q5_besoin_principal="lumiere_bleue",
        Q7_environnement="mixte", Q7_gene_lumiere="forte",
        ), {"type_verre": "Varilux XR", "design_varilux": "Varilux XR", "indice": "1.60",
            "traitement": "Crizal Prevencia", "couleur": "Transitions Gen S"}),

    ("S13_varilux_comfort_polarisant", _state(
        Q1_age=50, Q2_montage="nylor", Q3_correction_total="225-400", Q3_add=2.00,
        Q4_vision_laterale="yeux", Q4_conduite_nuit=True,
        Q5_besoin_principal="conduite_soir", Q7_environnement="exterieur",
        Q7_gene_lumiere="forte", Q7_sensible_soleil=True,
        Q7_mal_voir_soleil=True, Q7_reflets_genants=True,
        ), {"type_verre": "Varilux Physio 3.0", "design_varilux": "Varilux Physio 3.0", "indice": "1.56",
            "traitement": "Crizal Drive", "couleur": "Transitions X Polarized"}),

    ("S14_varilux_physio_diff_add", _state(
        Q1_age=62, Q3_correction_total="425-600", Q3_add=2.25, Q3_diff_od_og=">200",
        Q5_besoin_principal="nettoyage", Q7_environnement="mixte", Q7_gene_lumiere="moyenne",
        ), {"type_verre": "Varilux S Design", "design_varilux": "Varilux S Design", "indice": "1.60",
            "traitement": "Crizal Sapphire HR", "couleur": "Blanc"}),

    ("S15_eyezen_perce_solaire", _state(
        Q1_age=32, Q2_montage="perce", Q3_correction_total="<200",
        Q5_besoin_principal="rayures", Q7_paire_soleil=True,
        Q7_environnement="exterieur", Q7_gene_lumiere="forte",
    ), {"type_verre": "Eyezen", "indice": "1.60",
         "traitement": "Crizal Rock", "couleur": "Solaire"}),
]


@pytest.mark.parametrize("name,state,expected", SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_scenario(name, state, expected):
    rec = ENGINE.decide(state)
    assert rec.type_verre == expected["type_verre"], f"[{name}] type_verre"
    assert rec.design_varilux == rec.type_verre, f"[{name}] design=type"
    if "design_varilux" in expected:
        assert rec.design_varilux == expected["design_varilux"], f"[{name}] design_varilux"
    assert rec.indice == expected["indice"], f"[{name}] indice"
    assert rec.traitement == expected["traitement"], f"[{name}] traitement"
    assert rec.couleur == expected["couleur"], f"[{name}] couleur"


def test_age_never_overridden():
    """Quoi qu'il arrive, l'âge dicte le type. Aucune autre règle ne l'écrase."""
    s = _state(Q1_age=25, Q2_montage="perce", Q3_add=3.0, Q4_innovation=True)
    rec = ENGINE.decide(s)
    assert rec.type_verre == "Simple foyer"
    assert rec.design_varilux == "Simple foyer"


def test_perce_overrides_correction():
    """Percé impose 1.60 même avec correction faible."""
    rec = ENGINE.decide(_state(Q2_montage="perce", Q3_correction_total="<200"))
    assert rec.indice == "1.60"


def test_pathology_overrides_need():
    """Glaucome impose Crizal Prevencia, écrase le besoin nettoyage."""
    rec = ENGINE.decide(_state(Q5_besoin_principal="nettoyage", Q6_sante_oculaire="glaucome"))
    assert rec.traitement == "Crizal Prevencia"


def test_varilux_ranking_keeps_xr_over_x_design():
    """XR est au-dessus de X Design dans la hierarchie cumulative."""
    rec = ENGINE.decide(_state(
        Q1_age=50, Q3_add=2.75, Q4_innovation=True, Q3_correction_total="425-600",
    ))
    assert rec.type_verre == "Varilux XR"
    assert rec.design_varilux == "Varilux XR"


def test_varilux_ranking_never_downgrades():
    """Physio atteint ne doit pas etre retrograde par Comfort ou Liberty."""
    rec = ENGINE.decide(_state(
        Q1_age=50,
        Q3_add=0.0,
        Q4_vision_laterale="yeux",
        Q4_conduite_nuit=True,
        Q4_usage_ordinateur="jamais",
    ))
    assert rec.type_verre == "Varilux Physio 3.0"
    assert rec.design_varilux == "Varilux Physio 3.0"


def test_crizal_ranking_never_downgrades():
    """Prevencia atteint ne doit pas etre retrograde par Rock."""
    rec = ENGINE.decide(_state(Q5_besoin_principal="lumiere_bleue", Q22_style_vie_actif=True))
    assert rec.traitement == "Crizal Prevencia"


def test_crizal_same_rank_keeps_first_choice():
    """Sapphire HR et Prevencia ont le meme rang: le premier choix reste conserve."""
    rec = ENGINE.decide(_state(Q5_besoin_principal="transparence", Q6_sante_oculaire="glaucome"))
    assert rec.traitement == "Crizal Sapphire HR"
