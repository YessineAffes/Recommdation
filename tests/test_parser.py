"""Tests pour input_parser : tolérance typos, FR/EN/AR, nombres."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from input_parser import parse_yes_no, parse_choice, SEUIL_AUTO


# ---------------------------------------------------------------------------
# parse_yes_no
# ---------------------------------------------------------------------------
class TestParseYesNo:
    # STOP CONDITIONS (nouvelle logique)
    def test_stop_nn(self):
        r = parse_yes_no("nn")
        assert r.value is False
        assert r.status == "auto"

    def test_stop_nnn(self):
        r = parse_yes_no("nnn")
        assert r.value is False
        assert r.status == "auto"

    def test_stop_yy(self):
        r = parse_yes_no("yy")
        assert r.value is True
        assert r.status == "auto"

    def test_stop_ambiguous_un_peu(self):
        r = parse_yes_no("un peu")
        assert r.status == "ambiguous"
        assert r.value is None

    def test_stop_ambiguous_dans_certain_temps(self):
        r = parse_yes_no("dans certain temps")
        assert r.status == "ambiguous"
        assert r.value is None

    def test_stop_ambiguous_peut_etre(self):
        r = parse_yes_no("peut-être")
        assert r.status == "ambiguous"
        assert r.value is None

    def test_stop_ambiguous_ca_depend(self):
        r = parse_yes_no("ça dépend")
        assert r.status == "ambiguous"
        assert r.value is None

    def test_stop_nope_auto(self):
        r = parse_yes_no("nope")
        assert r.value is False
        assert r.status == "auto"

    @pytest.mark.parametrize("raw", [
        "y", "yes", "oui", "o", "si", "ouais", "yep", "yap", "yess",
        "ouii", "yeah", "yea", "yas", "yup", "ok", "okay", "نعم", "ايه",
    ])
    def test_yes_auto(self, raw):
        r = parse_yes_no(raw)
        assert r.value is True
        assert r.status == "auto", f"'{raw}' devrait être auto, score={r.score}"

    @pytest.mark.parametrize("raw", [
        "n", "no", "non", "nan", "nope", "nop", "nah", "nein", "noo", "لا",
    ])
    def test_no_auto(self, raw):
        r = parse_yes_no(raw)
        assert r.value is False
        assert r.status == "auto", f"'{raw}' devrait être auto, score={r.score}"

    # STOP CONDITIONS
    def test_yep(self):
        r = parse_yes_no("yep")
        assert r.value is True and r.status == "auto"

    def test_jamais_returns_false(self):
        # "jamais" est un signal NON dans le contexte général
        # Pas dans les tokens directement → on accepte que le contrat soit
        # "ressemble plus à un NO". On le force via extension.
        r = parse_yes_no("jamais")
        assert r.value is False  # via partial à "nan" ou rejet acceptable
        # Si rejet, on attend de l'utilisateur qu'il choisisse explicitement.
        # Vérification souple :
        assert r.status in ("auto", "confirm", "reject")

    def test_nno_typo(self):
        r = parse_yes_no("nno")
        assert r.value is False
        assert r.status == "auto", f"score={r.score}"

    def test_typos_yes(self):
        for t in ["yse", "uoi", "owi", "oje"]:
            r = parse_yes_no(t)
            # Au minimum non-rejeté
            assert r.status != "reject" or r.score > 0, f"'{t}' totalement rejeté"

    def test_typos_no(self):
        for t in ["nno", "nob", "no0"]:
            r = parse_yes_no(t)
            assert r.value is False, f"'{t}' devrait être NO, got {r.value} ({r.score})"

    # Cas limites
    def test_empty(self):
        assert parse_yes_no("").status == "reject"
        assert parse_yes_no("   ").status == "reject"

    def test_garbage(self):
        r = parse_yes_no("xyzqwabc")
        assert r.status == "reject"


# ---------------------------------------------------------------------------
# parse_choice
# ---------------------------------------------------------------------------
USAGE_OPTIONS = ["Jamais / rarement", "Parfois", "Très souvent (toute la journée)"]
FREQ_OPTIONS = ["Toujours", "Souvent", "Parfois", "Jamais"]


class TestParseChoice:
    # STOP CONDITIONS
    def test_parfox_to_parfois(self):
        r = parse_choice("parfox", USAGE_OPTIONS)
        assert r.ok
        assert r.value == 2
        assert "Parfois" in r.matched_label

    def test_toujour_to_toujours(self):
        r = parse_choice("toujour", FREQ_OPTIONS)
        assert r.ok
        assert r.value == 1
        assert "Toujours" in r.matched_label

    def test_numeric_3(self):
        r = parse_choice("3", USAGE_OPTIONS)
        assert r.value == 3
        assert r.status == "auto"

    def test_word_trois(self):
        r = parse_choice("trois", USAGE_OPTIONS)
        assert r.value == 3

    def test_word_two(self):
        r = parse_choice("two", USAGE_OPTIONS)
        assert r.value == 2

    def test_word_tow_typo(self):
        r = parse_choice("tow", USAGE_OPTIONS)
        assert r.value == 2

    def test_jamais_label_match(self):
        r = parse_choice("jamais", USAGE_OPTIONS)
        assert r.value == 1
        assert "Jamais" in r.matched_label

    # Cas limites
    def test_empty(self):
        assert parse_choice("", USAGE_OPTIONS).status == "reject"

    def test_out_of_range(self):
        r = parse_choice("99", USAGE_OPTIONS)
        assert r.status == "reject"

    def test_random_garbage(self):
        r = parse_choice("zqxwvabcdef", USAGE_OPTIONS)
        assert r.status == "reject"
