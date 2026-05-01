"""
Parsing tolérant aux typos des entrées utilisateur (CLI optique).
Aucun LLM, aucun appel réseau.

API publique :
    parse_yes_no(raw: str) -> ParseResult[bool]
    parse_choice(raw: str, options: list[str]) -> ParseResult[int]

ParseResult expose :
    .value           : la valeur parsée (bool ou index 1-based) ou None
    .status          : "auto" | "confirm" | "reject" | "ambiguous"
    .matched_label   : libellé reconnu (pour affichage "Compris comme : ...")
    .score           : score rapidfuzz (0..100)
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from rapidfuzz import fuzz, process

# ---------------------------------------------------------------------------
# Seuils
# ---------------------------------------------------------------------------
SEUIL_AUTO = 85       # ≥ : accepté silencieusement
SEUIL_CONFIRM = 60    # 60..84 : demander confirmation à l'utilisateur
                      # < 60   : rejeté

# Seuils spécifiques yes/no (plus stricts)
SEUIL_AUTO_YN = 90
SEUIL_CONFIRM_YN = 70

# ---------------------------------------------------------------------------
# Tokens YES / NO  (FR / EN / AR + variantes courantes)
# ---------------------------------------------------------------------------
YES_TOKENS: list[str] = [
    # FR
    "oui", "o", "ouais", "ouaip", "ouii", "ouioui", "si",
    # EN
    "yes", "y", "yep", "yap", "yess", "yeah", "yea", "yas", "yup", "ok", "okay",
    # AR (translit + script)
    "نعم", "ايه", "aywa", "naam", "na3am",
    # Connotations
    "vrai", "true", "1",
]

NO_TOKENS: list[str] = [
    # FR
    "non", "n", "nan", "nope", "jamais", "pas",
    # EN
    "no", "nop", "nope", "nah", "nein", "noo", "never",
    # AR
    "لا", "la", "lala", "mafich",
    # Variantes typos courantes (acceptées sans confirmation)
    "nno", "nob", "no0",
    # Connotations
    "faux", "false", "0",
]

AMBIGUOUS_TOKENS: list[str] = [
    "peu", "parfois", "temps", "depend", "peut", "bof",
    "moyen", "certain", "relative", "kadhi", "شوية",
]

# Numéraux écrits → entier
NUMBER_WORDS: dict[str, int] = {
    # FR
    "zero": 0, "zéro": 0,
    "un": 1, "une": 1, "premier": 1, "première": 1,
    "deux": 2, "deuxième": 2, "second": 2, "seconde": 2,
    "trois": 3, "troisième": 3,
    "quatre": 4, "quatrième": 4,
    "cinq": 5, "cinquième": 5,
    "six": 6, "sixième": 6,
    "sept": 7, "septième": 7,
    "huit": 8, "huitième": 8,
    "neuf": 9, "neuvième": 9,
    "dix": 10, "dixième": 10,
    # EN
    "zero_en": 0,
    "one": 1, "first": 1,
    "two": 2, "second_en": 2,
    "three": 3, "third": 3,
    "four": 4, "fourth": 4,
    "five": 5, "fifth": 5,
    "six_en": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

# Réécrit pour ne garder que les vrais mots utilisables en lookup
NUMBER_WORDS = {
    "zero": 0, "zéro": 0, "un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4,
    "cinq": 5, "six": 6, "sept": 7, "huit": 8, "neuf": 9, "dix": 10,
    "premier": 1, "première": 1, "deuxième": 2, "second": 2, "seconde": 2,
    "troisième": 3, "quatrième": 4, "cinquième": 5,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six_en": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "first": 1, "third": 3,
}


# ---------------------------------------------------------------------------
# Résultat
# ---------------------------------------------------------------------------
@dataclass
class ParseResult:
    value: Any
    status: str          # "auto" | "confirm" | "reject" | "ambiguous"
    matched_label: str | None = None
    score: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status in ("auto", "confirm")


# ---------------------------------------------------------------------------
# Helpers de normalisation
# ---------------------------------------------------------------------------
def _strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _normalize(text: str) -> str:
    """Minuscule, sans accents, sans ponctuation/espace de bord."""
    text = (text or "").strip().lower()
    text = _strip_accents(text)
    # Supprime ponctuation finale et espaces multiples
    text = re.sub(r"[!?.,;:]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _is_arabic(text: str) -> bool:
    return any("\u0600" <= c <= "\u06FF" for c in text)


def deduplicate_chars(s: str) -> str:
    """
    Réduit les répétitions de caractères pour capter les entrées emphatiques.
    Règles:
      - mots <= 4 chars: runs >=2 -> 1 ("nn" -> "n", "ouii" -> "oui")
      - mots  > 4 chars: runs >=3 -> 1 (préserve "belle", corrige "yyyep")
    """
    if not s:
        return s

    def _collapse_word(word: str) -> str:
        if not word:
            return word
        threshold = 2 if len(word) <= 4 else 3
        out: list[str] = []
        i = 0
        while i < len(word):
            j = i + 1
            while j < len(word) and word[j] == word[i]:
                j += 1
            run_len = j - i
            if run_len >= threshold:
                out.append(word[i])
            else:
                out.extend(word[i:j])
            i = j
        return "".join(out)

    parts = re.split(r"(\s+)", s)
    return "".join(_collapse_word(p) if not p.isspace() else p for p in parts)


# ---------------------------------------------------------------------------
# parse_yes_no
# ---------------------------------------------------------------------------
def parse_yes_no(raw: str) -> ParseResult:
    """Convertit une réponse libre en bool. Retourne ParseResult."""
    if not raw or not raw.strip():
        return ParseResult(value=None, status="reject", score=0.0)

    text = deduplicate_chars(raw.strip().lower())
    # 1. Match exact arabe (avant normalisation qui dépouille les diacritiques)
    if _is_arabic(text):
        if any(tok in text for tok in ("نعم", "ايه", "أيوة", "اه")):
            return ParseResult(value=True, status="auto", matched_label="نعم", score=100)
        if "لا" in text:
            return ParseResult(value=False, status="auto", matched_label="لا", score=100)
        if "شوية" in text:
            return ParseResult(value=None, status="ambiguous", matched_label=text, score=0)

    norm = _normalize(text)
    if not norm:
        return ParseResult(value=None, status="reject", score=0.0)

    # 2. Match exact (rapide) sur tokens normalisés
    yes_norm = {_normalize(t) for t in YES_TOKENS}
    no_norm = {_normalize(t) for t in NO_TOKENS}
    if norm in yes_norm:
        return ParseResult(value=True, status="auto", matched_label=norm, score=100)
    if norm in no_norm:
        return ParseResult(value=False, status="auto", matched_label=norm, score=100)

    # 2-bis. Ambiguïtés sémantiques (ni oui, ni non)
    ambiguous_norm = {_normalize(t) for t in AMBIGUOUS_TOKENS}
    words = set(norm.split())
    if any(tok in words or tok in norm for tok in ambiguous_norm):
        return ParseResult(value=None, status="ambiguous", matched_label=norm, score=0)

    # 3. Fuzzy : on cherche le meilleur match dans chaque pool
    yes_match = process.extractOne(norm, list(yes_norm), scorer=fuzz.ratio)
    no_match = process.extractOne(norm, list(no_norm), scorer=fuzz.ratio)

    yes_score = yes_match[1] if yes_match else 0
    no_score = no_match[1] if no_match else 0

    if yes_score < SEUIL_CONFIRM_YN and no_score < SEUIL_CONFIRM_YN:
        return ParseResult(value=None, status="reject", score=max(yes_score, no_score))

    # On choisit le pool gagnant
    if yes_score >= no_score:
        value, label, score = True, yes_match[0], yes_score
    else:
        value, label, score = False, no_match[0], no_score

    status = "auto" if score >= SEUIL_AUTO_YN else "confirm"
    return ParseResult(value=value, status=status, matched_label=label, score=score)


# ---------------------------------------------------------------------------
# parse_choice
# ---------------------------------------------------------------------------
def _parse_int(text: str) -> int | None:
    """Tente d'extraire un entier (chiffre ou mot)."""
    norm = _normalize(text)
    if not norm:
        return None
    # Chiffre brut éventuellement entouré
    m = re.fullmatch(r"\D*(\d+)\D*", norm)
    if m:
        return int(m.group(1))
    # Mot numéral exact
    if norm in NUMBER_WORDS:
        return NUMBER_WORDS[norm]
    # Anagramme (transposition de lettres : "tow" → "two", "thrre" → "three")
    sorted_norm = "".join(sorted(norm))
    for word, val in NUMBER_WORDS.items():
        if "".join(sorted(word)) == sorted_norm and abs(len(word) - len(norm)) <= 1:
            return val
    # Fuzzy classique (typos par substitution : "thre" → "three")
    keys = list(NUMBER_WORDS.keys())
    m_ratio = process.extractOne(norm, keys, scorer=fuzz.ratio)
    if m_ratio and m_ratio[1] >= 80:
        return NUMBER_WORDS[m_ratio[0]]
    return None


def parse_choice(raw: str, options: list[str]) -> ParseResult:
    """
    Convertit une réponse libre en index 1-based dans `options`.
    Stratégie :
      1. Numérique direct (chiffre ou mot)
      2. Fuzzy match sur les libellés
    """
    if not raw or not raw.strip() or not options:
        return ParseResult(value=None, status="reject", score=0.0)

    n = len(options)

    # 1. Numérique
    num = _parse_int(raw)
    if num is not None:
        if 1 <= num <= n:
            return ParseResult(
                value=num, status="auto",
                matched_label=options[num - 1], score=100,
            )
        # Hors range → on ne rejette pas tout de suite, on tente le fuzzy
        # mais on garde trace
        num_out_of_range = True
    else:
        num_out_of_range = False

    # 2. Fuzzy sur libellés (token_set_ratio = robuste aux mots multiples)
    norm_options = [_normalize(o) for o in options]
    norm_input = _normalize(raw)

    match = process.extractOne(
        norm_input, norm_options, scorer=fuzz.token_set_ratio
    )
    # On compare aussi avec partial_ratio (mot contenu) pour "jamais" dans
    # "Jamais / rarement"
    partial = process.extractOne(
        norm_input, norm_options, scorer=fuzz.partial_ratio
    )

    candidates = [m for m in (match, partial) if m is not None]
    if not candidates:
        return ParseResult(value=None, status="reject", score=0.0)

    best = max(candidates, key=lambda m: m[1])
    label_norm, score, idx = best  # process retourne (choice, score, index)
    idx_1 = idx + 1

    if score < SEUIL_CONFIRM:
        # Si on avait un nombre hors range, le signaler en priorité
        if num_out_of_range:
            return ParseResult(
                value=None, status="reject",
                matched_label=f"{num} hors plage 1..{n}", score=0.0,
            )
        return ParseResult(value=None, status="reject", score=score)

    status = "auto" if score >= SEUIL_AUTO else "confirm"
    return ParseResult(
        value=idx_1, status=status,
        matched_label=options[idx], score=score,
    )
