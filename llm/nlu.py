"""
NLU layer : convertit une réponse libre en langage naturel en clé structurée.
LLM = traducteur uniquement. Décision finale = moteur déterministe.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from pydantic import BaseModel, field_validator

from .llm_client import get_llm_client


class NLUResult(BaseModel):
    value: Any
    confidence: float = 1.0

    @field_validator("confidence")
    @classmethod
    def _conf_range(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))


_HEURISTICS: dict[str, list[tuple[re.Pattern, Any]]] = {
    "Q2_montage": [
        (re.compile(r"\bperc[eé]", re.I), "perce"),
        (re.compile(r"\bnylor\b", re.I), "nylor"),
        (re.compile(r"\bplastique|acetate", re.I), "plastique"),
        (re.compile(r"\bm[eé]tal", re.I), "metallique"),
    ],
    "Q3_correction_total": [
        (re.compile(r"inf[eé]rieur.*200|moins de 200|<\s*200", re.I), "<200"),
        (re.compile(r"225.*400|entre 225", re.I), "225-400"),
        (re.compile(r"425.*600|entre 425", re.I), "425-600"),
        (re.compile(r"sup[eé]rieur.*625|>=?\s*625|plus de 625", re.I), ">=625"),
    ],
    "Q3_diff_od_og": [
        (re.compile(r"sup[eé]rieur.*200|>\s*200|forte diff", re.I), ">200"),
        (re.compile(r"inf[eé]rieur|<=\s*200|aucune diff|pas de diff", re.I), "<=200"),
    ],
    "Q4_vision_laterale": [
        (re.compile(r"\byeux\b", re.I), "yeux"),
        (re.compile(r"\bt[eê]te\b", re.I), "tete"),
        (re.compile(r"mixte|les deux|un peu", re.I), "mixte"),
    ],
    "Q4_usage_ordinateur": [
        (re.compile(r"tr[eè]s souvent|toute la journ[eé]e|constamment", re.I), "tres_souvent"),
        (re.compile(r"parfois|occasion|de temps", re.I), "parfois"),
        (re.compile(r"jamais|rarement|peu", re.I), "jamais"),
    ],
    "Q4_conduite_nuit": [
        (re.compile(r"\boui|souvent|toujours", re.I), True),
        (re.compile(r"\bnon|jamais|rarement", re.I), False),
    ],
    "Q4_adaptation_facile": [
        (re.compile(r"\boui|adaptation|facile", re.I), True),
        (re.compile(r"\bnon", re.I), False),
    ],
    "Q4_innovation": [
        (re.compile(r"\boui|innovation|technolog|IA|nouvelle", re.I), True),
        (re.compile(r"\bnon", re.I), False),
    ],
    "Q5_besoin_principal": [
        (re.compile(r"transparen", re.I), "transparence"),
        (re.compile(r"lumi[eè]re bleue|bleue", re.I), "lumiere_bleue"),
        (re.compile(r"conduite|nuit|soir", re.I), "conduite_soir"),
        (re.compile(r"rayure|griffe|r[eé]sistance", re.I), "rayures"),
        (re.compile(r"nettoy|antireflet|easy", re.I), "nettoyage"),
    ],
    "Q6_sante_oculaire": [
        (re.compile(r"glaucome", re.I), "glaucome"),
        (re.compile(r"cataracte", re.I), "cataracte"),
        (re.compile(r"dmla", re.I), "dmla"),
        (re.compile(r"conjonctivite", re.I), "conjonctivite"),
        (re.compile(r"r[eé]tinopathie", re.I), "retinopathie_diabetique"),
        (re.compile(r"pseudophaque", re.I), "pseudophaque"),
        (re.compile(r"aphakie", re.I), "aphakie"),
        (re.compile(r"strabisme", re.I), "strabisme"),
        (re.compile(r"convergence", re.I), "insuffisance_convergence"),
        (re.compile(r"\bras\b|rien|aucun", re.I), "ras"),
    ],
    "Q7_paire_soleil": [
        (re.compile(r"\boui|soleil|solaire", re.I), True),
        (re.compile(r"\bnon", re.I), False),
    ],
    "Q7_environnement": [
        (re.compile(r"int[eé]rieur.*ext|mixte|alterne", re.I), "mixte"),
        (re.compile(r"int[eé]rieur", re.I), "interieur"),
        (re.compile(r"ext[eé]rieur|dehors", re.I), "exterieur"),
    ],
    "Q7_gene_lumiere": [
        (re.compile(r"tr[eè]s fort", re.I), "tres_forte"),
        (re.compile(r"fort", re.I), "forte"),
        (re.compile(r"moyen", re.I), "moyenne"),
        (re.compile(r"faible|peu", re.I), "faible"),
        (re.compile(r"aucune|pas|absente", re.I), "aucune"),
    ],
    "Q7_reflets_genants": [
        (re.compile(r"\boui|reflet|gênant|eblouiss", re.I), True),
        (re.compile(r"\bnon", re.I), False),
    ],
    "Q7_sensible_soleil": [
        (re.compile(r"\boui|sensible", re.I), True),
        (re.compile(r"\bnon", re.I), False),
    ],
    "Q7_mal_voir_soleil": [
        (re.compile(r"\boui|mal voir|difficult", re.I), True),
        (re.compile(r"\bnon", re.I), False),
    ],
}


def _extract_json_safe(raw: str) -> dict[str, Any]:
    """Extraction JSON robuste pour les modèles locaux peu stricts."""
    text = (raw or "").strip()
    if not text:
        return {"value": "UNKNOWN"}

    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.M)

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        snippet = match.group(0)
        try:
            data = json.loads(snippet)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    return {"value": "UNKNOWN"}


def _heuristic_parse(question_id: str, raw: str) -> NLUResult | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    if question_id == "Q1_age":
        m = re.search(r"\d{1,3}", raw)
        if m:
            return NLUResult(value=int(m.group()), confidence=1.0)
        return None
    if question_id == "Q3_add":
        m = re.search(r"(\d+(?:[\.,]\d+)?)", raw)
        if m:
            return NLUResult(value=float(m.group(1).replace(",", ".")), confidence=1.0)
        if re.search(r"sans|aucun|pas d", raw, re.I):
            return NLUResult(value=0.0, confidence=0.9)
        return None

    patterns = _HEURISTICS.get(question_id, [])
    for pattern, value in patterns:
        if pattern.search(raw):
            return NLUResult(value=value, confidence=0.85)
    return None


class NLULayer:
    def __init__(self, rules: dict[str, Any], use_llm: bool | None = None):
        self.rules = rules
        self.questions = self._normalize_questions(rules["questions"])

        provider = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()
        if use_llm is None:
            if provider in ("ollama", "openai_compat"):
                use_llm = True
            else:
                use_llm = bool(os.getenv("ANTHROPIC_API_KEY"))

        self.use_llm = use_llm
        self._client = get_llm_client(role="nlu") if use_llm else None

    def _normalize_questions(self, raw_questions: Any) -> dict[str, dict[str, Any]]:
        if isinstance(raw_questions, dict):
            return raw_questions
        questions: dict[str, dict[str, Any]] = {}
        for item in raw_questions:
            qtype = item.get("type", "choice")
            normalized_type = {"number": "int", "boolean": "bool", "choice": "enum"}.get(qtype, qtype)
            values = [opt["value"] for opt in item.get("options", [])]
            entry = {
                "label": item.get("label", item.get("state_key", "")),
                "type": normalized_type,
            }
            if values:
                entry["values"] = values
            if "min" in item:
                entry["min"] = item["min"]
            if "max" in item:
                entry["max"] = item["max"]
            for key in (item.get("state_key"), item.get("qid"), item.get("key")):
                if key:
                    questions[str(key)] = dict(entry)
            for sub in item.get("sub_questions", []) or []:
                sub_type = sub.get("type", "choice")
                sub_entry = {
                    "label": sub.get("label", sub.get("state_key", "")),
                    "type": {"number": "float", "boolean": "bool", "choice": "enum"}.get(sub_type, sub_type),
                }
                sub_values = [opt["value"] for opt in sub.get("options", [])]
                if sub_values:
                    sub_entry["values"] = sub_values
                if "min" in sub:
                    sub_entry["min"] = sub["min"]
                for key in (sub.get("state_key"), sub.get("key")):
                    if key:
                        questions[str(key)] = dict(sub_entry)
        return questions

    def parse(self, question_id: str, raw_answer: str) -> NLUResult:
        if question_id not in self.questions:
            raise ValueError(f"Question inconnue : {question_id}")

        result = _heuristic_parse(question_id, raw_answer)
        if result is not None:
            self._validate_against_schema(question_id, result.value)
            return result

        if self.use_llm and self._client is not None:
            result = self._llm_parse(question_id, raw_answer)
            self._validate_against_schema(question_id, result.value)
            return result

        raise ValueError(
            f"Impossible d'interpréter la réponse '{raw_answer}' pour {question_id} "
            f"(LLM désactivé et heuristique sans match)."
        )

    def _validate_against_schema(self, question_id: str, value: Any) -> None:
        q = self.questions[question_id]
        qtype = q["type"]
        if qtype == "int":
            if not isinstance(value, int):
                raise ValueError(f"{question_id} attend un entier, reçu {type(value).__name__}")
            if "min" in q and value < q["min"]:
                raise ValueError(f"{question_id} = {value} < min {q['min']}")
            if "max" in q and value > q["max"]:
                raise ValueError(f"{question_id} = {value} > max {q['max']}")
        elif qtype == "float":
            if not isinstance(value, (int, float)):
                raise ValueError(f"{question_id} attend un float")
        elif qtype == "bool":
            if not isinstance(value, bool):
                raise ValueError(f"{question_id} attend un booléen")
        elif qtype == "enum":
            if value not in q["values"]:
                raise ValueError(f"{question_id} = '{value}' non dans {q['values']}")

    def _llm_parse(self, question_id: str, raw: str) -> NLUResult:
        q = self.questions[question_id]
        qtype = q["type"]
        allowed = q.get("values")

        system = (
            "Tu es un extracteur JSON. Réponds UNIQUEMENT avec du JSON valide.\n"
            'Format obligatoire : {"value": "<ta_valeur>"}\n'
            "Règles strictes :\n"
            "- Aucun texte avant ou après le JSON\n"
            "- Aucun markdown, aucune explication\n"
            "- Si tu ne sais pas : {\"value\": \"UNKNOWN\"}\n"
        )

        user = (
            f"Question ID: {question_id}\n"
            f"Question: {q['label']}\n"
            f"Type attendu: {qtype}\n"
            f"Valeurs autorisées: {allowed if allowed is not None else 'libre'}\n"
            f"Réponse opticien: {raw}\n"
        )

        text = self._client.complete(system=system, user=user, temperature=0.0, max_tokens=200)
        data = _extract_json_safe(text)

        value = data.get("value", "UNKNOWN")
        if value == "UNKNOWN":
            raise ValueError(f"Réponse LLM ambiguë/non parseable pour {question_id}: {text!r}")

        # Coercions minimales pour les bool/int/float
        if qtype == "bool" and isinstance(value, str):
            v = value.strip().lower()
            if v in ("true", "oui", "yes", "1"):
                value = True
            elif v in ("false", "non", "no", "0"):
                value = False
        elif qtype == "int" and isinstance(value, str) and value.isdigit():
            value = int(value)
        elif qtype == "float" and isinstance(value, str):
            value = float(value.replace(",", "."))

        confidence = data.get("confidence", 0.7)
        return NLUResult(value=value, confidence=confidence)
