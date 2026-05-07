"""
Formulation layer aval: enrichit l'explication sans jamais toucher a la decision.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from .llm_client import get_llm_client


class FormulationLayer:
    def __init__(self):
        self.client = get_llm_client(role="format")

    def _fallback_justification(self, recommendation: dict[str, Any]) -> str:
        """Template statique, sans LLM, utilise uniquement la recommandation deja decidee."""
        type_verre = recommendation.get("type_verre", "N/A")
        indice = recommendation.get("indice", "N/A")
        traitement = recommendation.get("traitement", "N/A")
        couleur = recommendation.get("couleur", "N/A")
        return (
            f"Verre {type_verre} recommande pour votre profil. "
            f"Indice {indice} adapte a votre correction. "
            f"Traitement {traitement} retenu selon votre besoin principal. "
            f"Couleur {couleur} coherente avec votre sensibilite lumineuse et votre environnement."
        )

    def explain(
        self,
        recommendation: dict[str, Any],
        rag_chunks: list[str],
        state: dict[str, Any],
        temperature: float = 0.2,
    ) -> str:
        """
        Retourne TOUJOURS une string.
        Ne modifie jamais `recommendation` (copie defensive).
        """
        rec = deepcopy(recommendation)
        try:
            system = (
                "Tu expliques une recommandation OPTIQUE deja decidee. "
                "Interdiction de changer le resultat. "
                "Ton role est purement explicatif et pedagogique. "
                "Sois concis (4-6 phrases), factuel, sans inventer de nouvelles valeurs."
            )
            user = (
                "Recommendation figee (ne pas modifier):\n"
                f"{rec}\n\n"
                "Contexte questionnaire:\n"
                f"{state}\n\n"
                "Extraits RAG utiles:\n"
                + "\n---\n".join(rag_chunks[:3])
                + "\n\nRends uniquement une justification en francais, texte brut."
            )
            text = self.client.complete(
                system=system,
                user=user,
                temperature=temperature,
                max_tokens=320,
            )
            text = (text or "").strip()
            if not text:
                return self._fallback_justification(rec)
            return text
        except Exception:
            return self._fallback_justification(rec)
