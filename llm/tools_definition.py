from __future__ import annotations

from typing import Any

from core.decision_engine import DecisionEngine
from core.state_machine import get_next_action, get_question, get_question_flow, load_rules
from llm.formulation import FormulationLayer
from rag import rag_builder

RAG_COLLECTIONS = ["types_verres", "indices", "traitements", "couleurs"]

ANTHROPIC_TOOLS: list[dict[str, Any]] = [
    {
        "name": "tool_rag_retrieve",
        "description": "Recupere les regles et extraits RAG pertinents pour une question donnee.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Question ou terme produit a rechercher."},
                "collection": {
                    "type": "string",
                    "enum": RAG_COLLECTIONS,
                    "description": "Collection ChromaDB cible.",
                },
            },
            "required": ["query", "collection"],
            "additionalProperties": False,
        },
    },
    {
        "name": "tool_decide",
        "description": "Applique le moteur de decision deterministe. Regle absolue: le type de verre ne peut jamais etre contourne.",
        "input_schema": {
            "type": "object",
            "properties": {
                "state": {"type": "object", "description": "Etat structure du questionnaire optique."},
            },
            "required": ["state"],
            "additionalProperties": False,
        },
    },
    {
        "name": "tool_ask_next_question",
        "description": "Retourne la prochaine question a poser selon le state courant.",
        "input_schema": {
            "type": "object",
            "properties": {
                "state": {"type": "object", "description": "Etat structure du questionnaire optique."},
                "last_answer": {"type": "string", "description": "Derniere reponse utilisateur brute."},
            },
            "required": ["state", "last_answer"],
            "additionalProperties": False,
        },
    },
    {
        "name": "tool_explain",
        "description": "Formule la recommandation finale en langage naturel sans modifier la decision.",
        "input_schema": {
            "type": "object",
            "properties": {
                "recommendation": {"type": "object", "description": "Recommendation finale deja decidee."},
                "rag_context": {"type": "string", "description": "Contexte RAG utile pour l'explication."},
            },
            "required": ["recommendation", "rag_context"],
            "additionalProperties": False,
        },
    },
]

OPENAI_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["input_schema"],
        },
    }
    for tool in ANTHROPIC_TOOLS
]


class OpticalToolExecutor:
    def __init__(
        self,
        decision_engine: DecisionEngine,
        formulation_layer: FormulationLayer | None = None,
    ):
        self.decision_engine = decision_engine
        self.formulation_layer = formulation_layer

    def execute(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        if tool_name == "tool_rag_retrieve":
            return self.tool_rag_retrieve(
                query=str(tool_input.get("query", "")),
                collection=str(tool_input.get("collection", "")),
            )
        if tool_name == "tool_decide":
            return self.tool_decide(state=dict(tool_input.get("state") or {}))
        if tool_name == "tool_ask_next_question":
            return self.tool_ask_next_question(
                state=dict(tool_input.get("state") or {}),
                last_answer=str(tool_input.get("last_answer", "")),
            )
        if tool_name == "tool_explain":
            return self.tool_explain(
                recommendation=dict(tool_input.get("recommendation") or {}),
                rag_context=str(tool_input.get("rag_context", "")),
            )
        raise ValueError(f"Outil inconnu: {tool_name}")

    def tool_rag_retrieve(self, query: str, collection: str) -> dict[str, Any]:
        context = rag_builder.get_context(query, collection)
        return {"query": query, "collection": collection, "context": context}

    def tool_decide(self, state: dict[str, Any]) -> dict[str, Any]:
        recommendation = self.decision_engine.decide(state)
        return {"recommendation": recommendation.to_dict()}

    def tool_ask_next_question(self, state: dict[str, Any], last_answer: str) -> dict[str, Any]:
        rules = load_rules()
        current_action = _infer_current_action(state)
        next_action = 1 if current_action == 0 else get_next_action(current_action, state, rules=rules)
        while next_action <= 22:
            question = get_question(next_action)
            if question and question.get("state_key") not in state:
                return {
                    "is_complete": False,
                    "next_action": next_action,
                    "question": question,
                    "last_answer": last_answer,
                }
            next_action = get_next_action(next_action, state, rules=rules)
        return {"is_complete": True, "next_action": 23, "question": None, "last_answer": last_answer}

    def tool_explain(self, recommendation: dict[str, Any], rag_context: str) -> dict[str, Any]:
        if self.formulation_layer is None:
            self.formulation_layer = FormulationLayer()
        explanation = self.formulation_layer.explain(
            recommendation=recommendation,
            rag_chunks=[rag_context] if rag_context else [],
            state={},
            temperature=0.3,
        )
        return {"explanation": explanation}


def _infer_current_action(state: dict[str, Any]) -> int:
    completed: list[int] = []
    for question in get_question_flow():
        if question.get("state_key") in state:
            completed.append(int(question["id"]))
    return max(completed, default=0)