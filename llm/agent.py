from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from dotenv import load_dotenv

from core.corrections_store import CorrectionsStore
from core.decision_engine import DecisionEngine, Recommendation
from llm.tools_definition import ANTHROPIC_TOOLS, OPENAI_TOOLS, OpticalToolExecutor

load_dotenv()

SYSTEM_PROMPT = (
    "Tu es un assistant opticien expert. Tu aides a recommander des verres optiques "
    "en suivant des regles metier strictes.\n\n"
    "Tu as acces a 4 outils. Pour chaque question client :\n"
    "1. Utilise tool_rag_retrieve pour charger les regles pertinentes\n"
    "2. Utilise tool_decide pour obtenir la decision (JAMAIS toi-meme)\n"
    "3. Utilise tool_ask_next_question pour continuer la conversation\n"
    "4. Utilise tool_explain UNIQUEMENT pour la reponse finale\n\n"
    "REGLE ABSOLUE : le type de verre (Simple foyer/Eyezen/Varilux) "
    "est determine UNIQUEMENT par l'age via tool_decide. "
    "Tu ne peux jamais ecraser cette decision. "
    "Tu ne dois jamais recommander un verre sans passer par tool_decide."
)


class OpticalAgent:
    def __init__(
        self,
        *,
        decision_engine: DecisionEngine | None = None,
        corrections_store: CorrectionsStore | None = None,
        database_url: str | None = None,
        reload_on_decide: bool = True,
        anthropic_client: Any | None = None,
        anthropic_api_key: str | None = None,
        openai_client: Any | None = None,
        openai_api_key: str | None = None,
        openai_base_url: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        trace_path: str | Path = "agent_trace.log",
        max_iterations: int = 10,
    ):
        self.corrections_store = corrections_store or CorrectionsStore(database_url=database_url)
        self.decision_engine = decision_engine or DecisionEngine(
            corrections_store=self.corrections_store,
            reload_on_decide=reload_on_decide,
        )
        self.executor = OpticalToolExecutor(self.decision_engine)
        self._anthropic_client = anthropic_client
        self._openai_client = openai_client
        self.anthropic_api_key = (anthropic_api_key or "").strip() or None
        self.openai_api_key = (openai_api_key or os.getenv("OPENAI_API_KEY") or "ollama").strip()
        self.openai_base_url = (openai_base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434/v1").strip()
        self.provider = _normalize_provider(provider or os.getenv("LLM_AGENT_PROVIDER") or os.getenv("LLM_PROVIDER") or "anthropic")
        self.model = model or os.getenv("LLM_AGENT_MODEL") or os.getenv("LLM_FORMAT_MODEL") or _default_model(self.provider)
        self.trace_path = Path(trace_path)
        self.max_iterations = int(max_iterations)
        self.tool_call_counts: Counter[str] = Counter()
        self.tool_calls: list[dict[str, Any]] = []

    def decide(self, state: dict[str, Any]) -> Recommendation:
        result = self._execute_tool("tool_decide", {"state": state})
        return Recommendation(**result["recommendation"])

    def rag_retrieve(self, query: str, collection: str) -> str:
        result = self._execute_tool("tool_rag_retrieve", {"query": query, "collection": collection})
        return str(result.get("context", ""))

    def ask_next_question(self, state: dict[str, Any], last_answer: str = "") -> dict[str, Any]:
        return self._execute_tool("tool_ask_next_question", {"state": state, "last_answer": last_answer})

    def explain(self, recommendation: dict[str, Any], rag_context: str) -> str:
        result = self._execute_tool(
            "tool_explain",
            {"recommendation": recommendation, "rag_context": rag_context},
        )
        return str(result.get("explanation", ""))

    def run(self, user_message: str, conversation_history: list[dict[str, Any]]) -> dict[str, Any]:
        if self.provider == "anthropic":
            return self._run_anthropic(user_message, conversation_history)
        if self.provider in ("ollama", "openai_compat"):
            return self._run_openai_compatible(user_message, conversation_history)
        raise ValueError(f"Provider agent inconnu: {self.provider}")

    def _run_anthropic(self, user_message: str, conversation_history: list[dict[str, Any]]) -> dict[str, Any]:
        messages = list(conversation_history or [])
        start_call_index = len(self.tool_calls)
        if user_message:
            messages.append({"role": "user", "content": user_message})

        for iteration in range(self.max_iterations):
            response = self._client().messages.create(
                model=self.model,
                max_tokens=1200,
                temperature=0,
                system=SYSTEM_PROMPT,
                tools=ANTHROPIC_TOOLS,
                messages=messages,
            )
            blocks = list(getattr(response, "content", []) or [])
            tool_blocks = [block for block in blocks if _block_get(block, "type") == "tool_use"]

            if not tool_blocks:
                text = "\n".join(
                    str(_block_get(block, "text", ""))
                    for block in blocks
                    if _block_get(block, "type") == "text"
                ).strip()
                messages.append({"role": "assistant", "content": text})
                return {
                    "type": "text",
                    "message": text,
                    "conversation_history": messages,
                    "tool_calls": list(self.tool_calls[start_call_index:]),
                    "iterations": iteration + 1,
                }

            messages.append({"role": "assistant", "content": [_serialize_block(block) for block in blocks]})
            tool_results = []
            for block in tool_blocks:
                tool_name = str(_block_get(block, "name", ""))
                tool_input = dict(_block_get(block, "input", {}) or {})
                tool_use_id = str(_block_get(block, "id", ""))
                result = self._execute_tool(tool_name, tool_input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    }
                )
            messages.append({"role": "user", "content": tool_results})

        return {
            "type": "error",
            "message": "Nombre maximal d'iterations atteint sans reponse finale.",
            "conversation_history": messages,
            "tool_calls": list(self.tool_calls[start_call_index:]),
            "iterations": self.max_iterations,
        }

    def _run_openai_compatible(self, user_message: str, conversation_history: list[dict[str, Any]]) -> dict[str, Any]:
        messages = list(conversation_history or [])
        start_call_index = len(self.tool_calls)
        if user_message:
            messages.append({"role": "user", "content": user_message})

        for iteration in range(self.max_iterations):
            response = self._openai().chat.completions.create(
                model=self.model,
                temperature=0,
                max_tokens=1200,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, *messages],
                tools=OPENAI_TOOLS,
                tool_choice="auto",
            )
            message = response.choices[0].message
            tool_calls = list(getattr(message, "tool_calls", None) or [])

            if not tool_calls:
                text = str(getattr(message, "content", "") or "").strip()
                if not _has_tool_call(self.tool_calls[start_call_index:], "tool_decide"):
                    forced_result = self._force_decide_from_message(
                        user_message=user_message,
                        messages=messages,
                        iteration=iteration,
                        start_call_index=start_call_index,
                    )
                    if forced_result is not None:
                        return forced_result
                messages.append({"role": "assistant", "content": text})
                return {
                    "type": "text",
                    "message": text,
                    "conversation_history": messages,
                    "tool_calls": list(self.tool_calls[start_call_index:]),
                    "iterations": iteration + 1,
                }

            assistant_tool_calls = []
            for tool_call in tool_calls:
                function = getattr(tool_call, "function", None)
                assistant_tool_calls.append(
                    {
                        "id": str(getattr(tool_call, "id", "")),
                        "type": "function",
                        "function": {
                            "name": str(getattr(function, "name", "")),
                            "arguments": str(getattr(function, "arguments", "{}") or "{}"),
                        },
                    }
                )
            messages.append({"role": "assistant", "content": getattr(message, "content", None) or "", "tool_calls": assistant_tool_calls})

            for tool_call in tool_calls:
                function = getattr(tool_call, "function", None)
                tool_name = str(getattr(function, "name", ""))
                raw_arguments = str(getattr(function, "arguments", "{}") or "{}")
                try:
                    tool_input = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    tool_input = {}
                result = self._execute_tool(tool_name, dict(tool_input or {}))
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(getattr(tool_call, "id", "")),
                        "name": tool_name,
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    }
                )

        return {
            "type": "error",
            "message": "Nombre maximal d'iterations atteint sans reponse finale.",
            "conversation_history": messages,
            "tool_calls": list(self.tool_calls[start_call_index:]),
            "iterations": self.max_iterations,
        }

    def _force_decide_from_message(
        self,
        *,
        user_message: str,
        messages: list[dict[str, Any]],
        iteration: int,
        start_call_index: int,
    ) -> dict[str, Any] | None:
        state = _extract_state_from_message(user_message)
        if not state or "Q1_age" not in state:
            return None

        result = self._execute_tool("tool_decide", {"state": state})
        recommendation = dict(result.get("recommendation") or {})
        text = _format_forced_recommendation(recommendation)
        messages.append({"role": "assistant", "content": text})
        return {
            "type": "text",
            "message": text,
            "conversation_history": messages,
            "tool_calls": list(self.tool_calls[start_call_index:]),
            "iterations": iteration + 1,
        }

    def _client(self) -> Any:
        if self._anthropic_client is None:
            api_key = self.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY n'est pas configuree pour l'agent Anthropic.")
            self._anthropic_client = Anthropic(api_key=api_key)
        return self._anthropic_client

    def _openai(self) -> Any:
        if self._openai_client is None:
            from openai import OpenAI

            self._openai_client = OpenAI(base_url=self.openai_base_url, api_key=self.openai_api_key or "ollama")
        return self._openai_client

    def _execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        result = self.executor.execute(tool_name, tool_input)
        self.tool_call_counts[tool_name] += 1
        call = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool": tool_name,
            "input": tool_input,
            "result_keys": sorted(result.keys()),
        }
        self.tool_calls.append(call)
        self._write_trace(call)
        return result

    def _write_trace(self, call: dict[str, Any]) -> None:
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(call, ensure_ascii=False, default=str) + "\n")


def _block_get(block: Any, key: str, default: Any = None) -> Any:
    if isinstance(block, dict):
        return block.get(key, default)
    return getattr(block, key, default)


def _serialize_block(block: Any) -> dict[str, Any]:
    block_type = _block_get(block, "type")
    if block_type == "text":
        return {"type": "text", "text": str(_block_get(block, "text", ""))}
    if block_type == "tool_use":
        return {
            "type": "tool_use",
            "id": str(_block_get(block, "id", "")),
            "name": str(_block_get(block, "name", "")),
            "input": dict(_block_get(block, "input", {}) or {}),
        }
    return dict(block) if isinstance(block, dict) else {"type": str(block_type or "unknown")}


def _has_tool_call(tool_calls: list[dict[str, Any]], tool_name: str) -> bool:
    return any(call.get("tool") == tool_name for call in tool_calls)


def _extract_state_from_message(message: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(message or ""):
        if char != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(message[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(candidate, dict):
            continue
        if isinstance(candidate.get("state"), dict):
            return dict(candidate["state"])
        if any(str(key).startswith("Q") for key in candidate):
            return dict(candidate)
    return None


def _format_forced_recommendation(recommendation: dict[str, Any]) -> str:
    lines = [
        "Decision calculee par tool_decide.",
        f"Type : {recommendation.get('type_verre', '-')}",
        f"Indice : {recommendation.get('indice', '-')}",
        f"Traitement : {recommendation.get('traitement', '-')}",
        f"Couleur : {recommendation.get('couleur', '-')}",
    ]
    corridor = recommendation.get("corridor_type")
    if corridor:
        lines.append(f"Corridor : {corridor}")
    return "\n".join(lines)


def _normalize_provider(provider: str) -> str:
    value = provider.strip().lower()
    if value in ("openai-compatible", "openai_compatible", "openai"):
        return "openai_compat"
    return value


def _default_model(provider: str) -> str:
    if provider == "anthropic":
        return "claude-sonnet-4-6"
    return "mistral"