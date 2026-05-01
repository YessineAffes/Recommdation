from __future__ import annotations

import os
from typing import Protocol

from dotenv import load_dotenv

load_dotenv()


class LLMClient(Protocol):
    def complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> str:
        ...


class AnthropicClient:
    def __init__(self, model: str):
        from anthropic import Anthropic

        self.model = model
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> str:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        chunks = [
            block.text for block in msg.content
            if getattr(block, "type", None) == "text"
        ]
        return "\n".join(chunks).strip()


class OllamaClient:
    """Client OpenAI-compatible pour Ollama/LM Studio/vLLM/Groq/Together."""

    def __init__(self, model: str, base_url: str | None = None):
        from openai import OpenAI

        resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL") or "http://localhost:11434/v1"
        api_key = os.getenv("OPENAI_API_KEY") or "ollama"
        self.model = model
        self.client = OpenAI(base_url=resolved_base_url, api_key=api_key)

    def complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = resp.choices[0].message.content
        return (content or "").strip()


def _model_for_role(role: str) -> str:
    role = role.lower().strip()
    if role == "nlu":
        return os.getenv("LLM_NLU_MODEL", "claude-haiku-4-5-20251001")
    if role in ("format", "formulation"):
        return os.getenv("LLM_FORMAT_MODEL", "claude-sonnet-4-6")
    return os.getenv("LLM_NLU_MODEL", "claude-haiku-4-5-20251001")


def get_llm_client(role: str = "nlu") -> LLMClient:
    provider = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()
    model = _model_for_role(role)

    if provider == "anthropic":
        return AnthropicClient(model=model)
    if provider == "ollama":
        return OllamaClient(
            model=model,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        )
    if provider == "openai_compat":
        return OllamaClient(
            model=model,
            base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434/v1",
        )
    raise ValueError(f"Provider inconnu : {provider}")
