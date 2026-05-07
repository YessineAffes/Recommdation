from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from core.corrections_store import CorrectionsStore
from llm.agent import OpticalAgent


def _minimal_state(**overrides):
    state = {
        "Q1_age": 28,
        "Q2_montage": "plastique",
        "Q3_correction_total": "<200",
    }
    state.update(overrides)
    return state


@pytest.fixture
def optical_agent(tmp_path):
    store = CorrectionsStore(tmp_path / "corrections.db")
    return OpticalAgent(
        corrections_store=store,
        reload_on_decide=False,
        trace_path=tmp_path / "agent_trace.log",
    )


def test_agent_client_28_ans_simple_foyer_garanti(optical_agent):
    rec = optical_agent.decide(_minimal_state(Q1_age=28))

    assert rec.type_verre == "Simple foyer"
    assert optical_agent.tool_call_counts["tool_decide"] == 1


def test_agent_client_36_ans_eyezen_garanti(optical_agent):
    rec = optical_agent.decide(_minimal_state(Q1_age=36))

    assert rec.type_verre == "Eyezen"
    assert optical_agent.tool_call_counts["tool_decide"] == 1


def test_agent_client_55_ans_perce_varilux_indice_160_garanti(optical_agent):
    rec = optical_agent.decide(
        _minimal_state(
            Q1_age=55,
            Q2_montage="perce",
            Q3_correction_total="<200",
        )
    )

    assert "Varilux" in rec.type_verre
    assert rec.indice == "1.60"
    assert optical_agent.tool_call_counts["tool_decide"] == 1

    trace_lines = (optical_agent.trace_path).read_text(encoding="utf-8").splitlines()
    assert any(json.loads(line)["tool"] == "tool_decide" for line in trace_lines)


class _FakeMessages:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return _Response([
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "tool_decide",
                    "input": {"state": _minimal_state(Q1_age=36)},
                }
            ])
        return _Response([{"type": "text", "text": "La recommandation passe par Eyezen."}])


class _FakeAnthropicClient:
    def __init__(self):
        self.messages = _FakeMessages()


class _Response:
    def __init__(self, content):
        self.content = content


def test_agent_run_execute_la_boucle_tool_calling(tmp_path):
    store = CorrectionsStore(tmp_path / "corrections.db")
    agent = OpticalAgent(
        corrections_store=store,
        anthropic_client=_FakeAnthropicClient(),
        provider="anthropic",
        reload_on_decide=False,
        trace_path=tmp_path / "agent_trace.log",
    )

    result = agent.run("Client de 36 ans", [])

    assert result["type"] == "text"
    assert result["message"] == "La recommandation passe par Eyezen."
    assert result["conversation_history"][-1] == {"role": "assistant", "content": "La recommandation passe par Eyezen."}
    assert [call["tool"] for call in result["tool_calls"]] == ["tool_decide"]
    assert agent.tool_call_counts["tool_decide"] == 1


def test_agent_run_demande_une_cle_anthropic_si_aucun_client(monkeypatch, optical_agent):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    optical_agent.provider = "anthropic"
    optical_agent.anthropic_api_key = None

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        optical_agent.run("Bonjour", [])


class _FakeOpenAICompletions:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        assert kwargs["model"] == "mistral"
        assert kwargs["tools"][0]["type"] == "function"
        if self.calls == 1:
            message = SimpleNamespace(
                content="",
                tool_calls=[
                    SimpleNamespace(
                        id="call_1",
                        function=SimpleNamespace(
                            name="tool_decide",
                            arguments=json.dumps({"state": _minimal_state(Q1_age=55, Q2_montage="perce")}),
                        ),
                    )
                ],
            )
        else:
            message = SimpleNamespace(content="Decision confirmee via Ollama.", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _FakeOpenAIClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_FakeOpenAICompletions())


def test_agent_run_execute_tool_calling_openai_compatible(tmp_path):
    store = CorrectionsStore(tmp_path / "corrections.db")
    agent = OpticalAgent(
        corrections_store=store,
        openai_client=_FakeOpenAIClient(),
        provider="ollama",
        model="mistral",
        reload_on_decide=False,
        trace_path=tmp_path / "agent_trace.log",
    )

    result = agent.run("Client de 55 ans", [])

    assert result["type"] == "text"
    assert result["message"] == "Decision confirmee via Ollama."
    assert result["conversation_history"][-1] == {"role": "assistant", "content": "Decision confirmee via Ollama."}
    assert [call["tool"] for call in result["tool_calls"]] == ["tool_decide"]
    assert agent.tool_call_counts["tool_decide"] == 1


class _FakeOpenAINoToolCompletions:
    def create(self, **kwargs):
        message = SimpleNamespace(content="Reponse libre incorrecte.", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _FakeOpenAINoToolClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_FakeOpenAINoToolCompletions())


def test_agent_run_force_tool_decide_si_ollama_ne_declenche_pas_outil(tmp_path):
    store = CorrectionsStore(tmp_path / "corrections.db")
    agent = OpticalAgent(
        corrections_store=store,
        openai_client=_FakeOpenAINoToolClient(),
        provider="ollama",
        model="mistral",
        reload_on_decide=False,
        trace_path=tmp_path / "agent_trace.log",
    )

    payload = json.dumps(_minimal_state(Q1_age=36), ensure_ascii=False)
    result = agent.run(f"Analyse ce state avec tool_decide: {payload}", [])

    assert result["type"] == "text"
    assert "Eyezen" in result["message"]
    assert "Reponse libre incorrecte" not in result["message"]
    assert [call["tool"] for call in result["tool_calls"]] == ["tool_decide"]
    assert agent.tool_call_counts["tool_decide"] == 1