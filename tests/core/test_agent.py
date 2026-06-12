from __future__ import annotations

import logging

import pytest
from conftest import ScriptedProvider

from src.core.agent import ActParseError, Agent, AgentSetup, Phase, PhaseKind
from src.core.config import ProviderCfg
from src.core.memory import MemoryEntry
from src.providers.base import HttpAttempt, ProviderUnavailable


class RaisingProvider:
    """Дубль провайдера: бросает ProviderUnavailable с заданными HttpAttempt'ами."""

    def __init__(self, attempts):
        self._attempts = tuple(attempts)
        self.calls = 0

    async def complete(self, *, system, messages, temperature, max_tokens):
        self.calls += 1
        e = ProviderUnavailable("boom")
        e.request = {"model": "m", "messages": []}
        e.attempts = self._attempts
        raise e

    async def aclose(self):
        pass


def _agent(provider, persona="You are A.", **kw):
    cfg = ProviderCfg(base_url="http://x/v1", model="m", temperature=0.0, max_tokens=64)
    return Agent("A1", AgentSetup(persona, cfg, "You are AI agent {id}."), provider, **kw)


def _decide(context="Pick a number.", rules="RULES"):
    return Phase(PhaseKind.DECIDE, context, rules=rules)


def _talk(context="Negotiate.", rules="RULES"):
    return Phase(PhaseKind.TALK, context, rules=rules)


async def test_decide_clean_json():
    p = ScriptedProvider(['{"number": 4, "rationale": "ok"}'])
    r = await _agent(p).act(_decide())
    assert r.data == {"number": 4, "rationale": "ok"}
    assert r.public_text is None
    assert len(p.calls) == 1


async def test_decide_json_in_fence():
    p = ScriptedProvider(['```json\n{"number": 7, "rationale": "x"}\n```'])
    r = await _agent(p).act(_decide())
    assert r.data["number"] == 7


async def test_decide_json_among_prose():
    p = ScriptedProvider(['Sure! {"number": 2, "rationale": "y"} done'])
    r = await _agent(p).act(_decide())
    assert r.data["number"] == 2


async def test_decide_string_number_coerced():
    p = ScriptedProvider(['{"number": "3", "rationale": "z"}'])
    r = await _agent(p).act(_decide())
    assert r.data["number"] == 3


async def test_decide_out_of_range_then_valid():
    p = ScriptedProvider(
        ['{"number": 15, "rationale": "bad"}', '{"number": 3, "rationale": "good"}']
    )
    r = await _agent(p).act(_decide())
    assert r.data["number"] == 3
    assert len(p.calls) == 2


async def test_decide_invalid_twice_then_valid():
    p = ScriptedProvider(["nope", "still nope", '{"number": 1, "rationale": "r"}'])
    r = await _agent(p).act(_decide())
    assert r.data["number"] == 1
    assert len(p.calls) == 3


async def test_decide_persistent_failure_raises():
    p = ScriptedProvider(["no", "no", "no"])
    a = _agent(p)
    with pytest.raises(ActParseError) as ei:        # no substitution: abort instead of a fallback number
        await a.act(_decide())
    assert a.parse_failures == 1
    assert len(p.calls) == 3                         # retried with _CORRECTION, then gave up
    assert [c.status for c in ei.value.calls] == ["parse_error"] * 3
    assert ei.value.agent_id == "A1" and ei.value.phase == "decide"


async def test_decide_rejects_bool_number():
    # JSON `true` must not be accepted as the integer 1.
    p = ScriptedProvider(['{"number": true, "rationale": "x"}', '{"number": 5, "rationale": "ok"}'])
    r = await _agent(p).act(_decide())
    assert r.data["number"] == 5
    assert len(p.calls) == 2


def _reflect(context="Reflect on the outcome.", rules="RULES"):
    return Phase(PhaseKind.REFLECT, context, rules=rules)


async def test_reflect_clean_json():
    p = ScriptedProvider(['{"reflection": "partner kept the deal"}'])
    r = await _agent(p).act(_reflect())
    assert r.data == {"reflection": "partner kept the deal"}
    assert r.public_text is None
    assert len(p.calls) == 1


async def test_reflect_invalid_then_valid_retries_with_correction():
    p = ScriptedProvider(["nope", '{"reflection": "ok"}'])
    r = await _agent(p).act(_reflect())
    assert r.data["reflection"] == "ok"
    assert len(p.calls) == 2
    _, messages = p.calls[1]
    assert len(messages) == 1                     # поправка склеена в то же user-сообщение
    assert "reflection" in messages[-1].content  # correction names the expected key


async def test_reflect_persistent_failure_raises():
    p = ScriptedProvider(["no", "no", "no"])
    a = _agent(p)
    with pytest.raises(ActParseError):
        await a.act(_reflect())
    assert a.parse_failures == 1
    assert len(p.calls) == 3


async def test_system_and_messages_assembly():
    p = ScriptedProvider(['{"number": 0, "rationale": ""}'])
    await _agent(p, persona="PERSONA").act(Phase(PhaseKind.DECIDE, "SITUATION", rules="GAME RULES"))
    system, messages = p.calls[0]
    assert "PERSONA" in system and "GAME RULES" in system
    assert messages[-1].role == "user"
    assert messages[-1].content == "SITUATION"
    assert len(messages) == 1  # empty memory -> only the situation message


async def test_system_omits_persona_when_none():
    p = ScriptedProvider(['{"number": 0, "rationale": ""}'])
    await _agent(p, persona=None).act(Phase(PhaseKind.DECIDE, "SITUATION", rules="GAME RULES"))
    system, _ = p.calls[0]
    assert system == "You are AI agent A1.\n\nGAME RULES"


async def test_identity_prompt_from_setup_fills_id():
    p = ScriptedProvider(['{"number": 0, "rationale": ""}'])
    cfg = ProviderCfg(base_url="http://x/v1", model="m")
    agent = Agent("A1", AgentSetup(None, cfg, identity_prompt="Ты ИИ-игрок {id}."), p)
    await agent.act(Phase(PhaseKind.DECIDE, "SITUATION", rules="R"))
    system, _ = p.calls[0]
    assert system == "Ты ИИ-игрок A1.\n\nR"          # шаблон из AgentSetup, {id} подставлен агентом


async def test_usage_summed_over_retries():
    p = ScriptedProvider(["bad", '{"number": 5, "rationale": "r"}'])
    r = await _agent(p).act(_decide())
    assert r.usage == (4, 6)  # 2 calls * (2 prompt, 3 completion)


async def test_talk_clean_json():
    p = ScriptedProvider(['{"message": "let us both take 4", "ready": true}'])
    r = await _agent(p).act(_talk())
    assert r.data == {"message": "let us both take 4", "ready": True}
    assert r.public_text == "let us both take 4"
    assert len(p.calls) == 1


async def test_talk_ready_coercion():
    for raw, expected in [("yes", True), ("1", True), (1, True),
                          ("false", False), ("no", False), (0, False)]:
        p = ScriptedProvider(['{"message": "m", "ready": %s}'
                              % (raw if isinstance(raw, int) else '"%s"' % raw)])
        r = await _agent(p).act(_talk())
        assert r.data["ready"] is expected, (raw, expected)


async def test_talk_ready_missing_defaults_false():
    p = ScriptedProvider(['{"message": "still thinking"}'])
    r = await _agent(p).act(_talk())
    assert r.data == {"message": "still thinking", "ready": False}


async def test_talk_message_missing_then_valid():
    p = ScriptedProvider(['{"ready": true}', '{"message": "ok", "ready": true}'])
    r = await _agent(p).act(_talk())
    assert r.data["message"] == "ok"
    assert len(p.calls) == 2


async def test_talk_persistent_failure_raises():
    p = ScriptedProvider(["no", "no", "no"])
    a = _agent(p)
    with pytest.raises(ActParseError):               # talk too: abort, no empty-message substitution
        await a.act(_talk())
    assert a.parse_failures == 1
    assert len(p.calls) == 3


async def test_predict_phase_parses_number_and_rationale():
    p = ScriptedProvider(['{"number": 7, "rationale": "mid is safe"}'])
    r = await _agent(p).act(Phase(PhaseKind.PREDICT, "predict your partner", rules="R"))
    assert r.data["number"] == 7
    assert r.data["rationale"] == "mid is safe"
    assert r.public_text is None  # PREDICT produces no public message


async def test_memory_diary_precedes_situation_in_one_message():
    p = ScriptedProvider(['{"number": 0, "rationale": ""}'])
    a = _agent(p)
    a.memory.add(
        MemoryEntry(
            round=2,
            partner_id="A7",
            transcript=[{"speaker": "A7", "text": "take 5", "ready": True}],
            my_number=6,
            my_rationale="betray",
            partner_number=5,
            outcome="DC",
            payoff=5.0,
        )
    )
    await a.act(Phase(PhaseKind.DECIDE, "SITUATION", rules="R"))
    _system, messages = p.calls[0]
    assert len(messages) == 1 and messages[0].role == "user"   # дневник и ситуация склеены
    content = messages[0].content
    assert "Round 2" in content and "A7" in content            # дневник присутствует
    assert content.endswith("SITUATION")                       # ситуация — в конце
    assert content.index("Round 2") < content.index("SITUATION")   # дневник раньше ситуации


def _trace_records(caplog):
    return [r for r in caplog.records if r.name == "src.core.agent"]


async def test_decide_logs_full_llm_input_at_debug(caplog):
    caplog.set_level(logging.DEBUG, logger="src.core.agent")
    p = ScriptedProvider(['{"number": 4, "rationale": "ok"}'])
    await _agent(p, persona="PERSONA").act(
        Phase(PhaseKind.DECIDE, "SITUATION", rules="GAME RULES")
    )
    records = _trace_records(caplog)
    assert len(records) == 1
    msg = records[0].getMessage()
    assert "PERSONA" in msg and "GAME RULES" in msg  # system prompt
    assert "SITUATION" in msg                        # phase context message


async def test_predict_logs_llm_input_at_debug(caplog):
    caplog.set_level(logging.DEBUG, logger="src.core.agent")
    p = ScriptedProvider(['{"number": 7, "rationale": "guess"}'])
    await _agent(p).act(Phase(PhaseKind.PREDICT, "PREDICT-CTX", rules="R"))
    records = _trace_records(caplog)
    assert len(records) == 1
    assert "PREDICT-CTX" in records[0].getMessage()


async def test_talk_logs_no_llm_input(caplog):
    caplog.set_level(logging.DEBUG, logger="src.core.agent")
    p = ScriptedProvider(['{"message": "hi", "ready": true}'])
    await _agent(p).act(_talk())
    assert _trace_records(caplog) == []


async def test_decide_retry_logs_attempt_with_correction(caplog):
    caplog.set_level(logging.DEBUG, logger="src.core.agent")
    p = ScriptedProvider(["bad", '{"number": 1, "rationale": "r"}'])
    await _agent(p).act(_decide())
    records = _trace_records(caplog)
    assert len(records) == 2  # one record per provider attempt
    second = records[1].getMessage()
    assert "ONLY valid JSON" in second  # the correction message is part of the input


# ---- L2: захват сырых вызовов (ActResult.calls) ----

async def test_clean_decide_records_one_ok_call():
    p = ScriptedProvider(['{"number": 4, "rationale": "ok"}'])
    r = await _agent(p).act(_decide())
    assert len(r.calls) == 1
    c = r.calls[0]
    assert (c.agent_id, c.phase, c.attempt, c.http_attempt) == ("A1", "decide", 1, 1)
    assert c.status == "ok"
    assert c.status_code == 200
    assert c.response == '{"number": 4, "rationale": "ok"}'
    assert c.request["messages"][0]["role"] == "system"   # дословный payload на руках
    assert c.prompt_tokens == 2 and c.completion_tokens == 3


async def test_parse_retry_records_two_calls_with_statuses():
    p = ScriptedProvider(["garbage", '{"number": 1, "rationale": "r"}'])
    r = await _agent(p).act(_decide())
    assert [c.status for c in r.calls] == ["parse_error", "ok"]
    assert [c.attempt for c in r.calls] == [1, 2]          # отдельный complete() на попытку
    assert r.calls[0].response == "garbage"               # сырой невалидный ответ сохранён


async def test_all_parse_fail_raises_with_logged_calls():
    p = ScriptedProvider(["x", "y", "z"])
    with pytest.raises(ActParseError) as ei:
        await _agent(p).act(_decide())
    assert [c.status for c in ei.value.calls] == ["parse_error"] * 3   # all three attempts logged


async def test_provider_error_reraised_with_calls_and_context():
    attempts = [
        HttpAttempt("server_error", 503, {"m": 1}, None, "busy", "HTTP 503"),
        HttpAttempt("network", None, {"m": 1}, None, None, "boom"),
    ]
    p = RaisingProvider(attempts)
    with pytest.raises(ProviderUnavailable) as ei:
        await _agent(p).act(_decide())
    e = ei.value
    assert (e.agent_id, e.phase, e.attempt) == ("A1", "decide", 1)
    # каждая HTTP-попытка развёрнута в LLMCall с контекстом агента
    assert [c.status for c in e.calls] == ["server_error", "network"]
    assert [c.http_attempt for c in e.calls] == [1, 2]
    assert e.calls[0].response_raw == "busy"              # тело 5xx сохранено
