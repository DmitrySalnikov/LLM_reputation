from __future__ import annotations

import json

import pytest

from conftest import ScriptedProvider

from src.core.config import JudgeCfg, ProviderCfg
from src.games.base import PairingRecord
from src.judge import JudgeError, JudgeVerdict, MessageRef, judge_episode
from src.judge import judge as judge_mod


def _rec(round=0, a="A1", b="A2"):
    return PairingRecord(
        round=round, a_id=a, b_id=b,
        transcript=[
            {"speaker": a, "text": "remember, A3 broke his promise last round", "ready": False},
            {"speaker": b, "text": "then let us both pick 7", "ready": True},
        ],
        a_number=7, b_number=7, a_rationale="PRIVATE-RA", b_rationale="PRIVATE-RB",
        outcome="CC", a_payoff=3.0, b_payoff=3.0,
        usage={"prompt_tokens": 1, "completion_tokens": 1, "calls": 2},
    )


def _cfg(prompt=None):
    kwargs = {"prompt": prompt} if prompt is not None else {}
    return JudgeCfg(provider=ProviderCfg(base_url="http://j/v1", model="judge-m"), **kwargs)


def _verdict_json(emerged=True, evidence=("r0.p0.t0",)):
    return json.dumps(
        {"emerged": emerged, "explanation": "gossip about a third party", "evidence": list(evidence)}
    )


def _patch_provider(monkeypatch, provider):
    monkeypatch.setattr(judge_mod, "make_provider", lambda cfg: provider)
    return provider


async def test_parses_verdict_and_returns_dataclass(monkeypatch):
    provider = _patch_provider(monkeypatch, ScriptedProvider([_verdict_json()]))
    verdict = await judge_episode(_cfg(), [_rec()])
    assert verdict == JudgeVerdict(
        emerged=True,
        explanation="gossip about a third party",
        evidence=[MessageRef(round=0, pair=0, turn=0)],
    )
    assert provider.closed                            # судья закрывает свой провайдер


async def test_prompt_contains_public_transcript_only(monkeypatch):
    provider = _patch_provider(monkeypatch, ScriptedProvider([_verdict_json()]))
    await judge_episode(_cfg(), [_rec()])
    (system, messages), = provider.calls
    prompt = messages[-1].content
    assert "[r0.p0.t0] A1: remember, A3 broke his promise last round" in prompt
    assert "REPUTATION INSTITUTE" in prompt           # default prompt text made it in
    assert "PRIVATE-RA" not in prompt                 # nothing private leaks
    assert system == ""                               # всё в одном user-сообщении


async def test_custom_prompt_replaces_transcript_placeholder(monkeypatch):
    provider = _patch_provider(monkeypatch, ScriptedProvider([_verdict_json()]))
    await judge_episode(_cfg(prompt="JUDGE THIS:\n{transcript}\nJSON only."), [_rec()])
    prompt = provider.calls[0][1][-1].content
    assert prompt.startswith("JUDGE THIS:\n")
    assert "[r0.p0.t1] A2: then let us both pick 7" in prompt


async def test_invalid_evidence_refs_are_dropped(monkeypatch):
    reply = _verdict_json(evidence=("r0.p0.t1", "r9.p9.t9", "garbage"))
    _patch_provider(monkeypatch, ScriptedProvider([reply]))
    verdict = await judge_episode(_cfg(), [_rec()])
    assert verdict.evidence == [MessageRef(round=0, pair=0, turn=1)]


async def test_unparseable_reply_retries_once_with_correction(monkeypatch):
    provider = _patch_provider(
        monkeypatch, ScriptedProvider(["I think reputation emerged!", _verdict_json()])
    )
    verdict = await judge_episode(_cfg(), [_rec()])
    assert verdict.emerged is True
    assert len(provider.calls) == 2
    correction = provider.calls[1][1][-1].content     # последнее user-сообщение второй попытки
    assert "ONLY valid JSON" in correction


async def test_two_unparseable_replies_raise_judge_error(monkeypatch):
    provider = _patch_provider(monkeypatch, ScriptedProvider(["nope", "still nope"]))
    with pytest.raises(JudgeError):
        await judge_episode(_cfg(), [_rec()])
    assert provider.closed                            # провайдер закрыт и при ошибке


async def test_non_bool_emerged_triggers_retry(monkeypatch):
    bad = json.dumps({"emerged": "yes", "explanation": "x", "evidence": []})
    _patch_provider(monkeypatch, ScriptedProvider([bad, _verdict_json(emerged=False, evidence=())]))
    verdict = await judge_episode(_cfg(), [_rec()])
    assert verdict.emerged is False and verdict.evidence == []


async def test_non_list_evidence_triggers_retry(monkeypatch):
    bad = json.dumps({"emerged": True, "explanation": "x", "evidence": "r0.p0.t0"})
    _patch_provider(monkeypatch, ScriptedProvider([bad, _verdict_json()]))
    verdict = await judge_episode(_cfg(), [_rec()])
    assert verdict.emerged is True
    assert verdict.evidence == [MessageRef(round=0, pair=0, turn=0)]
