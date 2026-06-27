from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from src.core.config import JudgeCfg, ProviderCfg
from src.judge import judge as judge_mod
from src.storage import Storage

import judge_runs


def test_load_judge_cfg_takes_judge_from_config(monkeypatch):
    judge = JudgeCfg(provider=ProviderCfg(base_url="u", model="main-model"))
    monkeypatch.setattr(judge_runs, "load_episode", lambda path: SimpleNamespace(judge=judge))
    assert judge_runs.load_judge_cfg("any.yaml").provider.model == "main-model"


def test_load_judge_cfg_falls_back_when_no_judge_block(monkeypatch):
    monkeypatch.setattr(judge_runs, "load_episode", lambda path: SimpleNamespace(judge=None))
    assert judge_runs.load_judge_cfg("any.yaml") is judge_runs.JUDGE_DEFAULT


class ScriptedProvider:
    """Минимальный двойник провайдера: отдаёт заранее заданный текст вердикта."""

    def __init__(self, replies):
        self._queue = list(replies)

    async def complete(self, *, system, messages, temperature, max_tokens):
        from src.providers.base import Completion
        text = self._queue.pop(0)
        return Completion(text=text, prompt_tokens=1, completion_tokens=1,
                          raw={}, request={}, attempts=())

    async def aclose(self):
        pass


def _verdict_json():
    return json.dumps({"emerged": True, "explanation": "gossip",
                       "evidence": ["r0.p0.t0"]})


def _seed_run(st, run_id=1):
    c = st.conn
    c.execute("INSERT INTO runs(run_id,name,config,config_hash,seed,created_at,finished_at) "
              "VALUES (?,?,?,?,?,?,?)",
              (run_id, None, "{}", "d1", 0, "2026-01-01T00:00:00", "2026-01-01T01:00:00"))
    for aid in ("A1", "A2"):
        c.execute("INSERT INTO agents(run_id,agent_id,system_prompt,provider) VALUES (?,?,?,?)",
                  (run_id, aid, "sys", "{}"))
    c.execute("INSERT INTO rounds(run_id,round_idx) VALUES (?,?)", (run_id, 0))
    c.execute("INSERT INTO pairings(run_id,round_idx,pair_idx,a_id,b_id,finished,"
              "a_number,b_number,a_outcome,a_payoff,b_payoff) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
              (run_id, 0, 0, "A1", "A2", 1, 7, 7, "CC", 3.0, 3.0))
    c.executemany("INSERT INTO messages(run_id,round_idx,pair_idx,turn_idx,speaker,text,ready) "
                  "VALUES (?,?,?,?,?,?,?)",
                  [(run_id, 0, 0, 0, "A1", "hi", 0), (run_id, 0, 0, 1, "A2", "pick 7", 1)])
    c.commit()


def test_judge_run_writes_verdict_then_skips(monkeypatch):
    monkeypatch.setattr(judge_mod, "make_provider", lambda cfg: ScriptedProvider([_verdict_json()]))
    st = Storage(":memory:")
    try:
        _seed_run(st, 1)
        status = asyncio.run(judge_runs.judge_run(st, 1, judge_runs.JUDGE_DEFAULT, force=False))
        assert status == "judged"
        assert st.has_verdict(1) is True
        # второй прогон без --force — пропуск (провайдер не дёргается)
        status2 = asyncio.run(judge_runs.judge_run(st, 1, judge_runs.JUDGE_DEFAULT, force=False))
        assert status2 == "skipped"
    finally:
        st.close()


def test_judge_run_force_rejudges(monkeypatch):
    monkeypatch.setattr(judge_mod, "make_provider",
                        lambda cfg: ScriptedProvider([_verdict_json(), _verdict_json()]))
    st = Storage(":memory:")
    try:
        _seed_run(st, 1)
        asyncio.run(judge_runs.judge_run(st, 1, judge_runs.JUDGE_DEFAULT, force=False))
        status = asyncio.run(judge_runs.judge_run(st, 1, judge_runs.JUDGE_DEFAULT, force=True))
        assert status == "judged"
    finally:
        st.close()
