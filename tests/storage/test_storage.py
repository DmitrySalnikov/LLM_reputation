from __future__ import annotations

import json
import random
from dataclasses import replace

import pytest

from src.core import orchestrator as orch
from src.core.config import AgentSpec, EpisodeCfg, GameCfg, JudgeCfg, PopulationCfg, ProviderCfg
from src.judge import JudgeVerdict, MessageRef
from src.games.base import PairingRecord
from src.matchmaking.base import RoundPlan
from src.population import base as popbase
from src.population import make_population
from src.providers.base import Completion
from src.storage import Storage


class FixedProvider:
    def __init__(self, cfg):
        self.cfg = cfg

    async def complete(self, **kw):
        return Completion(text='{"number": 4, "rationale": "r"}', prompt_tokens=2, completion_tokens=3, raw={})

    async def aclose(self):
        pass


@pytest.fixture(autouse=True)
def _fake_providers(monkeypatch):
    monkeypatch.setattr(popbase, "make_provider", lambda cfg: FixedProvider(cfg))


def _cfg(seed=0, n=3, rounds=2):
    spec = AgentSpec(persona="p", provider=ProviderCfg(base_url="http://x/v1", model="m"), count=n)
    return EpisodeCfg(
        seed=seed, rounds=rounds, matchmaker="random",
        population=PopulationCfg(kind="roster", agents=[spec]),
        game=GameCfg(max_talk_turns=0),
    )


def _pop(cfg):
    return make_population(cfg.population, context_window=cfg.context_window).build(random.Random(cfg.seed))


def _store(tmp_path, name="t.db"):
    return Storage(str(tmp_path / name))


# ---- Slice 1: begin + run_id ----

def test_begin_writes_runs_and_agents(tmp_path):
    cfg = _cfg(n=3)
    st = _store(tmp_path)
    try:
        run_id = st.begin(cfg, _pop(cfg))
        c = st._conn
        assert c.execute("SELECT run_id, seed FROM runs").fetchall() == [(run_id, 0)]
        agents = c.execute("SELECT agent_id, persona FROM agents ORDER BY agent_id").fetchall()
        assert [a for a, _ in agents] == ["A1", "A2", "A3"]
        assert json.loads(c.execute("SELECT config FROM runs").fetchone()[0])["matchmaker"] == "random"
        assert json.loads(c.execute("SELECT provider FROM agents LIMIT 1").fetchone()[0])["model"] == "m"
    finally:
        st.close()


def test_run_id_deterministic_and_config_sensitive(tmp_path):
    a, b, d = _store(tmp_path, "a.db"), _store(tmp_path, "b.db"), _store(tmp_path, "d.db")
    try:
        id_a = a.begin(_cfg(seed=1), _pop(_cfg(seed=1)))
        id_b = b.begin(_cfg(seed=1), _pop(_cfg(seed=1)))
        id_d = d.begin(_cfg(seed=2), _pop(_cfg(seed=2)))
        assert id_a == id_b          # same config -> same run_id
        assert id_a != id_d          # different seed -> different run_id
    finally:
        a.close(); b.close(); d.close()


# ---- Slice 2: observe (one txn per round) ----

def test_observe_writes_round_tables(tmp_path):
    cfg = _cfg(n=3)
    st = _store(tmp_path)
    try:
        rid = st.begin(cfg, _pop(cfg))      # agents A1..A3 exist (FK targets)
        rec = PairingRecord(
            round=0, a_id="A1", b_id="A2",
            transcript=[
                {"speaker": "A1", "text": "hi", "ready": False},
                {"speaker": "A2", "text": "ok", "ready": True},
            ],
            a_number=4, b_number=4, a_rationale="ra", b_rationale="rb",
            outcome="CC", a_payoff=3.0, b_payoff=3.0,
            usage={"prompt_tokens": 10, "completion_tokens": 5, "calls": 4},
        )
        st.observe(0, RoundPlan(pairings=[("A1", "A2")], idle=["A3"], events=[]), [rec])
        c = st._conn
        assert c.execute("SELECT round_idx FROM rounds").fetchall() == [(0,)]
        assert c.execute("SELECT agent_id FROM idle").fetchall() == [("A3",)]
        p = c.execute("SELECT a_id, b_id, a_outcome, usage_calls FROM pairings").fetchone()
        assert p == ("A1", "A2", "CC", 4)
        msgs = c.execute("SELECT turn_idx, speaker, text, ready FROM messages ORDER BY turn_idx").fetchall()
        assert msgs == [(0, "A1", "hi", 0), (1, "A2", "ok", 1)]
        assert rid  # non-empty
    finally:
        st.close()


# ---- Slices 1+2 end-to-end: Storage as the orchestrator observer ----

async def test_logs_full_episode(tmp_path):
    cfg = _cfg(n=3, rounds=2)
    pop = _pop(cfg)
    st = _store(tmp_path)
    rid = st.begin(cfg, pop)
    try:
        await orch.run_episode(cfg, pop, observer=st.observe)
        st.finish(pop)
    finally:
        await pop.aclose()
    c = st._conn
    try:
        assert c.execute("SELECT finished_at FROM runs WHERE run_id=?", (rid,)).fetchone()[0] is not None
        assert c.execute("SELECT COUNT(*) FROM rounds").fetchone()[0] == 2
        assert c.execute("SELECT COUNT(*) FROM pairings").fetchone()[0] == 2     # 1 pair/round
        assert c.execute("SELECT COUNT(*) FROM idle").fetchone()[0] == 2         # 1 idle/round (N=3)
        assert c.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0     # max_talk_turns=0
        assert all(o == "CC" for (o,) in c.execute("SELECT a_outcome FROM pairings"))
        stored = dict(c.execute("SELECT agent_id, final_score FROM agents").fetchall())
        assert stored == {a.id: a.score for a in pop}
        # integrity: idle gives each round's idle agent +idle_payoff, sum matches
        assert sum(stored.values()) == pytest.approx(14.0)
    finally:
        st.close()


# ---- Slice 4: LLM judge — run_id stability and verdict persistence ----

def _judge_cfg():
    return JudgeCfg(provider=ProviderCfg(base_url="http://j/v1", model="judge-m"))


def test_run_id_ignores_judge_block(tmp_path):
    base = _cfg(seed=1)
    judged = replace(base, judge=_judge_cfg())
    a, b = _store(tmp_path, "a.db"), _store(tmp_path, "b.db")
    try:
        assert a.begin(base, _pop(base)) == b.begin(judged, _pop(judged))  # судья — аналитика, не геймплей
    finally:
        a.close(); b.close()


def test_judge_config_still_persisted_in_runs(tmp_path):
    cfg = replace(_cfg(), judge=_judge_cfg())
    st = _store(tmp_path)
    try:
        st.begin(cfg, _pop(cfg))
        stored = json.loads(st._conn.execute("SELECT config FROM runs").fetchone()[0])
        assert stored["judge"]["provider"]["model"] == "judge-m"
    finally:
        st.close()


def test_save_verdict_roundtrip(tmp_path):
    cfg = _cfg()
    st = _store(tmp_path)
    try:
        st.begin(cfg, _pop(cfg))
        st.save_verdict(
            JudgeVerdict(emerged=True, explanation="gossip observed",
                         evidence=[MessageRef(round=0, pair=0, turn=1)]),
            model="judge-m",
        )
        row = st._conn.execute(
            "SELECT emerged, explanation, evidence, model, created_at FROM judge_verdicts"
        ).fetchone()
        assert row[0] == 1
        assert row[1] == "gossip observed"
        assert json.loads(row[2]) == [{"round": 0, "pair": 0, "turn": 1}]
        assert row[3] == "judge-m"
        assert row[4]                                  # created_at заполнен
    finally:
        st.close()
