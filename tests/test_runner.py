from __future__ import annotations

from dataclasses import replace

import pytest

from src import runner
from src.core.config import (
    AgentSpec, EpisodeCfg, GameCfg, JudgeCfg, PopulationCfg, ProviderCfg,
)
from src.judge import JudgeError, JudgeVerdict, MessageRef
from src.population import base as popbase
from src.providers.base import Completion


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


def _cfg(judge=None):
    spec = AgentSpec(persona="p", count=2)
    return EpisodeCfg(
        seed=0, rounds=1, matchmaker="random",
        population=PopulationCfg(kind="roster", agents=[spec],
                                 provider=ProviderCfg(base_url="http://x/v1", model="m")),
        game=GameCfg(max_talk_turns=0),
        judge=judge,
    )


def _judge():
    return JudgeCfg(provider=ProviderCfg(base_url="http://j/v1", model="judge-m"))


async def test_unfinished_duplicate_is_deleted_and_rerun(tmp_path, capsys):
    import random
    import sqlite3

    from src.population import make_population
    from src.storage import Storage

    db = str(tmp_path / "t.db")
    cfg = _cfg()
    # создаём оборванный прогон того же конфига: begin без finish
    st = Storage(db)
    pop = make_population(cfg.population, context_window=cfg.context_window).build(random.Random(cfg.seed))
    rid = st.begin(cfg, pop)
    st.close()
    await pop.aclose()

    run_id = await runner.run_experiment(cfg, db)      # дубль оборван -> удалить и прогнать заново
    assert run_id == rid
    assert "deleting it and re-running" in capsys.readouterr().out

    conn = sqlite3.connect(db)
    try:
        assert conn.execute("SELECT finished_at FROM runs WHERE run_id=?", (rid,)).fetchone()[0] is not None
        assert conn.execute("SELECT COUNT(*) FROM runs WHERE run_id=?", (rid,)).fetchone()[0] == 1   # не задвоился
    finally:
        conn.close()


async def test_finished_duplicate_is_left_alone(tmp_path, capsys):
    db = str(tmp_path / "t.db")
    cfg = _cfg()
    await runner.run_experiment(cfg, db)               # доигран до конца
    capsys.readouterr()
    again = await runner.run_experiment(cfg, db)       # тот же конфиг
    assert again is None                               # ничего не делаем
    assert "nothing to do" in capsys.readouterr().out


async def test_no_judge_block_means_no_judge_call(tmp_path, monkeypatch):
    called = []
    async def fake_judge(cfg, records):
        called.append(1)
    monkeypatch.setattr(runner, "judge_episode", fake_judge)
    await runner.run_experiment(_cfg(), str(tmp_path / "t.db"))
    assert called == []


async def test_judge_verdict_printed_and_persisted(tmp_path, monkeypatch, capsys):
    seen = {}
    async def fake_judge(cfg, records):
        seen["model"] = cfg.provider.model
        seen["n_records"] = len(records)
        return JudgeVerdict(emerged=True, explanation="history-based trust",
                            evidence=[MessageRef(round=0, pair=0, turn=0)])
    monkeypatch.setattr(runner, "judge_episode", fake_judge)

    run_id = await runner.run_experiment(_cfg(judge=_judge()), str(tmp_path / "t.db"))

    assert seen == {"model": "judge-m", "n_records": 1}   # судья получил собранные записи
    out = capsys.readouterr().out
    assert "JUDGE VERDICT" in out and "YES" in out and "history-based trust" in out

    import sqlite3
    conn = sqlite3.connect(str(tmp_path / "t.db"))
    try:
        row = conn.execute("SELECT emerged, model FROM judge_verdicts WHERE run_id=?", (run_id,)).fetchone()
        assert row == (1, "judge-m")
    finally:
        conn.close()


async def test_judge_failure_does_not_lose_the_run(tmp_path, monkeypatch, capsys):
    async def failing_judge(cfg, records):
        raise JudgeError("судья вернул неразборчивый ответ после повторной попытки")
    monkeypatch.setattr(runner, "judge_episode", failing_judge)

    run_id = await runner.run_experiment(_cfg(judge=_judge()), str(tmp_path / "t.db"))

    assert run_id is not None                         # эпизод сохранён
    assert "судья не смог вынести вердикт" in capsys.readouterr().out

    import sqlite3
    conn = sqlite3.connect(str(tmp_path / "t.db"))
    try:
        assert conn.execute("SELECT finished_at FROM runs").fetchone()[0] is not None
        assert conn.execute("SELECT COUNT(*) FROM judge_verdicts").fetchone()[0] == 0
    finally:
        conn.close()
