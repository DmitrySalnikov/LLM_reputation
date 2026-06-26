from __future__ import annotations

import json
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


def _cfg(judge=None, rounds=1, n=2):
    spec = AgentSpec(count=n)
    return EpisodeCfg(
        seed=0, rounds=rounds, matchmaker="random",
        population=PopulationCfg(kind="roster", agents=[spec],
                                 provider=ProviderCfg(base_url="http://x/v1", model="m")),
        game=GameCfg(max_talk_turns=0),
        judge=judge,
    )


def _final_scores(db, rid):
    import sqlite3

    conn = sqlite3.connect(db)
    try:
        return dict(conn.execute(
            "SELECT agent_id, final_score FROM agents WHERE run_id=?", (rid,)).fetchall())
    finally:
        conn.close()


def _judge():
    return JudgeCfg(provider=ProviderCfg(base_url="http://j/v1", model="judge-m"))


# ---- quiet mode (для свипов: research.py гоняет сотни прогонов) ----

async def test_quiet_run_suppresses_narration(tmp_path, capsys):
    db = str(tmp_path / "t.db")
    await runner.run_experiment(_cfg(rounds=1), db, quiet=True)
    out = capsys.readouterr().out
    assert "ROUND" not in out and "FINAL SCOREBOARD" not in out
    assert "Running experiment" not in out


async def test_loud_run_narrates_by_default(tmp_path, capsys):
    db = str(tmp_path / "t.db")
    await runner.run_experiment(_cfg(rounds=1), db)        # quiet=False по умолчанию
    out = capsys.readouterr().out
    assert "ROUND" in out and "FINAL SCOREBOARD" in out


async def test_quiet_resume_suppresses_narration(tmp_path, capsys):
    db = str(tmp_path / "t.db")
    rid = await runner.run_experiment(_cfg(rounds=2), db, quiet=True)
    capsys.readouterr()
    await runner.resume_run(rid, db, rounds=4, quiet=True)  # extend 2 -> 4 тихо
    out = capsys.readouterr().out
    assert "ROUND" not in out and "Resuming run" not in out


async def test_quiet_run_still_persists(tmp_path):
    db = str(tmp_path / "t.db")
    rid = await runner.run_experiment(_cfg(rounds=1, n=2), db, quiet=True)
    assert _final_scores(db, rid)                          # данные записаны, несмотря на тишину


async def test_rerunning_config_creates_new_run_and_leaves_existing(tmp_path):
    import random
    import sqlite3

    from src.population import make_population
    from src.storage import Storage

    db = str(tmp_path / "t.db")
    cfg = _cfg()
    # уже есть оборванный прогон того же конфига (begin без finish)
    st = Storage(db)
    pop = make_population(cfg.population, context_window=cfg.context_window).build(random.Random(cfg.seed))
    rid = st.begin(cfg, pop)
    st.close()
    await pop.aclose()

    run_id = await runner.run_experiment(cfg, db)      # дедупа нет -> новый прогон, старый не трогаем
    assert run_id != rid

    conn = sqlite3.connect(db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 2                            # оба сохранены
        assert conn.execute("SELECT finished_at FROM runs WHERE run_id=?", (rid,)).fetchone()[0] is None       # оборванный так и не доигран
        assert conn.execute("SELECT finished_at FROM runs WHERE run_id=?", (run_id,)).fetchone()[0] is not None  # новый доигран
    finally:
        conn.close()


async def test_rerunning_finished_config_creates_second_run(tmp_path):
    import sqlite3

    db = str(tmp_path / "t.db")
    cfg = _cfg()
    first = await runner.run_experiment(cfg, db)       # доигран до конца
    second = await runner.run_experiment(cfg, db)      # тот же конфиг -> новый номер, не «nothing to do»
    assert first != second

    conn = sqlite3.connect(db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 2
        hs = [h for (h,) in conn.execute("SELECT config_hash FROM runs")]
        assert hs[0] == hs[1]                          # одинаковый конфиг -> одинаковый config_hash
    finally:
        conn.close()


# ---- resume / extend (по явному номеру прогона) ----

async def test_resume_unfinished_run_completes_to_configured_rounds(tmp_path):
    # Эталон — прямой прогон на 4 раунда. Затем строим ОБОРВАННЫЙ прогон того же конфига
    # (сыграно 2 из 4, без finish) и возобновляем — итог должен совпасть с эталоном.
    import random

    from src.core.orchestrator import run_episode
    from src.population import make_population
    from src.storage import Storage

    cfg = _cfg(n=3, rounds=4)
    ref_db = str(tmp_path / "ref.db")
    ref_id = await runner.run_experiment(cfg, ref_db)
    ref_scores = _final_scores(ref_db, ref_id)

    db = str(tmp_path / "t.db")
    pop = make_population(cfg.population, context_window=cfg.context_window).build(random.Random(cfg.seed))
    st = Storage(db)
    rid = st.begin(cfg, pop)                                  # config хранит rounds=4
    await run_episode(replace(cfg, rounds=2), pop, observer=st.observe)   # но сыграно лишь 2, без finish
    st.close()
    await pop.aclose()

    done = await runner.resume_run(rid, db)                  # без rounds -> доиграть до сохранённых 4
    assert done == rid
    assert _final_scores(db, rid) == ref_scores              # возобновлённый == прямому прогону


async def test_extend_finished_run_matches_straight_run(tmp_path):
    # Доигранный прогон на 2 раунда, доращённый до 4, должен совпасть с прямым прогоном на 4.
    cfg2 = _cfg(n=3, rounds=2)
    cfg4 = _cfg(n=3, rounds=4)
    ref_db = str(tmp_path / "ref.db")
    ref_id = await runner.run_experiment(cfg4, ref_db)
    ref_scores = _final_scores(ref_db, ref_id)

    db = str(tmp_path / "t.db")
    rid = await runner.run_experiment(cfg2, db)              # доигран до 2
    same = await runner.resume_run(rid, db, rounds=4)        # extend до 4
    assert same == rid
    assert _final_scores(db, rid) == ref_scores
    import sqlite3
    conn = sqlite3.connect(db)
    try:
        # rounds в сохранённом config вырос до 4, finished_at снова проставлен
        assert json.loads(conn.execute("SELECT config FROM runs WHERE run_id=?", (rid,)).fetchone()[0])["rounds"] == 4
        assert conn.execute("SELECT finished_at FROM runs WHERE run_id=?", (rid,)).fetchone()[0] is not None
        assert conn.execute("SELECT COUNT(*) FROM rounds WHERE run_id=?", (rid,)).fetchone()[0] == 4
    finally:
        conn.close()


async def test_extend_honors_schedule_patch_for_new_rounds(tmp_path):
    # Прогон с patch, вступающим в силу на раунде 3; короткий прогон (2 раунда) до него не доходит,
    # extend до 4 должен сыграть раунды 3–4 уже по патчу и совпасть с прямым прогоном того же
    # расписания на 4 раунда.
    from src.core.config import ChangePoint

    sched = (ChangePoint(from_round=3, patch={"game": {"payoffs": {"R": 7}}}),)
    cfg4 = replace(_cfg(n=3, rounds=4), schedule=sched)
    cfg2 = replace(cfg4, rounds=2)

    ref_db = str(tmp_path / "ref.db")
    ref_id = await runner.run_experiment(cfg4, ref_db)
    ref_scores = _final_scores(ref_db, ref_id)

    db = str(tmp_path / "t.db")
    rid = await runner.run_experiment(cfg2, db)         # сыграно 2 (патч ещё не активен)
    same = await runner.resume_run(rid, db, rounds=4)   # extend до 4 -> раунды 3–4 по патчу
    assert same == rid
    assert _final_scores(db, rid) == ref_scores


async def test_resume_already_complete_is_noop(tmp_path, capsys):
    db = str(tmp_path / "t.db")
    rid = await runner.run_experiment(_cfg(rounds=2), db)
    capsys.readouterr()
    res = await runner.resume_run(rid, db)                   # уже сыграно 2 из 2
    out = capsys.readouterr().out.lower()
    assert res == rid
    assert "nothing to do" in out
    import sqlite3
    conn = sqlite3.connect(db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM rounds WHERE run_id=?", (rid,)).fetchone()[0] == 2  # без новых раундов
    finally:
        conn.close()


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
