from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from src.core.config import EpisodeCfg
from src.core.memory import Memory, MemoryEntry
from src.games.base import PairingRecord
from src.judge import JudgeVerdict
from src.matchmaking import RoundPlan
from src.population import Population
from src.storage.schema import init_schema

# Исход с точки зрения A -> с точки зрения B (как _FLIP в reputation_pd).
_FLIP = {"CC": "CC", "DD": "DD", "DC": "CD", "CD": "DC"}


@dataclass
class RunState:
    """Снимок прогона, восстановленный из БД для возобновления (resume).

    last_round — номер последнего записанного раунда (0, если их нет). scores — накопленный
    счёт по агенту. memories — восстановленная Memory по агенту (дневник + заметки), готовая
    к наложению на свежесобранную популяцию (A4)."""

    last_round: int
    scores: dict[str, float]
    memories: dict[str, Memory]


def _hash_config_dict(d: dict) -> str:
    """Хеш «дизайна эксперимента»: конфиг без `judge` и без `rounds`.

    `judge` — аналитика, не геймплей. `rounds` исключён намеренно: при per-round rng раунд r
    одинаков независимо от общей длины, поэтому «20 раундов» — это «10 раундов», доигранные
    дальше; число раундов — «докуда досимулировали», а не часть идентичности дизайна. Значит
    прогон, его повторы и его продолжения разной длины делят один config_hash (одна «семья»)."""
    d = dict(d)
    d.pop("judge", None)
    d.pop("rounds", None)
    canon = json.dumps(d, sort_keys=True)               # stable across processes
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


def _config_hash(cfg: EpisodeCfg) -> str:
    """config_hash прогона — хеш дизайна (см. _hash_config_dict). Не идентичность прогона
    (она — целочисленный runs.run_id), а метка для группировки прогонов одного дизайна."""
    return _hash_config_dict(asdict(cfg))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Storage:
    """Persists one episode to SQLite (L1). Subscribes to the orchestrator's observer
    seam — the engine is unchanged. See agent-games-logger-plan.md."""

    def __init__(self, db_path: str):
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        init_schema(self._conn)
        self._run_id: int | None = None

    def begin(self, cfg: EpisodeCfg, pop: Population, name: str | None = None) -> int:
        """Step 0: write the run + agents; returns run_id — целочисленный автоинкремент.

        Каждый вызов создаёт НОВЫЙ прогон (дедупа по конфигу больше нет: повторный запуск
        того же конфига — это новый номер). Метка конфига едет в config_hash. `name` —
        опциональный человеческий ярлык (метаданные)."""
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO runs(name, config, config_hash, seed, created_at) VALUES (?,?,?,?,?)",
                (name, json.dumps(asdict(cfg)), _config_hash(cfg), cfg.seed, _now()),
            )
            run_id = cur.lastrowid
            self._run_id = run_id
            self._conn.executemany(
                "INSERT INTO agents(run_id, agent_id, system_prompt, provider) VALUES (?,?,?,?)",
                [
                    (run_id, a.id, a.setup.system_prompt, json.dumps(asdict(a.setup.provider_cfg)))
                    for a in pop
                ],
            )
        return run_id

    def is_finished(self, run_id: int) -> bool:
        """True, если прогон доигран (проставлен finished_at); False для оборванного/отсутствующего."""
        row = self._conn.execute(
            "SELECT finished_at FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
        return bool(row and row[0])

    def run_config(self, run_id: int) -> str | None:
        """Вернуть сохранённый config (JSON-строка) прогона или None, если его нет.
        Используется при возобновлении: из него восстанавливается EpisodeCfg."""
        row = self._conn.execute(
            "SELECT config FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
        return row[0] if row else None

    def resume(self, run_id: int, cfg: EpisodeCfg) -> None:
        """Подготовить Storage к дозаписи в существующий прогон (resume/extend).

        Запоминает run_id (для observe/finish), снимает finished_at (помечаем «в процессе» —
        если доращивание оборвётся, прогон корректно останется недоигранным) и обновляет
        config (при extend выросло число раундов; config_hash НЕ трогаем — rounds в него не
        входит, семья дизайна сохраняется)."""
        self._run_id = run_id
        with self._conn:
            self._conn.execute(
                "UPDATE runs SET finished_at=NULL, config=? WHERE run_id=?",
                (json.dumps(asdict(cfg)), run_id),
            )

    def delete_run(self, run_id: int) -> None:
        """Удалить прогон и все его строки — дочерние таблицы уходят каскадом
        (FK с ON DELETE CASCADE; PRAGMA foreign_keys=ON выставлен в __init__)."""
        with self._conn:
            self._conn.execute("DELETE FROM runs WHERE run_id=?", (run_id,))

    def load_state(self, run_id: int, idle_payoff: float) -> RunState:
        """Восстановить состояние прогона из БД (чистое чтение) для возобновления.

        По каждому агенту собирает Memory (дневник из messages/pairings, заметки и их границу
        noted_upto из a_notes/b_notes) и накопленный счёт (сумма payoff доигранных пар +
        idle_payoff за пропущенные раунды), плюс номер последнего записанного раунда. Записи
        строятся в порядке раундов, чтобы поле score («счёт ДО раунда») в дневнике совпадало
        с живым прогоном.

        Сорванные пары (finished=0) в память и счёт не входят — как и в живой игре их агенты
        в тот раунд не доиграли. idle_payoff передаётся из конфига (в БД его нет)."""
        c = self._conn
        agent_ids = [r[0] for r in c.execute(
            "SELECT agent_id FROM agents WHERE run_id=? ORDER BY agent_id", (run_id,))]
        memories: dict[str, Memory] = {aid: Memory() for aid in agent_ids}
        running: dict[str, float] = {aid: 0.0 for aid in agent_ids}

        rounds = [r[0] for r in c.execute(
            "SELECT round_idx FROM rounds WHERE run_id=? ORDER BY round_idx", (run_id,))]

        idle_by_round: dict[int, list[str]] = defaultdict(list)
        for ri, aid in c.execute(
                "SELECT round_idx, agent_id FROM idle WHERE run_id=?", (run_id,)):
            idle_by_round[ri].append(aid)

        pair_by_round: dict[int, list] = defaultdict(list)
        for row in c.execute(
                """SELECT round_idx, pair_idx, a_id, b_id, a_number, b_number,
                          a_rationale, b_rationale, a_outcome, a_payoff, b_payoff,
                          a_predicted, b_predicted, a_reflection, b_reflection, a_notes, b_notes
                   FROM pairings WHERE run_id=? AND finished=1
                   ORDER BY round_idx, pair_idx""", (run_id,)):
            pair_by_round[row[0]].append(row)

        for ri in rounds:
            for aid in idle_by_round.get(ri, ()):          # idle: только счёт, без записи в дневник
                if aid in running:
                    running[aid] += idle_payoff
            for row in pair_by_round.get(ri, ()):
                (_, pair_idx, a_id, b_id, a_number, b_number, a_rationale, b_rationale,
                 a_outcome, a_payoff, b_payoff, a_predicted, b_predicted,
                 a_reflection, b_reflection, a_notes, b_notes) = row
                transcript = self._load_transcript(run_id, ri, pair_idx)
                # сторона A — как есть; сторона B — зеркалим перспективу (партнёр, числа, исход)
                self._restore_entry(memories[a_id], running, a_id, ri, b_id, transcript,
                                    a_number, a_rationale, b_number, a_outcome,
                                    a_payoff, b_payoff, a_predicted, a_reflection, a_notes)
                self._restore_entry(memories[b_id], running, b_id, ri, a_id, transcript,
                                    b_number, b_rationale, a_number, _FLIP.get(a_outcome, a_outcome),
                                    b_payoff, a_payoff, b_predicted, b_reflection, b_notes)

        last_round = rounds[-1] if rounds else 0
        return RunState(last_round=last_round, scores=dict(running), memories=memories)

    def _load_transcript(self, run_id: int, round_idx: int, pair_idx: int) -> list[dict]:
        return [
            {"speaker": s, "text": t, "ready": bool(r)}
            for s, t, r in self._conn.execute(
                "SELECT speaker, text, ready FROM messages "
                "WHERE run_id=? AND round_idx=? AND pair_idx=? ORDER BY turn_idx",
                (run_id, round_idx, pair_idx))
        ]

    @staticmethod
    def _restore_entry(memory: Memory, running: dict[str, float], aid: str, round: int,
                       partner: str, transcript: list[dict], my_number, my_rationale,
                       partner_number, outcome, payoff, partner_payoff, my_predicted,
                       my_reflection, notes) -> None:
        # score — счёт ДО этого раунда (как в живом _remember: agent.score - payoff)
        memory.add(MemoryEntry(
            round=round, my_id=aid, partner_id=partner, transcript=transcript,
            my_number=my_number, my_rationale=my_rationale or "",
            partner_number=partner_number, outcome=outcome,
            payoff=payoff, partner_payoff=partner_payoff, score=running[aid],
            my_predicted=my_predicted, my_reflection=my_reflection,
        ))
        running[aid] += payoff
        if notes is not None:                              # раунд, на котором агент свернул заметки
            memory.set_notes(notes)

    def observe(self, round: int, plan: RoundPlan, recs: list[PairingRecord]) -> None:
        """Step R: one transaction per round — rounds + idle + pairings + messages.
        This is the orchestrator observer (sync; sqlite3 is synchronous)."""
        rid = self._run_id
        with self._conn:
            self._conn.execute(
                "INSERT INTO rounds(run_id, round_idx) VALUES (?,?)", (rid, round)
            )
            self._conn.executemany(
                "INSERT INTO idle(run_id, round_idx, agent_id) VALUES (?,?,?)",
                [(rid, round, aid) for aid in plan.idle],
            )
            for pair_idx, rec in enumerate(recs):
                u = rec.usage or {}
                self._conn.execute(
                    """INSERT INTO pairings(
                           run_id, round_idx, pair_idx, a_id, b_id, finished,
                           a_number, b_number, a_rationale, b_rationale,
                           a_outcome, a_payoff, b_payoff, a_predicted, b_predicted,
                           a_reflection, b_reflection, a_notes, b_notes,
                           usage_prompt_tokens, usage_completion_tokens, usage_calls)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        rid, round, pair_idx, rec.a_id, rec.b_id, int(rec.finished),
                        rec.a_number, rec.b_number, rec.a_rationale, rec.b_rationale,
                        rec.outcome, rec.a_payoff, rec.b_payoff, rec.a_predicted, rec.b_predicted,
                        rec.a_reflection, rec.b_reflection, rec.a_notes, rec.b_notes,
                        u.get("prompt_tokens"), u.get("completion_tokens"), u.get("calls"),
                    ),
                )
                self._conn.executemany(
                    """INSERT INTO messages(run_id, round_idx, pair_idx, turn_idx, speaker, text, ready)
                       VALUES (?,?,?,?,?,?,?)""",
                    [
                        (rid, round, pair_idx, ti, t["speaker"], t["text"], int(bool(t["ready"])))
                        for ti, t in enumerate(rec.transcript)
                    ],
                )
                # L2: сырые вызовы LLM (по одной строке на HTTP-попытку), call_idx — порядок
                self._conn.executemany(
                    """INSERT INTO llm_calls(
                           run_id, round_idx, pair_idx, call_idx, agent_id, phase, turn_idx,
                           attempt, http_attempt, status, status_code,
                           request, response, response_raw, error,
                           prompt_tokens, completion_tokens)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    [
                        (rid, round, pair_idx, call_idx, c.agent_id, c.phase, c.turn_idx,
                         c.attempt, c.http_attempt, c.status, c.status_code,
                         json.dumps(c.request), c.response, c.response_raw, c.error,
                         c.prompt_tokens, c.completion_tokens)
                        for call_idx, c in enumerate(rec.llm_calls)
                    ],
                )

    def finish(self, pop: Population) -> None:
        """Step F: stamp finished_at and write each agent's final score."""
        rid = self._run_id
        with self._conn:
            self._conn.execute(
                "UPDATE runs SET finished_at=? WHERE run_id=?", (_now(), rid)
            )
            self._conn.executemany(
                "UPDATE agents SET final_score=? WHERE run_id=? AND agent_id=?",
                [(a.score, rid, a.id) for a in pop],
            )

    def save_verdict(self, verdict: JudgeVerdict, *, model: str) -> None:
        """Шаг J: сохранить вердикт LLM-судьи (одна строка на run)."""
        with self._conn:
            self._conn.execute(
                """INSERT INTO judge_verdicts(run_id, emerged, explanation, evidence, model, created_at)
                   VALUES (?,?,?,?,?,?)""",
                (
                    self._run_id,
                    int(verdict.emerged),
                    verdict.explanation,
                    json.dumps([asdict(e) for e in verdict.evidence]),
                    model,
                    _now(),
                ),
            )

    def close(self) -> None:
        self._conn.close()
