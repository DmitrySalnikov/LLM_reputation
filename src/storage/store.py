from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone

from src.core.config import EpisodeCfg
from src.games.base import PairingRecord
from src.judge import JudgeVerdict
from src.matchmaking import RoundPlan
from src.population import Population
from src.storage.schema import init_schema


def _run_id(cfg: EpisodeCfg) -> str:
    d = asdict(cfg)
    d.pop("judge", None)            # судья — аналитика, не геймплей: не влияет на run_id
    canon = json.dumps(d, sort_keys=True)               # stable across processes
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


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
        self._run_id: str | None = None

    def begin(self, cfg: EpisodeCfg, pop: Population, name: str | None = None) -> str:
        """Step 0: write the run + agents; returns run_id (hash of the config).
        `name` is an optional human label — metadata only, not part of run_id."""
        run_id = _run_id(cfg)
        self._run_id = run_id
        with self._conn:
            self._conn.execute(
                "INSERT INTO runs(run_id, name, config, seed, created_at) VALUES (?,?,?,?,?)",
                (run_id, name, json.dumps(asdict(cfg)), cfg.seed, _now()),
            )
            self._conn.executemany(
                "INSERT INTO agents(run_id, agent_id, persona, provider) VALUES (?,?,?,?)",
                [
                    (run_id, a.id, a.setup.persona, json.dumps(asdict(a.setup.provider_cfg)))
                    for a in pop
                ],
            )
        return run_id

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
                           a_reflection, b_reflection,
                           usage_prompt_tokens, usage_completion_tokens, usage_calls)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        rid, round, pair_idx, rec.a_id, rec.b_id, int(rec.finished),
                        rec.a_number, rec.b_number, rec.a_rationale, rec.b_rationale,
                        rec.outcome, rec.a_payoff, rec.b_payoff, rec.a_predicted, rec.b_predicted,
                        rec.a_reflection, rec.b_reflection,
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
