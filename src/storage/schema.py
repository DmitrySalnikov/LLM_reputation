from __future__ import annotations

import sqlite3

# Normalized L1 schema (one DB, many runs). Configs are JSON (runs.config,
# agents.provider); everything operational is normalized. See agent-games-logger-plan §4.
#
# run_id — целочисленный автоинкремент (человекочитаемые 1, 2, 3 …), а НЕ хеш конфига.
# Идентичность прогона — это его номер; «дизайн» же хешируется в runs.config_hash (хеш
# конфига без `judge` и без `rounds`) — для группировки прогонов одного дизайна в семью
# (повторы и продолжения разной длины делят config_hash). replay принимает и число, и хеш.
SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT,
    config      TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    seed        INTEGER NOT NULL,
    created_at  TEXT NOT NULL,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS agents (
    run_id        INTEGER NOT NULL,
    agent_id      TEXT NOT NULL,
    system_prompt TEXT,
    provider      TEXT NOT NULL,
    final_score   REAL,
    PRIMARY KEY (run_id, agent_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS rounds (
    run_id    INTEGER NOT NULL,
    round_idx INTEGER NOT NULL,
    PRIMARY KEY (run_id, round_idx),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS idle (
    run_id    INTEGER NOT NULL,
    round_idx INTEGER NOT NULL,
    agent_id  TEXT NOT NULL,
    PRIMARY KEY (run_id, round_idx, agent_id),
    FOREIGN KEY (run_id, round_idx) REFERENCES rounds(run_id, round_idx) ON DELETE CASCADE,
    FOREIGN KEY (run_id, agent_id)  REFERENCES agents(run_id, agent_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS pairings (
    run_id      INTEGER NOT NULL,
    round_idx   INTEGER NOT NULL,
    pair_idx    INTEGER NOT NULL,
    a_id        TEXT NOT NULL,
    b_id        TEXT NOT NULL,
    finished    INTEGER NOT NULL DEFAULT 1,  -- 1 = доиграна; 0 = сорвана LLM-сбоем (результатов нет)
    a_number    INTEGER,                     -- результаты NULL, если finished=0 (см. CHECK)
    b_number    INTEGER,
    a_rationale TEXT,
    b_rationale TEXT,
    a_outcome   TEXT,
    a_payoff    REAL,
    b_payoff    REAL,
    a_predicted INTEGER,                    -- prediction strategy: a's guess of b's number (NULL for direct)
    b_predicted INTEGER,
    a_reflection TEXT,                      -- post-game reflection (NULL when game.reflection=false)
    b_reflection TEXT,
    a_notes     TEXT,                       -- memory notes после раунда (NULL, если не свёртывали)
    b_notes     TEXT,
    usage_prompt_tokens     INTEGER,
    usage_completion_tokens INTEGER,
    usage_calls             INTEGER,
    PRIMARY KEY (run_id, round_idx, pair_idx),
    FOREIGN KEY (run_id, round_idx) REFERENCES rounds(run_id, round_idx) ON DELETE CASCADE,
    FOREIGN KEY (run_id, a_id) REFERENCES agents(run_id, agent_id) ON DELETE CASCADE,
    FOREIGN KEY (run_id, b_id) REFERENCES agents(run_id, agent_id) ON DELETE CASCADE,
    CHECK (finished = 0 OR a_number IS NOT NULL),   -- доиграна ⇒ результат есть
    CHECK (finished = 1 OR a_number IS NULL)        -- сорвана  ⇒ результат пуст
);

CREATE TABLE IF NOT EXISTS messages (
    run_id    INTEGER NOT NULL,
    round_idx INTEGER NOT NULL,
    pair_idx  INTEGER NOT NULL,
    turn_idx  INTEGER NOT NULL,
    speaker   TEXT NOT NULL,
    text      TEXT NOT NULL,
    ready     INTEGER NOT NULL,
    PRIMARY KEY (run_id, round_idx, pair_idx, turn_idx),
    FOREIGN KEY (run_id, round_idx, pair_idx) REFERENCES pairings(run_id, round_idx, pair_idx) ON DELETE CASCADE,
    FOREIGN KEY (run_id, speaker) REFERENCES agents(run_id, agent_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS llm_calls (
    run_id        INTEGER NOT NULL,
    round_idx     INTEGER NOT NULL,
    pair_idx      INTEGER NOT NULL,
    call_idx      INTEGER NOT NULL,   -- порядок вызова внутри пары (порядок исполнения)
    agent_id      TEXT    NOT NULL,   -- кто вызывал
    phase         TEXT    NOT NULL,   -- talk | decide | predict | reflect | note
    turn_idx      INTEGER,            -- NULL кроме TALK; FK на конкретную реплику messages
    attempt       INTEGER NOT NULL,   -- парс-попытка Agent.act (1..3)
    http_attempt  INTEGER NOT NULL,   -- сетевой ретрай внутри complete() (1..5)
    status        TEXT    NOT NULL,   -- ok | parse_error | bad_json | bad_shape | http_error | server_error | network
    status_code   INTEGER,            -- HTTP-код попытки (NULL при сетевой ошибке)
    request       TEXT    NOT NULL,   -- ДОСЛОВНЫЙ payload (JSON)
    response      TEXT,               -- извлечённый текст (только на финальной ok-попытке)
    response_raw  TEXT,               -- ДОСЛОВНОЕ тело resp.text (вкл. тело 5xx); NULL при сетевой ошибке
    error         TEXT,               -- сообщение сбоя
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (run_id, round_idx, pair_idx, call_idx),
    FOREIGN KEY (run_id, round_idx, pair_idx) REFERENCES pairings(run_id, round_idx, pair_idx) ON DELETE CASCADE,
    FOREIGN KEY (run_id, agent_id) REFERENCES agents(run_id, agent_id) ON DELETE CASCADE,
    FOREIGN KEY (run_id, round_idx, pair_idx, turn_idx)
        REFERENCES messages(run_id, round_idx, pair_idx, turn_idx) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS judge_verdicts (
    run_id      INTEGER PRIMARY KEY,
    emerged     INTEGER NOT NULL,
    explanation TEXT NOT NULL,
    evidence    TEXT NOT NULL,      -- JSON: [{"round":0,"pair":1,"turn":2}, ...]
    model       TEXT NOT NULL,      -- модель судьи, для истории
    created_at  TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS ix_llm_calls_agent  ON llm_calls(run_id, agent_id);
CREATE INDEX IF NOT EXISTS ix_llm_calls_status ON llm_calls(run_id, status);
CREATE INDEX IF NOT EXISTS ix_runs_config_hash ON runs(config_hash);
"""


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
