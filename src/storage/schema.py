from __future__ import annotations

import sqlite3

# Normalized L1 schema (one DB, many runs). Configs are JSON (runs.config,
# agents.provider); everything operational is normalized. See agent-games-logger-plan §4.
SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    name        TEXT,
    config      TEXT NOT NULL,
    seed        INTEGER NOT NULL,
    created_at  TEXT NOT NULL,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS agents (
    run_id      TEXT NOT NULL,
    agent_id    TEXT NOT NULL,
    persona     TEXT NOT NULL,
    provider    TEXT NOT NULL,
    final_score REAL,
    PRIMARY KEY (run_id, agent_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS rounds (
    run_id    TEXT NOT NULL,
    round_idx INTEGER NOT NULL,
    PRIMARY KEY (run_id, round_idx),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS idle (
    run_id    TEXT NOT NULL,
    round_idx INTEGER NOT NULL,
    agent_id  TEXT NOT NULL,
    PRIMARY KEY (run_id, round_idx, agent_id),
    FOREIGN KEY (run_id, round_idx) REFERENCES rounds(run_id, round_idx),
    FOREIGN KEY (run_id, agent_id)  REFERENCES agents(run_id, agent_id)
);

CREATE TABLE IF NOT EXISTS pairings (
    run_id      TEXT NOT NULL,
    round_idx   INTEGER NOT NULL,
    pair_idx    INTEGER NOT NULL,
    a_id        TEXT NOT NULL,
    b_id        TEXT NOT NULL,
    a_number    INTEGER NOT NULL,
    b_number    INTEGER NOT NULL,
    a_rationale TEXT,
    b_rationale TEXT,
    a_outcome   TEXT NOT NULL,
    a_payoff    REAL NOT NULL,
    b_payoff    REAL NOT NULL,
    a_predicted INTEGER,                    -- prediction strategy: a's guess of b's number (NULL for direct)
    b_predicted INTEGER,
    a_reflection TEXT,                      -- post-game reflection (NULL when game.reflection=false)
    b_reflection TEXT,
    usage_prompt_tokens     INTEGER NOT NULL,
    usage_completion_tokens INTEGER NOT NULL,
    usage_calls             INTEGER NOT NULL,
    PRIMARY KEY (run_id, round_idx, pair_idx),
    FOREIGN KEY (run_id, round_idx) REFERENCES rounds(run_id, round_idx),
    FOREIGN KEY (run_id, a_id) REFERENCES agents(run_id, agent_id),
    FOREIGN KEY (run_id, b_id) REFERENCES agents(run_id, agent_id)
);

CREATE TABLE IF NOT EXISTS messages (
    run_id    TEXT NOT NULL,
    round_idx INTEGER NOT NULL,
    pair_idx  INTEGER NOT NULL,
    turn_idx  INTEGER NOT NULL,
    speaker   TEXT NOT NULL,
    text      TEXT NOT NULL,
    ready     INTEGER NOT NULL,
    PRIMARY KEY (run_id, round_idx, pair_idx, turn_idx),
    FOREIGN KEY (run_id, round_idx, pair_idx) REFERENCES pairings(run_id, round_idx, pair_idx),
    FOREIGN KEY (run_id, speaker) REFERENCES agents(run_id, agent_id)
);
"""


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
