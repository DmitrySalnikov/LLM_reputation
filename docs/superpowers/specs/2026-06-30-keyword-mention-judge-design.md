# Keyword/number mention judge (deterministic, non-LLM)

## Problem

The only post-episode verdict today is the **LLM judge** (`src/judge/judge.py`,
`judge_runs.py`): one model call over the public cheap-talk decides `emerged: bool`.
We want an **alternative judge** that is fully deterministic and uses **no LLM** — it
searches the public cheap-talk for a user-supplied term (a number or word) and reports
how often it is mentioned, so a researcher can ask e.g. "how many agents talked about
player 123?" without paying for or trusting a model.

## Requirements

- **No LLM.** The search is hardcoded string matching.
- **Search replies only.** Match against each message's `text` (the agent's reply),
  **never** the `speaker` label / player name. A player whose name equals the term but
  who never types the term scores nothing.
- **Plain substring**, **case-sensitive** matching (`term in text`). `"123"` matches
  inside `"1234"`; `"Trust"` does not match term `"trust"`.
- **Count = distinct speakers.** Per run, the result is the number of distinct speakers
  whose reply text contains the term at least once (a reply containing the term twice
  still counts its speaker once).
- **Output to print + CSV + DB.**
- The existing LLM judge (`JudgeCfg`, `judge_runs.py`, `judge_verdicts`) is **untouched**.

## Design

### 1. Pure counting logic — `src/judge/keyword.py`

Lives inside `src/`, so it does no printing and no persistence (engine convention).

```python
@dataclass(frozen=True)
class KeywordCount:
    """Результат поиска термина в публичном cheap-talk одного эпизода."""
    term: str
    count: int                  # число РАЗНЫХ говорящих, чьи реплики содержат термин
    speakers: tuple[str, ...]   # их id, отсортированные (для трассировки)

def count_mentions(records: list[PairingRecord], term: str) -> KeywordCount:
    ...
```

Algorithm: iterate every message of every record; if `term in msg["text"]`
(case-sensitive substring), add `msg["speaker"]` to a set. `count = len(set)`,
`speakers = tuple(sorted(set))`. Only `text` is inspected; `speaker` is used solely as
the set key, never matched against the term. Works on both `PairingRecord` (live) and
`ReplayRecord` (backfill) since both expose `.transcript` of `{speaker, text, ready}`.

Russian Google-style docstrings; `from __future__ import annotations`.

### 2. DB table — `src/storage/schema.py`

```sql
CREATE TABLE IF NOT EXISTS keyword_counts (
    run_id     INTEGER NOT NULL,
    term       TEXT NOT NULL,
    count      INTEGER NOT NULL,   -- число разных говорящих, упомянувших термин
    speakers   TEXT NOT NULL,      -- JSON-список id говорящих
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, term),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);
```

PK `(run_id, term)` → re-running the same term for a run replaces the prior row.

### 3. Storage method — `Storage.save_keyword_count`

```python
def save_keyword_count(self, count: KeywordCount, *, run_id: int) -> None:
    """Сохранить счётчик упоминаний термина для прогона (upsert по (run_id, term))."""
```

Uses `INSERT OR REPLACE` with `json.dumps(list(count.speakers))` and `_now()`.

### 4. Runner — `keyword_judge.py` (repo root, sibling of `judge_runs.py`)

```
uv run python keyword_judge.py TERM [--db experiment.db] [--csv keyword_counts.csv] \
                               [--design HASH ...] [--exclude-design HASH ...] \
                               [--name LABEL ...] [--exclude-name LABEL ...]
```

- `TERM` — required first positional (first argv element that is not a flag/flag-value).
  Missing term → Russian error + non-zero exit.
- Run selection reuses `filter_from_argv` + `selected_run_ids` (same filters as
  `judge_runs.py`). Transcripts via `reconstruct_records`.
- Per run: `count_mentions` → `save_keyword_count` → collect a row. Prints a per-run
  table (`run_id`, `name`, `count`) and a total line; writes CSV with header
  `run_id,name,term,count`. All print/CSV text in Russian where it is prose.
- `--db` defaults to `experiment.db`; `--csv` defaults to `keyword_counts.csv`.

### Data flow

```
runs in DB ──selected_run_ids──> [run_id...] ──reconstruct_records──> records
records + TERM ──count_mentions──> KeywordCount ──save_keyword_count──> keyword_counts table
                                              └──> stdout table + CSV row
```

## Testing (TDD — failing tests first, AAA, behavioural names)

`tests/judge/test_keyword.py` (pure, no provider/network):

- `should_count_distinct_speakers_not_occurrences` — one speaker repeats the term twice
  in a reply → count 1.
- `should_ignore_speaker_name_and_match_only_reply_text` — a speaker whose **name**
  equals the term but whose text never contains it → count 0; proves names are excluded.
- `should_match_case_sensitively` — term `"trust"` vs text `"Trust"` → count 0.
- `should_match_substring` — term `"123"` inside text `"1234"` → count 1.
- `should_return_zero_for_no_records` / empty transcripts → count 0, empty speakers.

`tests/storage/` (extend or new file):

- `should_upsert_keyword_count_on_repeat_term` — saving the same `(run_id, term)` twice
  replaces, not duplicates (one row, latest count).

## Out of scope

- No aggregate/plot tooling for keyword counts (CSV + DB are enough for now; a future
  `collect`/`plot` analogue can be added if needed).
- No changes to the LLM judge, `JudgeCfg`, YAML judge block, or `judge_verdicts`.
- No live (in-orchestrator) keyword judging — this is a backfill-style runner only,
  matching how the LLM judge backfill (`judge_runs.py`) already works.
