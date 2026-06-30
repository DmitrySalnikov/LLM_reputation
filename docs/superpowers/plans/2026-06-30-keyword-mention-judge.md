# Keyword/Number Mention Judge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic, non-LLM judge that counts, per run, the number of distinct speakers whose public reply text contains a user-supplied term (number or word), with output to print + CSV + a new DB table.

**Architecture:** Pure counting logic in `src/judge/keyword.py` (no I/O, engine convention), a new `keyword_counts` SQLite table with a `Storage.save_keyword_count` upsert, and a root runner `keyword_judge.py` that selects runs (reusing the existing `src/stats/selection.py` filters), reconstructs transcripts (`reconstruct_records`), counts, persists, prints, and writes CSV. The LLM judge is left completely untouched.

**Tech Stack:** Python 3.12, sqlite3, pytest (auto async mode, `pythonpath = ["."]`), `uv` for running.

---

## Conventions (apply to every task)

- `from __future__ import annotations` at the top of every new `src/` module.
- Russian Google-style docstrings; Russian `print`/error text. Keep English terms
  (cheap-talk, term, run) untranslated. LLM-facing text — none here (no prompts).
- TDD: write the failing test first, watch it fail, then minimal code.
- Run tests with `uv run pytest`. Tests import `src.*` directly (`pythonpath = ["."]`).
- Commit after each green task.

## File Structure

- **Create** `src/judge/keyword.py` — `KeywordCount` dataclass + `count_mentions`. One job: turn records + term into a distinct-speaker count. No I/O.
- **Modify** `src/judge/__init__.py` — export `KeywordCount`, `count_mentions`.
- **Modify** `src/storage/schema.py` — add `keyword_counts` table.
- **Modify** `src/storage/store.py` — import `KeywordCount`, add `save_keyword_count`.
- **Create** `keyword_judge.py` (repo root) — CLI runner, glue over tested pieces.
- **Create** `tests/judge/test_keyword.py` — counting behaviour.
- **Create** `tests/storage/test_keyword_counts.py` — upsert/replace behaviour.

---

## Task 1: Pure counting logic (`count_mentions`)

**Files:**
- Create: `src/judge/keyword.py`
- Test: `tests/judge/test_keyword.py`
- Modify: `src/judge/__init__.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/judge/test_keyword.py`:

```python
from __future__ import annotations

from src.games.base import PairingRecord
from src.judge.keyword import KeywordCount, count_mentions


def _rec(round=0, a="A1", b="A2", transcript=None):
    """Запись пары только с публичным transcript (остальное не важно для подсчёта)."""
    return PairingRecord(round=round, a_id=a, b_id=b, transcript=transcript or [])


def test_should_count_distinct_speakers_not_occurrences():
    # один говорящий повторяет термин дважды в одной реплике -> 1
    rec = _rec(transcript=[{"speaker": "A1", "text": "123 and again 123", "ready": False}])
    result = count_mentions([rec], "123")
    assert result.count == 1
    assert result.speakers == ("A1",)


def test_should_ignore_speaker_name_and_match_only_reply_text():
    # говорящий назван термином, но в его тексте термина нет -> 0 (имена не учитываются)
    rec = _rec(transcript=[{"speaker": "123", "text": "hello there", "ready": False}])
    result = count_mentions([rec], "123")
    assert result.count == 0
    assert result.speakers == ()


def test_should_match_case_sensitively():
    rec = _rec(transcript=[{"speaker": "A1", "text": "Trust me", "ready": False}])
    result = count_mentions([rec], "trust")
    assert result.count == 0


def test_should_match_substring():
    rec = _rec(transcript=[{"speaker": "A1", "text": "pick 1234 now", "ready": False}])
    result = count_mentions([rec], "123")
    assert result.count == 1


def test_should_return_zero_for_no_records():
    result = count_mentions([], "123")
    assert result == KeywordCount(term="123", count=0, speakers=())


def test_should_count_each_distinct_speaker_once_across_records():
    recs = [
        _rec(round=0, transcript=[{"speaker": "A1", "text": "trust 7", "ready": False},
                                  {"speaker": "A2", "text": "no", "ready": False}]),
        _rec(round=1, transcript=[{"speaker": "A1", "text": "trust again", "ready": False},
                                  {"speaker": "A3", "text": "i trust you", "ready": False}]),
    ]
    result = count_mentions(recs, "trust")
    assert result.count == 2
    assert result.speakers == ("A1", "A3")   # отсортированы, A1 не задвоен
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/judge/test_keyword.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.judge.keyword'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/judge/keyword.py`:

```python
"""Детерминированный судья: подсчёт упоминаний термина в публичном cheap-talk.

Альтернатива LLM-судье. Никакого LLM — ищем подстроку (с учётом регистра) в ТЕКСТЕ
реплик. Имена говорящих (player names) НЕ учитываются: поле speaker используется лишь
как ключ множества, но никогда не сопоставляется с термином. Результат эпизода —
число РАЗНЫХ говорящих, чьи реплики содержат термин хотя бы раз.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.games.base import PairingRecord


@dataclass(frozen=True)
class KeywordCount:
    """Результат поиска термина в публичном cheap-talk одного эпизода.

    Attributes:
        term: Искомая подстрока (число или слово).
        count: Число разных говорящих, чьи реплики содержат термин.
        speakers: Их id, отсортированные (для трассировки).
    """

    term: str
    count: int
    speakers: tuple[str, ...]


def count_mentions(records: list[PairingRecord], term: str) -> KeywordCount:
    """Подсчитать разных говорящих, упомянувших термин в своих репликах.

    Поиск подстроки с учётом регистра (`term in text`). Проверяется только text
    реплики; speaker не сопоставляется с термином (имена не учитываются).

    Args:
        records: Записи пар эпизода; берётся только публичный transcript каждой
            ({speaker, text, ready}). Подходят и PairingRecord, и ReplayRecord.
        term: Искомая подстрока.

    Returns:
        KeywordCount с числом разных говорящих и их отсортированными id.
    """
    speakers: set[str] = set()
    for rec in records:
        for msg in rec.transcript:
            if term in msg["text"]:
                speakers.add(msg["speaker"])
    return KeywordCount(term=term, count=len(speakers), speakers=tuple(sorted(speakers)))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/judge/test_keyword.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Export from the judge package**

Modify `src/judge/__init__.py` to:

```python
from src.judge.base import JudgeError, JudgeVerdict, MessageRef
from src.judge.judge import judge_episode
from src.judge.keyword import KeywordCount, count_mentions

__all__ = ["JudgeError", "JudgeVerdict", "MessageRef", "judge_episode",
           "KeywordCount", "count_mentions"]
```

- [ ] **Step 6: Run the full judge test package**

Run: `uv run pytest tests/judge/ -v`
Expected: PASS (existing judge tests + new keyword tests).

- [ ] **Step 7: Commit**

```bash
git add src/judge/keyword.py src/judge/__init__.py tests/judge/test_keyword.py
git commit -m "feat: deterministic keyword mention counter (distinct speakers)"
```

---

## Task 2: Persist counts (`keyword_counts` table + `save_keyword_count`)

**Files:**
- Modify: `src/storage/schema.py` (add table after `judge_verdicts`, before the indexes)
- Modify: `src/storage/store.py` (import + new method)
- Test: `tests/storage/test_keyword_counts.py`

- [ ] **Step 1: Write the failing test**

Create `tests/storage/test_keyword_counts.py`:

```python
from __future__ import annotations

from src.judge import KeywordCount
from src.storage import Storage


def _insert_run(st, run_id=1):
    st.conn.execute(
        "INSERT INTO runs(run_id, name, config, config_hash, seed, created_at, finished_at) "
        "VALUES (?, 'demo', '{}', 'd1', 0, '2026-01-01T00:00:00', '2026-01-01T01:00:00')",
        (run_id,),
    )
    st.conn.commit()


def test_should_save_keyword_count_row():
    st = Storage(":memory:")
    try:
        _insert_run(st, 1)
        st.save_keyword_count(KeywordCount(term="123", count=2, speakers=("A1", "A3")), run_id=1)
        row = st.conn.execute(
            "SELECT term, count, speakers FROM keyword_counts WHERE run_id=1"
        ).fetchone()
        assert row == ("123", 2, '["A1", "A3"]')
    finally:
        st.close()


def test_should_upsert_keyword_count_on_repeat_term():
    # повторный (run_id, term) заменяет строку, не дублирует
    st = Storage(":memory:")
    try:
        _insert_run(st, 1)
        st.save_keyword_count(KeywordCount(term="123", count=1, speakers=("A1",)), run_id=1)
        st.save_keyword_count(KeywordCount(term="123", count=5, speakers=("A1", "A2")), run_id=1)
        rows = st.conn.execute(
            "SELECT count FROM keyword_counts WHERE run_id=1 AND term='123'"
        ).fetchall()
        assert rows == [(5,)]
    finally:
        st.close()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/storage/test_keyword_counts.py -v`
Expected: FAIL — `AttributeError: 'Storage' object has no attribute 'save_keyword_count'`
(or a `no such table: keyword_counts` error).

- [ ] **Step 3: Add the table to the schema**

In `src/storage/schema.py`, insert this block immediately after the `judge_verdicts`
table definition (the `CREATE TABLE ... judge_verdicts (...)` ending at its `);`) and
before the `CREATE INDEX` lines:

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

- [ ] **Step 4: Add the import and method to Storage**

In `src/storage/store.py`, change the judge import line:

```python
from src.judge import JudgeVerdict
```

to:

```python
from src.judge import JudgeVerdict, KeywordCount
```

Then add this method right after `save_verdict` (immediately before `def close`):

```python
    def save_keyword_count(self, count: KeywordCount, *, run_id: int) -> None:
        """Сохранить счётчик упоминаний термина для прогона (upsert по (run_id, term)).

        Повторный запуск того же термина для прогона заменяет прежнюю строку."""
        with self._conn:
            self._conn.execute(
                """INSERT OR REPLACE INTO
                       keyword_counts(run_id, term, count, speakers, created_at)
                   VALUES (?,?,?,?,?)""",
                (run_id, count.term, count.count,
                 json.dumps(list(count.speakers)), _now()),
            )
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/storage/test_keyword_counts.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Run the full storage package to check no regression**

Run: `uv run pytest tests/storage/ -v`
Expected: PASS (existing storage tests + new ones).

- [ ] **Step 7: Commit**

```bash
git add src/storage/schema.py src/storage/store.py tests/storage/test_keyword_counts.py
git commit -m "feat: keyword_counts table + Storage.save_keyword_count upsert"
```

---

## Task 3: CLI runner (`keyword_judge.py`)

**Files:**
- Create: `keyword_judge.py` (repo root)

This task is glue over already-tested pieces (`count_mentions`, `save_keyword_count`,
`selected_run_ids`, `reconstruct_records`). It is verified manually against the existing
`experiment.db` rather than with a unit test, mirroring how `judge_runs.py` is a thin,
untested runner over tested functions.

- [ ] **Step 1: Write the runner**

Create `keyword_judge.py`:

```python
"""Детерминированный судья: подсчёт упоминаний термина в сохранённых прогонах.

Альтернатива LLM-судье (judge_runs.py): без LLM. Для каждого выбранного прогона ищем
ТЕРМИН (число или слово) в тексте публичных реплик и считаем число РАЗНЫХ говорящих,
упомянувших его. Имена говорящих не учитываются — сопоставляется только текст реплики.
Результат пишется в БД (таблица keyword_counts), в CSV и на экран.

    uv run python keyword_judge.py TERM [--db experiment.db] [--csv keyword_counts.csv] \\
                                   [--design HASH ...] [--exclude-design HASH ...] \\
                                   [--name LABEL ...] [--exclude-name LABEL ...]
"""

from __future__ import annotations

import csv
import sys

from src.judge import count_mentions
from src.stats.selection import filter_from_argv, selected_run_ids
from src.storage import Storage
from src.storage.records import reconstruct_records

DB = "experiment.db"
CSV_OUT = "keyword_counts.csv"

_FLAGS_WITH_VALUE = {"--db", "--csv", "--design", "--exclude-design",
                     "--name", "--exclude-name"}


def _positional_term(argv: list[str]) -> str | None:
    """Первый аргумент, не являющийся флагом и не значением флага, — это ТЕРМИН."""
    skip = False
    for a in argv:
        if skip:
            skip = False
            continue
        if a in _FLAGS_WITH_VALUE:
            skip = True
            continue
        if a.startswith("--"):       # неизвестный флаг без значения
            continue
        return a
    return None


def _opt(argv: list[str], name: str, default: str) -> str:
    """Значение одиночного `--name V` или default."""
    return argv[argv.index(name) + 1] if name in argv else default


def run(argv: list[str]) -> int:
    """Посчитать упоминания термина по выбранным прогонам; вернуть код возврата."""
    term = _positional_term(argv)
    if term is None:
        print("Ошибка: не задан ТЕРМИН для поиска.")
        print("Использование: uv run python keyword_judge.py TERM [--db ...] [--csv ...] "
              "[--design H] [--name L] ...")
        return 2

    db_path = _opt(argv, "--db", DB)
    csv_path = _opt(argv, "--csv", CSV_OUT)
    flt = filter_from_argv(argv)

    st = Storage(db_path)
    rows: list[tuple[int, str, int]] = []   # (run_id, name, count)
    try:
        run_ids = selected_run_ids(st.conn, flt)
        print(f"Термин: {term!r}; под фильтр попало прогонов: {len(run_ids)}")
        for rid in run_ids:
            name_row = st.conn.execute(
                "SELECT name FROM runs WHERE run_id=?", (rid,)
            ).fetchone()
            name = name_row[0] if name_row and name_row[0] is not None else ""
            records = reconstruct_records(st.conn, rid)
            kc = count_mentions(records, term)
            st.save_keyword_count(kc, run_id=rid)
            rows.append((rid, name, kc.count))
            print(f"  прогон {rid} ({name}): {kc.count}")
    finally:
        st.close()

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "name", "term", "count"])
        for rid, name, count in rows:
            w.writerow([rid, name, term, count])

    total = sum(c for _, _, c in rows)
    print(f"\nИтог: суммарно говорящих с упоминанием — {total}; CSV: {csv_path}")
    return 0


def main() -> None:
    sys.exit(run(sys.argv[1:]))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the missing-term guard**

Run: `uv run python keyword_judge.py`
Expected: prints `Ошибка: не задан ТЕРМИН ...` and the usage line; process exits non-zero
(`echo $?` → `2`).

- [ ] **Step 3: Verify a real run against `experiment.db`**

Run: `uv run python keyword_judge.py 7`
Expected: prints `Термин: '7'; под фильтр попало прогонов: N`, one `прогон <id> (...): <count>`
line per finished run, then the `Итог:` line. Creates/overwrites `keyword_counts.csv`.

- [ ] **Step 4: Verify the DB was written and upsert holds on re-run**

Run: `uv run python keyword_judge.py 7` again, then:
`uv run python -c "import sqlite3; c=sqlite3.connect('experiment.db'); print(c.execute('SELECT run_id, term, count FROM keyword_counts ORDER BY run_id').fetchall())"`
Expected: one row per run for `term='7'` (re-running did not duplicate rows — upsert works).

- [ ] **Step 5: Inspect the CSV**

Run: `uv run python -c "print(open('keyword_counts.csv').read())"`
Expected: header `run_id,name,term,count` followed by one row per selected run.

- [ ] **Step 6: Commit**

```bash
git add keyword_judge.py
git commit -m "feat: keyword_judge.py runner — count term mentions across stored runs"
```

Note: `keyword_counts.csv` is a generated artifact. If the repo gitignores `stats.csv`
or similar, add `keyword_counts.csv` to `.gitignore` in this commit; otherwise leave it
untracked (do not commit the generated CSV).

---

## Self-Review notes

- **Spec coverage:** no-LLM substring search → Task 1; case-sensitive → `test_should_match_case_sensitively`; reply-text-only / names-ignored → `test_should_ignore_speaker_name_and_match_only_reply_text`; distinct-speaker count → Task 1 tests; print+CSV+DB → Tasks 2 & 3; LLM judge untouched → no edits to `judge.py`/`JudgeCfg`/`judge_verdicts`.
- **Type consistency:** `KeywordCount(term, count, speakers)` defined in Task 1 and used identically in Tasks 2 & 3; `count_mentions(records, term)` and `save_keyword_count(count, *, run_id)` signatures match across tasks.
- **No placeholders:** every code step is complete and runnable.
