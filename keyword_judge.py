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
