# uv run python research.py

from __future__ import annotations

import asyncio
import sqlite3
import time
from dataclasses import replace

from dotenv import load_dotenv

from export_runs import _out_dir_for, export_run
from src.core.config import EpisodeCfg, load_episode
from src.runner import resume_run, run
from src.storage import Storage

load_dotenv()                       # ключи API из .env (TOGETHER_API_KEY)

CONFIG = "config/research.yaml"
DB = "qwen3.db"
SPLIT_DIR = _out_dir_for(DB)        # папка с по-прогонными файлами = имя БД без расширения (qwen3.db -> qwen3/)
GAMES_PER_MODEL = 100


def _split_off(run_id: int) -> None:
    """Выгрузить прогон в отдельный файл SPLIT_DIR/<номер>.db, перезаписывая существующий."""
    conn = sqlite3.connect(DB)
    try:
        export_run(conn, run_id, SPLIT_DIR, overwrite=True)
    finally:
        conn.close()
MODELS = [
    # ("llama-3-8b",      "meta-llama/Meta-Llama-3-8B-Instruct-Lite"),
    # ("qwen2.5-7b",      "Qwen/Qwen2.5-7B-Instruct-Turbo"),
    # ("deepseek-v4-pro", "deepseek-ai/DeepSeek-V4-Pro"),
    # ("gpt-oss-20b",     "openai/gpt-oss-20b"),
    ("qwen3-FP8",       "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"),
]


def _cfg_for_model(model_id: str) -> EpisodeCfg:
    """Свежий конфиг исследования с подменённой моделью.

    load_episode перечитывает research.yaml целиком, поэтому `seed: random` даёт НОВЫЙ сид
    на каждый вызов (каждая игра — своя). Меняем только provider.model — остальной дизайн
    (популяция, payoff'ы, промпты) фиксирован конфигом."""
    cfg = load_episode(CONFIG)
    provider = replace(cfg.population.provider, model=model_id)
    return replace(cfg, population=replace(cfg.population, provider=provider))


async def _resume_unfinished() -> None:
    """Фаза 1: доиграть все незавершённые прогоны (finished_at IS NULL)."""
    st = Storage(DB)
    try:
        unfinished = st.unfinished_runs()
    finally:
        st.close()
    for run_id, name in unfinished:
        print(f"resume {name}")
        await resume_run(run_id, DB, quiet=True)
        _split_off(run_id)                        # доигран — выгрузить в файл


async def _fill_missing() -> None:
    """Фаза 2: добить недостающие игры по плану (модель × номер). Имя прогона = '<модель> <i>',
    номер итерации i — с 1 (по модели).

    Ищем по имени: если прогон с таким именем уже есть (доигран в фазе 1 либо ранее, или ещё
    открыт) — пропускаем; иначе считаем новый. Так продолжаем с первой непрогнанной записи.
    Печатаем `calculating <name>` до запуска и `done <wall-time>` после."""
    for label, model_id in MODELS:
        for i in range(1, GAMES_PER_MODEL + 1):      # номер итерации у модели — с 1
            name = f"{label} {i}"
            st = Storage(DB)
            try:
                exists = st.run_id_by_name(name) is not None
            finally:
                st.close()
            if exists:
                continue
            print(f"calculating {name}")
            t0 = time.monotonic()
            run_id = await run(_cfg_for_model(model_id), DB, name, quiet=True)
            _split_off(run_id)                       # посчитан — выгрузить в файл
            print(f"done {time.monotonic() - t0:.1f}s")


async def _main() -> None:
    await _resume_unfinished()
    await _fill_missing()


if __name__ == "__main__":
    asyncio.run(_main())
