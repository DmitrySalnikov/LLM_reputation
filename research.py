"""Прогон исследования: сетка «модель × игра» в один research.db.

Дизайн исследования (config/research.yaml): 10 игроков (8 обычных + 2 «плохих»/дефектора),
temperature 0.7, 10 раундов, без memory notes. Здесь задаётся только то, что варьируется
между прогонами: список моделей (MODELS) и число игр на модель (GAMES_PER_MODEL).

Каждая игра — отдельный прогон (run) в общей БД; имя прогона = "<модель> <номер>"
(напр. "llama-3-8b 0"). Скрипт ИДЕМПОТЕНТЕН и возобновим:

  uv run python research.py

  1) сперва доигрывает ВСЕ незавершённые прогоны (resume);
  2) затем добивает недостающие игры по плану (модель × номер), пропуская уже посчитанные
     — поиск по имени прогона.

На каждый прогон в консоль печатается ТОЛЬКО его имя и статус — `resume` или `calculating`
(движок гонится в тихом режиме, quiet=True: пораундовая narration и scoreboard подавлены,
наружу пробиваются лишь сообщения об обрыве прогона).
Для каждой НОВОЙ игры конфиг грузится заново (seed: random -> свежий сид), а provider.model
переопределяется на текущую модель. Возобновление берёт сохранённый конфиг прогона как есть.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

from dotenv import load_dotenv

from src.core.config import EpisodeCfg, load_episode
from src.runner import resume_run, run
from src.storage import Storage

load_dotenv()                       # ключи API из .env (TOGETHER_API_KEY)

CONFIG = "config/research.yaml"
DB = "research.db"
GAMES_PER_MODEL = 100

# (метка прогона, Together model id). Метка идёт в имя прогона ("<метка> <номер>").
# NB: точные slug'и DeepSeek-V4-Pro и gpt-oss-20b на Together уточни по каталогу
# (https://api.together.xyz/models) — при несовпадении прогоны посыплются ProviderError.
MODELS = [
    ("llama-3-8b",      "meta-llama/Meta-Llama-3-8B-Instruct-Lite"),
    ("qwen2.5-7b",      "Qwen/Qwen2.5-7B-Instruct-Turbo"),
    ("deepseek-v4-pro", "deepseek-ai/DeepSeek-V4-Pro"),
    ("gpt-oss-20b",     "openai/gpt-oss-20b"),
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
        print(f"[{name}] resume")
        await resume_run(run_id, DB, quiet=True)


async def _fill_missing() -> None:
    """Фаза 2: добить недостающие игры по плану (модель × номер), пропуская уже начатые.

    Ищем по имени прогона: если прогон с таким именем уже есть (доигран в фазе 1 либо ранее,
    или ещё открыт) — пропускаем; иначе считаем новый. Так продолжаем с первой непрогнанной
    записи в фиксированном порядке MODELS × индекс."""
    for label, model_id in MODELS:
        for i in range(GAMES_PER_MODEL):
            name = f"{label} {i}"
            st = Storage(DB)
            try:
                exists = st.run_id_by_name(name) is not None
            finally:
                st.close()
            if exists:
                continue
            print(f"[{name}] calculating")
            await run(_cfg_for_model(model_id), DB, name, quiet=True)


async def _main() -> None:
    await _resume_unfinished()
    await _fill_missing()


if __name__ == "__main__":
    asyncio.run(_main())
