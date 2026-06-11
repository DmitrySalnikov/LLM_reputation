"""Run one reputation experiment from a YAML config.

The experiment definition lives in a YAML file — `config/experiment.yaml` by default.
This script only loads that file and hands it to the runner; all the run logic
(build population → drive episode → persist + narrate → score) lives in src/runner.py.
Point it at another config to run a different episode (e.g. config/example.yaml).

Run from the repo root. Optional positional args: a config path, then a human label for
the run (stored in runs.name):

    uv run python experiment.py [config.yaml] ["run name"]

Every run is appended to one DB ("one DB, many runs"), keyed by a hash of its config
(run_id): re-running an identical config is skipped, so the DB de-dups. To force a fresh
run, change the seed (or anything else in the config).

Read a stored run back, round by round, with:

    uv run python replay.py <run_id>
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from dataclasses import replace

from dotenv import load_dotenv

from src.core.config import JudgeCfg, ProviderCfg, load_episode  # noqa: F401 (JudgeCfg/ProviderCfg used in commented example below)
from src.runner import run

load_dotenv()                       # подхватить ключи API из .env (например TOGETHER_API_KEY)

DB = "experiment.db"                 # one DB, many runs; appended to on every run
DEFAULT_CONFIG = "config/experiment.yaml"

# --- LLM judge (optional) ------------------------------------------------------
# Отдельная модель, которая в конце эпизода решает, возник ли институт репутации.
# Судья видит только публичный cheap-talk; вердикт печатается, сохраняется в БД и
# подсвечивается в replay.py. None = судья выключен.
# Чтобы включить: раскомментируй JUDGE ниже — он подменит блок judge из YAML
# (или задай блок `judge:` прямо в YAML-конфигурации — это эквивалентно).
JUDGE = None
# JUDGE = JudgeCfg(provider=ProviderCfg(
#     base_url="https://api.together.xyz/v1",
#     api_key_env="TOGETHER_API_KEY",
#     model="Qwen/Qwen2.5-72B-Instruct-Turbo",
# ))


def configure_llm_trace() -> None:
    """LLM_TRACE=1 (env или .env) -> DEBUG-вывод входа LLM фаз DECIDE/PREDICT/REFLECT.

    Хендлер настраивает этот caller-скрипт, а не движок: src/core/agent.py лишь пишет в
    логгер `src.core.agent` (CLAUDE.md: handlers configured by the caller, never in src/).
    """
    if os.environ.get("LLM_TRACE", "0") in ("", "0"):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("\n%(message)s"))
    logger = logging.getLogger("src.core.agent")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)


if __name__ == "__main__":
    configure_llm_trace()           # opt-in трассировка LLM-входа (env/.env), до запуска
    args = sys.argv[1:]
    config_path = args[0] if args else DEFAULT_CONFIG   # which episode YAML to run
    name = args[1] if len(args) > 1 else None           # optional human label for the run
    cfg = load_episode(config_path)
    if JUDGE is not None:           # Python-переопределение судьи поверх YAML (см. JUDGE выше)
        cfg = replace(cfg, judge=JUDGE)
    asyncio.run(run(cfg, DB, name))
