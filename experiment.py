"""Run one reputation experiment from a YAML config.

The experiment definition lives in a YAML file — `config/experiment.yaml` by default.
This script only loads that file and hands it to the runner; all the run logic
(build population → drive episode → persist + narrate → score) lives in src/runner.py.
Point it at another config to run a different episode (e.g. config/example.yaml).

Run from the repo root. Optional positional args: a config path, then a human label for
the run (stored in runs.name):

    PYTHONPATH=. uv run python experiment.py [config.yaml] ["run name"]

Every run is appended to one DB ("one DB, many runs"), keyed by a hash of its config
(run_id): re-running an identical config is skipped, so the DB de-dups. To force a fresh
run, change the seed (or anything else in the config).

Read a stored run back, round by round, with:

    PYTHONPATH=. uv run python replay.py <run_id>
"""

from __future__ import annotations

import asyncio
import sys

from dotenv import load_dotenv

from src.core.config import load_episode
from src.runner import run

load_dotenv()                       # подхватить ключи API из .env (например TOGETHER_API_KEY)

DB = "experiment.db"                 # one DB, many runs; appended to on every run
DEFAULT_CONFIG = "config/experiment.yaml"


if __name__ == "__main__":
    args = sys.argv[1:]
    config_path = args[0] if args else DEFAULT_CONFIG   # which episode YAML to run
    name = args[1] if len(args) > 1 else None           # optional human label for the run
    asyncio.run(run(load_episode(config_path), DB, name))
