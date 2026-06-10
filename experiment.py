"""One reputation experiment — the config you edit, then run.

This file is ONLY configuration plus a one-line call into the runner. All the run logic
(build population → drive episode → persist + narrate → score) lives in src/runner.py.

Run from the repo root (needs a local Ollama). An optional first argument names the run
(stored in runs.name as a human label):

    PYTHONPATH=. .venv/bin/python experiment.py ["run name"]

Every run is appended to one DB ("one DB, many runs"). A run is keyed by a hash of its
config (run_id): re-running an identical config is skipped, so the DB de-dups. To force a
fresh run, change the seed (or anything else in the config).

Read a stored run back, round by round, with:

    PYTHONPATH=. .venv/bin/python replay.py <run_id>
"""

import asyncio
import sys

from src.core.config import (
    AgentSpec,
    EpisodeCfg,
    GameCfg,
    Payoffs,
    PopulationCfg,
    ProviderCfg,
)
from src.runner import run

import os
from dotenv import load_dotenv

load_dotenv()

# ============================== CONFIG — EDIT HERE ==============================
DB = "experiment.db"            # one DB, many runs; appended to on every run

# --- LLM provider (shared by every agent below) -------------------------------
PROVIDER = ProviderCfg(
    # base_url="http://localhost:11434/v1",
    # model="llama3.1:8b",
    base_url="https://api.together.xyz/v1",   # Together.ai (OpenAI-совместимый)
    api_key_env=os.getenv("TOGETHER_API_KEY"),           # ключ берётся из .env (см. .gitignore)
    model="Qwen/Qwen2.5-7B-Instruct-Turbo",
    temperature=0.7,
    max_tokens=256,
    timeout_s=120.0,
)

# --- Personas (one string per agent type) -------------------------------------
PRAGMATIC = "You are a pragmatic, self-interested player who tries to win."
CAUTIOUS = "You are a cautious player who values trust."

# --- Payoffs (R both-cooperate, T betrayal, P miscoordinate, S betrayed) -------
# invariants: T > R > P > S and 2R > T + S
PAYOFFS = Payoffs(R=3.0, T=5.0, P=1.0, S=0.0)

# --- Game prompts — what every agent is told. Stored verbatim in the run's config.
# RULES is concrete: the payoff numbers come straight from PAYOFFS (no placeholders).
# talk/decide keep the runtime placeholders {partner}/{round}/{feed} — those values
# aren't known until a turn is played, so the runner fills them in then.
RULES = (
    "You are playing a repeated game. Each round you face one partner.\n"
    "Both of you secretly pick an integer from 0 to 9 at the same time.\n"
    f"- Equal numbers: you each score {PAYOFFS.R:g}.\n"
    f"- Your number exactly one higher than your partner's (0 counts as following 9): "
    f"you score {PAYOFFS.T:g}, your partner {PAYOFFS.S:g}.\n"
    f"- Your partner's number exactly one higher than yours: they score {PAYOFFS.T:g}, you {PAYOFFS.S:g}.\n"
    f"- Any other case: you each score {PAYOFFS.P:g}.\n"
    "Before choosing you may exchange short messages. Messages are not binding; "
    "the final choice is secret and simultaneous. Maximize your own total score."
)

TALK_PROMPT = (
    "Your partner this round is {partner}. Round {round}.\n"
    "Negotiation so far:\n{feed}\n\n"
    'Send a short message to your partner. Set "ready": true when you have nothing more to say.\n'
    'Respond ONLY as JSON: {"message": "<your message>", "ready": <true|false>}'
)

DECIDE_PROMPT = (
    "Your partner this round is {partner}. Round {round}.\n"
    "Negotiation:\n{feed}\n\n"
    "Now secretly choose your number from 0 to 9.\n"
    'Respond ONLY as JSON: {"number": <0-9>, "rationale": "<short reason>"}'
)

# --- The single experiment to run ---------------------------------------------
# Each AgentSpec builds `count` agents of that type; population size = sum(counts).
CONFIG = EpisodeCfg(
    seed=2,
    rounds=2,
    matchmaker="random",
    population=PopulationCfg(
        kind="roster",
        agents=[
            AgentSpec(persona=PRAGMATIC, provider=PROVIDER, count=1),
            AgentSpec(persona=CAUTIOUS, provider=PROVIDER, count=1),
        ],
    ),
    game=GameCfg(
        payoffs=PAYOFFS,
        max_talk_turns=1,
        rules=RULES,
        talk_prompt=TALK_PROMPT,
        decide_prompt=DECIDE_PROMPT,
    ),
    context_window=None,        # None = each agent sees its whole memory; int = last N entries
    idle_payoff=1.0,            # what an agent scores in a round it sits out
    max_concurrency=4,          # pairings played in parallel per round
)
# ===============================================================================


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else None    # optional human label for the run
    asyncio.run(run(CONFIG, DB, name))
