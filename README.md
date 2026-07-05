# LLM_reputation

A study of whether an institution of **reputation** emerges on its own in groups of AI agents.

A population of LLM agents plays a coordination game round after round, with pre-play
negotiation (cheap-talk). The goal is to check whether reputation forms between agents
without any external rules — purely from repeated interactions.

## How it works

One **episode** = one YAML configuration file. The orchestrator builds a population of
agents, a matchmaker splits them into pairs each round, and within each pair the agents
exchange short messages and then secretly pick a number 0–9. The payoff depends on
whether the numbers matched, whether one was exactly one above the other (mod 10), or
neither.

The play strategy is chosen in the configuration:
- `direct` — the agent picks a number directly;
- `prediction` — the agent predicts its partner's number, and a mapping
  (`match` / `one_above`) turns the prediction into the agent's own choice.

## Installation and usage

```bash
uv sync --extra dev          # install dependencies (including pytest)
```

Create a `.env` with your provider key (Together.ai, an OpenAI-compatible endpoint, is
used by default):

```
TOGETHER_API_KEY=your_key
```

Run an episode:

```bash
uv run python examples/orchestrator_demo.py                          # config/example.yaml
uv run python examples/orchestrator_demo.py config/example_prediction.yaml
```

Experiments with SQLite persistence: the configuration lives in YAML (default
`config/experiment.yaml`); `experiment.py` only loads and runs it. Each run is appended
to the shared `experiment.db` database (re-running the same configuration is skipped).
`replay.py` replays a stored run round by round without any LLM calls:

```bash
uv run python experiment.py                       # config/experiment.yaml
uv run python experiment.py config/example.yaml   # a different episode
uv run python experiment.py config/experiment.yaml "run name"
uv run python replay.py                # list runs in the database
uv run python replay.py <run_id>       # replay a run
uv run python replay.py <run_id> -c    # + prompts, roster and config parameters
```

`coplayer.py` looks at a run from a different angle — through the eyes of a single
agent: it collects all of the agent's encounters and groups them by **co-player**,
highlighting repeated encounters, where memory and reputation come into play. By
default it reads `research.db`:

```bash
uv run python coplayer.py                       # finished gpt-oss runs (--all — all runs)
uv run python coplayer.py <run_id>              # overview: each agent's co-players and score
uv run python coplayer.py <run_id> "<agent>"    # focus: one agent's encounters by co-player
uv run python coplayer.py <run_id> "<agent>" --chat   # + the cheap-talk of every encounter
```

To enable the **LLM judge** — a separate model that decides after the episode whether
an institution of reputation emerged — add a `judge:` block to the YAML configuration
(see `config/example.yaml`, commented-out example) or uncomment `JUDGE` in
`experiment.py` (using `JudgeCfg`). The verdict is printed at the end of the episode
and saved to the database, and `replay.py` highlights the messages the judge cited in
yellow and adds a JUDGE VERDICT section.

**The judge is configured fully independently of the agents.** The `judge` block has
its own `provider` (separate `base_url`, `model`, `api_key_env`), so the agents and the
judge can live on different endpoints: for example, agents on an external API
(Together.ai) and the judge on a local model via Ollama (or the other way around). Any
OpenAI-compatible combination works.

```yaml
# Agents — external API (Together.ai), judge — a local model via Ollama.
provider_default: &default
  base_url: https://api.together.xyz/v1   # external API
  api_key_env: TOGETHER_API_KEY           # key from .env
  model: Qwen/Qwen2.5-7B-Instruct-Turbo

judge:
  provider:
    base_url: http://localhost:11434/v1   # local Ollama, OpenAI-compatible
    model: qwen2.5:72b                     # no api_key_env needed (defaults to sk-noauth)
  # prompt: optional replacement for the default English judge prompt ({transcript})

population:
  provider: *default                       # agents use the external provider
  # ... the rest of the population
```

To swap them (agents local, judge on an external API), set
`base_url: http://localhost:11434/v1` in `provider_default`, and put the external
endpoint with `api_key_env` in `judge.provider`.

### Judging stored runs after the fact (backfill)

`judge_runs.py` runs the LLM judge over runs already stored in `experiment.db`: it
reconstructs the public cheap-talk, calls the judge and writes the verdict to the
database (which `replay.py` then highlights). The judge model is taken from the
`judge:` block of the config (default `config/experiment.yaml`) — set it there (via
`judge.provider`, or the `*provider` anchor to make the judge use the same model as the
agents). Already-judged runs are skipped unless `--force` is passed.

```bash
uv run python judge_runs.py                            # judge all finished runs
uv run python judge_runs.py --force                    # re-judge, including already judged
uv run python judge_runs.py --config config/example.yaml   # take the judge from another config
uv run python judge_runs.py --design <HASH>            # only one design (flag is repeatable)
uv run python judge_runs.py --exclude-name <LABEL>     # exclude runs by name
```

Runs need a reachable judge provider; the key is read from `.env`. The filters
(`--design` / `--exclude-design` / `--name` / `--exclude-name`) select which runs to
judge.

### Deterministic judge: term mentions

`keyword_judge.py` is an LLM-free alternative to the LLM judge. For each selected run
it searches for a TERM (a number or a word) in the text of public messages and counts
the number of DISTINCT speakers that mentioned it (speaker names are ignored — only the
message text is matched). The search is a case-sensitive substring match. The result is
written to the database (the `keyword_counts` table, upsert by `(run_id, term)`), to a
CSV and to the screen.

```bash
uv run python keyword_judge.py 123                     # mentions of "123" across all finished runs
uv run python keyword_judge.py 123 --db research.db    # a different database (default experiment.db)
uv run python keyword_judge.py trust --csv out.csv     # a word term + a different CSV path
uv run python keyword_judge.py 7 --design <HASH>       # only one design (flag is repeatable)
uv run python keyword_judge.py 7 --exclude-name <LABEL>    # exclude runs by name
```

`keyword_judge.py` accepts the same run-selection filters as `judge_runs.py`
(`--design` / `--exclude-design` / `--name` / `--exclude-name`); only finished runs are
considered. No LLM is needed — no provider or key required.

### Collecting statistics

`collect_stats.py` aggregates judge verdicts by design (`config_hash`): the share of
runs in which an institution of reputation emerged, with a 95% Wilson confidence
interval. It prints a table to the console and writes `stats.json` + `stats.csv`.

```bash
uv run python collect_stats.py                         # all judged runs -> stats.json + stats.csv
uv run python collect_stats.py --design <HASH>         # only selected designs (flag is repeatable)
uv run python collect_stats.py --out s.json --csv s.csv    # different artifact paths
```

`collect_stats.py` accepts the same run-selection filters as `judge_runs.py`. Only runs
that already have a judge verdict are counted — run `judge_runs.py` first.

For debugging, you can enable tracing of the exact LLM input before a number is chosen
(the flag can also be set in `.env`):

```bash
LLM_TRACE=1 uv run python examples/orchestrator_demo.py
```

Tests:

```bash
uv run pytest
```

Unit tests never hit the network (the LLM is replaced with a stub). Smoke tests talk to
a local Ollama and are skipped automatically if it is unavailable.

## Documentation

Architecture and layer design live in `docs/`: English overviews (`architecture.md`,
`configuration.md`, `testing.md`, `conventions.md`) and detailed per-layer design
documents (`agent-games-*-plan.md`, in Russian).

## Team

[Andrey Seryakov](https://github.com/AndreySeryakov)
[Ekaterina Krupkina](https://github.com/ktchka)
[Andrey Bystrov](https://github.com/Shougakusei)
[Dmitrii Salnikov](https://github.com/DmitrySalnikov)

## TODO

- describe how it works in more detail
- think about manually substituting specific steps and design the experiment
- rationale for reasoning models — write thinking
- show agents their reasoning from past rounds
