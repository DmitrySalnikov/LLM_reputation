# Conventions

## Language (from the user's global instructions)

- **Docstrings** (modules, classes, functions) are written in **Russian**,
  Google-style. See `src/strategy/base.py` and `src/games/prompts.py` for the
  house style (Args / Returns / Raises).
- **All `print`, logging, and error messages are in Russian** — e.g. the
  `ValueError`s in `src/core/config.py:88` and `src/strategy/mappings.py:31`.
- **Do not translate established English terms** (cheap-talk, payoff, matchmaker,
  prediction, …). Inline code comments are English; many predate the docstring rule.
- **LLM-facing prompt text is English** (`src/games/prompts.py`) — that is the
  model's working language, separate from the codebase's documentation language.
- Planning/chat with Claude is in English.

## Code patterns

- **Frozen dataclasses for config** (`src/core/config.py`). Build them through the
  `_*_cfg` helpers and `load_episode`, not by hand, so validation runs.
- **Protocols for seams** (`Protocol` + a `make_*` factory): `LLMProvider`,
  `PlayStrategy`, `Matchmaker`, `Game`, `PopulationGenerator`. New implementations
  register in the matching `make_*` factory.
- **Lazy imports to break cycles** between `games` and `strategy` — keep new
  cross-layer imports inside functions if they would otherwise cycle (see
  `src/games/reputation_pd.py:20`).
- **Fail fast at load time**: validate config once in `_validate`
  (`src/core/config.py:88`) rather than deep in the run.
- **`from __future__ import annotations`** at the top of every module.
- **rng is passed in, never created ad hoc** inside library code, so seeds control
  everything (`build(rng)`, `Matchmaker.setup(..., rng, ...)`).

## Output & ownership

- The orchestrator's **only output channel is the `observer` callback**. Don't add
  printing or persistence inside `src/` library code — emit a `PairingRecord` and
  let the caller (demo / future Logger) handle it.
- **The caller owns the `Population`**: it builds it, reads final scores from it,
  and must `await pop.aclose()` (close it in a `finally`).

## Dependency direction

`src → tests` only; never import from `tests/` into `src/`. Within `src/`, respect
the layering in `docs/architecture.md` — lower layers never import higher ones.
