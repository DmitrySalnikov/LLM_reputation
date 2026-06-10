# LLM input trace for the number-choice call — design

**Date:** 2026-06-10
**Status:** approved

## Problem

When debugging an episode there is no way to see what the LLM actually receives
before an agent picks its number. The full payload (system prompt with persona and
rules, the rendered memory diary, the phase context with the negotiation feed, and
any parse-retry correction) exists only transiently inside `Agent.act`
(`src/core/agent.py`); `PairingRecord` stores outputs only.

## Goal

Live debug tracing: print the exact LLM input for the DECIDE and PREDICT phases
while an episode runs. Cheap-talk (TALK) calls are NOT traced. Nothing is persisted;
this is not research data collection.

## Design

### Mechanism: DEBUG logging inside `Agent.act`

- Add a module-level logger in `src/core/agent.py`:
  `_log = logging.getLogger("src.core.agent")` (module `__name__`).
- In `Agent.act`, immediately before each `provider.complete(...)` call, if
  `phase.kind in (PhaseKind.DECIDE, PhaseKind.PREDICT)` and the logger is enabled
  for DEBUG, emit ONE DEBUG record per attempt containing:
  - agent id, phase kind, attempt number (1..3, so retry corrections are visible);
  - the full `system` string (persona + rules);
  - every message in `messages` rendered as `role: content`
    (memory diary, phase context, correction message if present).
- One multi-line record with clear delimiters. Log scaffolding text in Russian
  (project convention); prompt content stays English as-is.
- No behavior change. With no logging configuration the logger has no handler and
  nothing is emitted — `src/` stays output-silent, consistent with the
  observer-only output rule.

### Enabling: `LLM_TRACE=1` in the demo

- `examples/orchestrator_demo.py` checks `os.environ.get("LLM_TRACE")` after
  `load_dotenv()` (so the flag may live in `.env`).
- If set to a non-empty value other than `"0"`, attach a `StreamHandler` at DEBUG
  level to the `src.core.agent` logger only (not the root logger).
- Usage:

  ```bash
  LLM_TRACE=1 PYTHONPATH=. .venv/bin/python examples/orchestrator_demo.py
  ```

### Testing (TDD, written first)

Unit tests in `tests/core/test_agent.py` using the existing `ScriptedProvider`
stub and pytest's `caplog`:

1. A DECIDE call logs one DEBUG record containing the system prompt and the phase
   context.
2. A TALK call logs nothing.
3. A parse retry on DECIDE logs a second record that contains the correction
   message.

### Error handling

None beyond the above: the `logging` module never propagates handler errors into
the game loop.

## Out of scope

- Tracing TALK calls.
- Persisting prompts (e.g. into `PairingRecord` / observer payloads).
- Any provider-level logging wrapper.
