# LLM Decide-Trace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Log the exact LLM input (system prompt + all messages) for the DECIDE/PREDICT phases at DEBUG level, switchable in the demo via `LLM_TRACE=1`.

**Architecture:** A module-level logger in `src/core/agent.py` emits one DEBUG record per `provider.complete` attempt when the phase is DECIDE or PREDICT — the only place where the final payload (persona+rules system, memory diary, phase context, retry correction) exists. With no logging config nothing is emitted, so `src/` stays output-silent. `examples/orchestrator_demo.py` opts in by attaching a `StreamHandler` to that logger when env var `LLM_TRACE` is set.

**Tech Stack:** Python 3.12, stdlib `logging`, pytest + pytest-asyncio (auto mode), `caplog` fixture, `ScriptedProvider` test stub.

**Spec:** `docs/superpowers/specs/2026-06-10-llm-decide-trace-design.md`

**Conventions that apply (from CLAUDE.md):**
- Docstrings / log text in Russian; keep English terms (LLM, system) untranslated; prompt content stays English.
- TDD: write the failing test first.
- Use `uv run pytest` to run tests.

---

### Task 1: Trace logging in `Agent.act`

**Files:**
- Modify: `src/core/agent.py`
- Test: `tests/core/test_agent.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/core/test_agent.py` (the file already imports `Agent, AgentSetup, Phase, PhaseKind` and has helpers `_agent`, `_decide`, `_talk` — see its top). Add `import logging` next to the existing imports at the top of the file:

```python
import logging
```

Append at the end of the file:

```python
def _trace_records(caplog):
    return [r for r in caplog.records if r.name == "src.core.agent"]


async def test_decide_logs_full_llm_input_at_debug(caplog):
    caplog.set_level(logging.DEBUG, logger="src.core.agent")
    p = ScriptedProvider(['{"number": 4, "rationale": "ok"}'])
    await _agent(p, persona="PERSONA").act(
        Phase(PhaseKind.DECIDE, "SITUATION", rules="GAME RULES")
    )
    records = _trace_records(caplog)
    assert len(records) == 1
    msg = records[0].getMessage()
    assert "PERSONA" in msg and "GAME RULES" in msg  # system prompt
    assert "SITUATION" in msg                        # phase context message


async def test_predict_logs_llm_input_at_debug(caplog):
    caplog.set_level(logging.DEBUG, logger="src.core.agent")
    p = ScriptedProvider(['{"number": 7, "rationale": "guess"}'])
    await _agent(p).act(Phase(PhaseKind.PREDICT, "PREDICT-CTX", rules="R"))
    records = _trace_records(caplog)
    assert len(records) == 1
    assert "PREDICT-CTX" in records[0].getMessage()


async def test_talk_logs_no_llm_input(caplog):
    caplog.set_level(logging.DEBUG, logger="src.core.agent")
    p = ScriptedProvider(['{"message": "hi", "ready": true}'])
    await _agent(p).act(_talk())
    assert _trace_records(caplog) == []


async def test_decide_retry_logs_attempt_with_correction(caplog):
    caplog.set_level(logging.DEBUG, logger="src.core.agent")
    p = ScriptedProvider(["bad", '{"number": 1, "rationale": "r"}'])
    await _agent(p).act(_decide())
    records = _trace_records(caplog)
    assert len(records) == 2  # one record per provider attempt
    second = records[1].getMessage()
    assert "ONLY valid JSON" in second  # the correction message is part of the input
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
uv run pytest tests/core/test_agent.py -k "logs" -v
```

Expected: 4 FAILED — `assert len(records) == 1` fails with `len == 0` (no logging exists yet); the TALK test may PASS vacuously — that's fine, the other three must fail.

- [ ] **Step 3: Implement trace logging in `src/core/agent.py`**

Add `logging` to the imports at the top of `src/core/agent.py`:

```python
import json
import logging
import random
import re
```

After the `_MAX_PARSE_RETRIES = 2` line, add the module logger:

```python
_log = logging.getLogger(__name__)
```

In `Agent.act`, replace the loop header and add the trace call. Current code (src/core/agent.py:89-90):

```python
        for _ in range(_MAX_PARSE_RETRIES + 1):
            messages = base if correction is None else base + [Message("user", correction)]
```

becomes:

```python
        for attempt in range(1, _MAX_PARSE_RETRIES + 2):
            messages = base if correction is None else base + [Message("user", correction)]
            if phase.kind in (PhaseKind.DECIDE, PhaseKind.PREDICT) and _log.isEnabledFor(logging.DEBUG):
                _log.debug(_render_trace(self.id, phase.kind, attempt, system, messages))
```

(`attempt` runs 1..3 — same number of iterations as before, just 1-based for the log.)

At module level, next to the other `_`-helpers (e.g. after `_fallback`), add:

```python
def _render_trace(agent_id: str, kind: PhaseKind, attempt: int,
                  system: str, messages: list[Message]) -> str:
    """Отрендерить точный вход LLM (system + все сообщения) для записи в лог.

    Args:
        agent_id: Идентификатор агента, делающего запрос.
        kind: Фаза запроса (DECIDE или PREDICT).
        attempt: Номер попытки запроса (1..3, повторы из-за ошибок парсинга).
        system: Полный системный промпт (персона + правила).
        messages: Сообщения запроса (дневник памяти, контекст фазы, поправка).

    Returns:
        Многострочный текст записи лога.
    """
    parts = [
        f"LLM-вход: агент {agent_id}, фаза {kind.value}, попытка {attempt}",
        f"--- system ---\n{system}",
    ]
    parts += [f"--- {m.role} ---\n{m.content}" for m in messages]
    return "\n".join(parts)
```

- [ ] **Step 4: Run the new tests to verify they pass**

Run:

```bash
uv run pytest tests/core/test_agent.py -k "logs" -v
```

Expected: 4 PASSED.

- [ ] **Step 5: Run the full test suite (no regressions)**

Run:

```bash
uv run pytest
```

Expected: all tests pass (Ollama smoke tests may auto-skip).

- [ ] **Step 6: Commit**

```bash
git add src/core/agent.py tests/core/test_agent.py
git commit -m "feat: log full LLM input for DECIDE/PREDICT at DEBUG level"
```

---

### Task 2: `LLM_TRACE` switch in the demo

**Files:**
- Modify: `examples/orchestrator_demo.py`

No unit test: this is glue in `examples/` (not `src/`), existing demos have no tests, and exercising it end-to-end needs a live provider. Verified by running the episode manually (Step 3, optional if no API key).

- [ ] **Step 1: Wire the env-var switch**

In `examples/orchestrator_demo.py`, extend the stdlib imports (currently `asyncio`, `random`, `sys` at examples/orchestrator_demo.py:15-17):

```python
import asyncio
import logging
import os
import random
import sys
```

After the `load_dotenv()` line (examples/orchestrator_demo.py:25), add:

```python
if os.environ.get("LLM_TRACE", "0") not in ("", "0"):
    # Включить трассировку LLM-входа фаз DECIDE/PREDICT (см. src/core/agent.py)
    _trace_handler = logging.StreamHandler()
    _trace_handler.setFormatter(logging.Formatter("\n%(message)s"))
    _trace_logger = logging.getLogger("src.core.agent")
    _trace_logger.setLevel(logging.DEBUG)
    _trace_logger.addHandler(_trace_handler)
```

Also extend the module docstring's run instructions (the docstring ends at examples/orchestrator_demo.py:13 with the `PYTHONPATH=...` example) — add one line right after that example, inside the docstring:

```
    Трассировка LLM-входа перед выбором числа: LLM_TRACE=1 (флаг можно задать в .env).
```

- [ ] **Step 2: Sanity-check the wiring without network**

Run:

```bash
LLM_TRACE=1 uv run python -c "
import examples.orchestrator_demo, logging
lg = logging.getLogger('src.core.agent')
assert lg.level == logging.DEBUG and lg.handlers, 'трассировка не включилась'
print('OK: trace logger configured')
"
```

Expected output: `OK: trace logger configured`

- [ ] **Step 3 (optional, needs `TOGETHER_API_KEY` in `.env`): see it live**

Run:

```bash
LLM_TRACE=1 uv run python examples/orchestrator_demo.py
```

Expected: before each round's results, multi-line `LLM-вход: агент …, фаза decide, попытка 1` blocks showing system prompt, memory diary, and the decide context.

- [ ] **Step 4: Commit**

```bash
git add examples/orchestrator_demo.py
git commit -m "feat: enable LLM input trace in demo via LLM_TRACE env var"
```
