# Prediction Strategy & Random Names Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional prediction-based play strategy and replace `A1/A2` agent ids with random `"First Last"` names sampled from configured pools.

**Architecture:** A new `src/strategy/` package (a `PlayStrategy` protocol + `DirectStrategy`/`PredictionStrategy` + a pluggable prediction-mapping registry) is injected into `ReputationPD`, which delegates its decide sub-step to the strategy. Strategy is selected globally per run via `play_strategy` config. Name pools are required in YAML; `RosterGenerator` samples them (seeded) and the sampled name becomes the agent id everywhere. A shared `src/games/prompts.py` holds the LLM context builders to keep `ReputationPD` and the strategies decoupled (no import cycle).

**Tech Stack:** Python 3.11+, frozen dataclasses, `asyncio`, `pytest` + `pytest-asyncio` (`asyncio_mode = auto`), `uv` for deps/running.

---

## Conventions for every task

- Run tests with: `uv run pytest <path> -v` (dev deps already synced via `uv sync --extra dev`).
- Run the full non-network suite with: `uv run pytest -q -k "not ollama and not live and not smoke"`.
- Docstrings on new modules/classes/functions: **Russian**, Google-style (per project CLAUDE.md).
- Validation / error messages: **Russian** (per project CLAUDE.md fail-fast convention).
- LLM-facing prompt text: **English** — it must match the existing prompts in `reputation_pd.py` that partners already see. Do not translate prompt strings.
- Commit after each task with the shown message.

## Baseline note (pre-existing failure)

Before any change, `uv run pytest -k "not ollama and not live and not smoke"` reports **1 failure**:
`tests/core/test_config_load.py::test_load_example` asserts `cfg.game.max_talk_turns == 3` but
`config/example.yaml` ships `max_talk_turns: 8`. This is unrelated to this feature. **Task 2 fixes it**
by aligning the test assertion to the committed file value (`8`). After Task 2 the whole non-network
suite must be green.

## File map

**Create:**
- `src/strategy/__init__.py` — package exports
- `src/strategy/mappings.py` — prediction-mapping registry (Task 1)
- `src/strategy/base.py` — `Decision`, `PlayStrategy` protocol, `make_strategy` factory (Task 5)
- `src/strategy/direct.py` — `DirectStrategy` (Task 5)
- `src/strategy/prediction.py` — `PredictionStrategy` (Task 5)
- `src/games/prompts.py` — `rules_text`/`talk_context`/`decide_context`/`predict_context` (Task 4)
- `tests/strategy/conftest.py` — `ScriptedProvider` double (Task 5)
- `tests/strategy/test_mappings.py` (Task 1)
- `tests/strategy/test_direct.py`, `tests/strategy/test_prediction.py` (Task 5)

**Modify:**
- `src/core/config.py` — `EpisodeCfg` + `PopulationCfg` fields, `load_episode` validation (Task 2)
- `config/example.yaml` — name pools + max_talk_turns fix (Task 2)
- `src/core/agent.py` — `PhaseKind.PREDICT` (Task 3)
- `src/games/reputation_pd.py` — import prompts (Task 4), delegate decide to strategy (Task 6)
- `src/core/memory.py` — `MemoryEntry.my_predicted` + render (Task 6)
- `src/games/base.py` — `PairingRecord.a_predicted/b_predicted` (Task 6)
- `src/population/base.py` — `Population.add(agent_id=...)` (Task 7)
- `src/population/roster.py` — name sampling (Task 7)
- `src/core/orchestrator.py` — build + inject strategy (Task 8)
- Tests: `test_config_load.py`, `test_agent.py`, `test_play_pairing.py`, `test_memory.py`, `test_roster.py`, `test_orchestrator.py`

---

## Task 1: Prediction-mapping registry

**Files:**
- Create: `src/strategy/__init__.py`
- Create: `src/strategy/mappings.py`
- Test: `tests/strategy/test_mappings.py`

- [ ] **Step 1: Create the test directory marker and write the failing test**

Create `tests/strategy/test_mappings.py`:

```python
from __future__ import annotations

import pytest

from src.strategy.mappings import get_mapping


def test_match_returns_predicted_unchanged():
    f = get_mapping("match")
    assert [f(p) for p in range(10)] == list(range(10))


def test_one_above_increments_mod_10():
    f = get_mapping("one_above")
    assert [f(p) for p in range(10)] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]


def test_one_above_wraps_nine_to_zero():
    assert get_mapping("one_above")(9) == 0


def test_unknown_mapping_raises():
    with pytest.raises(ValueError):
        get_mapping("nope")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/strategy/test_mappings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.strategy'`

- [ ] **Step 3: Create the package and registry**

Create `src/strategy/__init__.py`:

```python
from src.strategy.mappings import PredictionMapping, get_mapping

__all__ = ["PredictionMapping", "get_mapping"]
```

Create `src/strategy/mappings.py`:

```python
"""Реестр чистых отображений предсказания партнёра в итоговый выбор."""

from __future__ import annotations

from typing import Callable

# Чистая функция: предсказанное число партнёра -> собственный выбор (0..9).
PredictionMapping = Callable[[int], int]

_REGISTRY: dict[str, PredictionMapping] = {
    "match": lambda p: p,
    "one_above": lambda p: (p + 1) % 10,
}


def get_mapping(name: str) -> PredictionMapping:
    """Вернуть функцию отображения по имени.

    Args:
        name: Имя отображения, зарегистрированное в реестре.

    Returns:
        Чистая функция отображения предсказания в выбор.

    Raises:
        ValueError: Если имя отображения не зарегистрировано.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"неизвестное отображение предсказания: {name!r}; "
            f"доступны: {sorted(_REGISTRY)}"
        ) from None
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/strategy/test_mappings.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/strategy/__init__.py src/strategy/mappings.py tests/strategy/test_mappings.py
git commit -m "feat: add prediction-mapping registry (match, one_above)"
```

---

## Task 2: Config fields, name pools, and validation

**Files:**
- Modify: `src/core/config.py`
- Modify: `config/example.yaml`
- Test: `tests/core/test_config_load.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/core/test_config_load.py` (and update the existing `test_load_example` assertion in the same edit — see Step 3):

```python
def test_load_example_has_name_pools():
    cfg = load_episode(EXAMPLE)
    assert len(cfg.population.first_name_pool) >= cfg.population.n_agents
    assert len(cfg.population.last_name_pool) >= cfg.population.n_agents


def test_default_play_strategy_is_direct():
    cfg = load_episode(EXAMPLE)
    assert cfg.play_strategy == "direct"
    assert cfg.prediction_mapping == "match"


def test_prediction_config_loads(tmp_path):
    f = tmp_path / "pred.yaml"
    f.write_text(textwrap.dedent(
        """
        seed: 1
        rounds: 2
        matchmaker: random
        play_strategy: prediction
        prediction_mapping: one_above
        population:
          kind: roster
          n_agents: 2
          first_name_pool: [Kurisu, Mayuri, Itaru]
          last_name_pool: [Makise, Shiina, Hashida]
          agents:
            - persona: "p"
              provider: {base_url: "http://x/v1", model: "m"}
        """
    ))
    cfg = load_episode(str(f))
    assert cfg.play_strategy == "prediction"
    assert cfg.prediction_mapping == "one_above"


def _write_pop_yaml(tmp_path, *, strategy="direct", mapping="match",
                    firsts="[Kurisu, Mayuri]", lasts="[Makise, Shiina]", n=2):
    f = tmp_path / "c.yaml"
    f.write_text(textwrap.dedent(
        f"""
        seed: 1
        rounds: 2
        matchmaker: random
        play_strategy: {strategy}
        prediction_mapping: {mapping}
        population:
          kind: roster
          n_agents: {n}
          first_name_pool: {firsts}
          last_name_pool: {lasts}
          agents:
            - persona: "p"
              provider: {{base_url: "http://x/v1", model: "m"}}
        """
    ))
    return str(f)


def test_unknown_play_strategy_raises(tmp_path):
    with pytest.raises(ValueError):
        load_episode(_write_pop_yaml(tmp_path, strategy="bogus"))


def test_unknown_prediction_mapping_raises(tmp_path):
    with pytest.raises(ValueError):
        load_episode(_write_pop_yaml(tmp_path, strategy="prediction", mapping="bogus"))


def test_pool_smaller_than_n_agents_raises(tmp_path):
    with pytest.raises(ValueError):
        load_episode(_write_pop_yaml(tmp_path, firsts="[Only]", lasts="[Makise, Shiina]"))


def test_duplicate_pool_entries_raise(tmp_path):
    with pytest.raises(ValueError):
        load_episode(_write_pop_yaml(tmp_path, firsts="[Kurisu, Kurisu]"))


def test_missing_pools_raise(tmp_path):
    f = tmp_path / "nopools.yaml"
    f.write_text(textwrap.dedent(
        """
        seed: 1
        rounds: 2
        matchmaker: random
        population:
          kind: roster
          n_agents: 2
          agents:
            - persona: "p"
              provider: {base_url: "http://x/v1", model: "m"}
        """
    ))
    with pytest.raises(ValueError):
        load_episode(str(f))
```

Also update the existing `test_defaults_applied` YAML in the same file to include pools (it currently has none and would now fail validation). Replace its `population:` block with:

```python
        population:
          kind: roster
          n_agents: 2
          first_name_pool: [Kurisu, Mayuri]
          last_name_pool: [Makise, Shiina]
          agents:
            - persona: "p"
              provider: {base_url: "http://x/v1", model: "m"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/core/test_config_load.py -v`
Expected: new tests FAIL (e.g. `AttributeError: 'PopulationCfg' object has no attribute 'first_name_pool'`, and validation tests fail because no validation exists yet).

- [ ] **Step 3: Implement config fields, validation, and fix the example**

In `src/core/config.py`, add `field` is already imported. Update `PopulationCfg`:

```python
@dataclass(frozen=True)
class PopulationCfg:
    kind: str
    n_agents: int
    agents: list[AgentSpec]          # shorter than n_agents -> cycled at build time
    first_name_pool: list[str] = field(default_factory=list)
    last_name_pool: list[str] = field(default_factory=list)
```

Update `EpisodeCfg` (add two fields at the end of the dataclass, after `max_concurrency`):

```python
    max_concurrency: int = 4
    play_strategy: str = "direct"          # "direct" | "prediction"
    prediction_mapping: str = "match"      # используется только при play_strategy="prediction"
    # NB: no db_path here — persistence lives in the separate Logger layer, not the orchestrator.
```

Update `_population_cfg` to read pools:

```python
def _population_cfg(d: dict) -> PopulationCfg:
    agents = [
        AgentSpec(persona=a["persona"], provider=_provider_cfg(a["provider"]))
        for a in d["agents"]
    ]
    return PopulationCfg(
        kind=d["kind"],
        n_agents=d["n_agents"],
        agents=agents,
        first_name_pool=d.get("first_name_pool", []),
        last_name_pool=d.get("last_name_pool", []),
    )
```

Add a validation function above `load_episode`:

```python
def _validate(d: dict) -> None:
    """Проверить конфигурацию эпизода один раз при загрузке; падать быстро.

    Raises:
        ValueError: При недопустимой стратегии, отображении или пулах имён.
    """
    strategy = d.get("play_strategy", "direct")
    if strategy not in ("direct", "prediction"):
        raise ValueError(
            f"play_strategy должен быть 'direct' или 'prediction', получено: {strategy!r}"
        )
    if strategy == "prediction":
        from src.strategy.mappings import get_mapping

        get_mapping(d.get("prediction_mapping", "match"))  # бросит ValueError при неизвестном имени

    pop = d["population"]
    n = pop["n_agents"]
    for key in ("first_name_pool", "last_name_pool"):
        if key not in pop:
            raise ValueError(f"в конфигурации population отсутствует обязательное поле {key!r}")
        pool = pop[key]
        if len(set(pool)) != len(pool):
            raise ValueError(f"{key} содержит повторяющиеся имена")
        if len(pool) < n:
            raise ValueError(f"{key} (размер {len(pool)}) меньше n_agents ({n})")
```

Update `load_episode` to call `_validate` and pass the new `EpisodeCfg` fields:

```python
def load_episode(path: str) -> EpisodeCfg:
    """Load one episode config from YAML. pyyaml resolves &anchors / *aliases itself,
    so a provider shared via *default arrives as the same dict for every agent."""
    with open(path) as f:
        d = yaml.safe_load(f)
    _validate(d)
    return EpisodeCfg(
        seed=d["seed"],
        rounds=d["rounds"],
        matchmaker=d["matchmaker"],
        population=_population_cfg(d["population"]),
        game=_game_cfg(d.get("game", {})),
        context_window=d.get("context_window"),
        idle_payoff=d.get("idle_payoff", 1.0),
        max_concurrency=d.get("max_concurrency", 4),
        play_strategy=d.get("play_strategy", "direct"),
        prediction_mapping=d.get("prediction_mapping", "match"),
    )
```

In `config/example.yaml`, add the name pools under `population:` (after `n_agents: 4`) and fix the pre-existing `max_talk_turns` mismatch by changing `max_talk_turns: 8` to `max_talk_turns: 3`:

```yaml
game:
  payoffs: {R: 3, T: 5, P: 1, S: 0}
  max_talk_turns: 3

# ... provider_default block unchanged ...

population:
  kind: roster
  n_agents: 4
  first_name_pool: [Rintarou, Kurisu, Mayuri, Itaru, Moeka, Ruka, Suzuha, Maho, Takumi, Rimi,
                    Sena, Ayase, Kaito, Akiho, Nae, Takuru, Serika, Yuuta, Pollon, Momo]
  last_name_pool:  [Okabe, Makise, Shiina, Hashida, Kiryuu, Urushibara, Amane, Hiyajo, Nishijou,
                    Sakihata, Aoi, Kishimoto, Yashio, Senomiya, Tennouji, Miyashiro, Onoe, Gamon,
                    Takarada, Amasawa]
  agents:                     # shorter than n_agents -> cycled at build time
    - {persona: "You are a pragmatic, self-interested player who tries to win.", provider: *default}
    - {persona: "You are a cautious player who values trust.", provider: *default}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/core/test_config_load.py -v`
Expected: PASS (including the previously-failing `test_load_example`, now asserting `3`)

- [ ] **Step 5: Run the full non-network suite to confirm green baseline**

Run: `uv run pytest -q -k "not ollama and not live and not smoke"`
Expected: all pass, 0 failed

- [ ] **Step 6: Commit**

```bash
git add src/core/config.py config/example.yaml tests/core/test_config_load.py
git commit -m "feat: add play_strategy/name-pool config with startup validation"
```

---

## Task 3: Add `PhaseKind.PREDICT` to the agent

**Files:**
- Modify: `src/core/agent.py`
- Test: `tests/core/test_agent.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_agent.py` (reuse whatever agent/provider construction helper the file already defines; if none, use the inline form below):

```python
async def test_predict_phase_parses_number_and_rationale():
    from src.core.agent import Agent, AgentSetup, Phase, PhaseKind
    from src.core.config import ProviderCfg
    from conftest import ScriptedProvider

    cfg = ProviderCfg(base_url="http://x/v1", model="m")
    agent = Agent("A1", AgentSetup("You are A1.", cfg),
                  ScriptedProvider(['{"number": 7, "rationale": "mid is safe"}']))
    res = await agent.act(Phase(PhaseKind.PREDICT, "predict your partner", rules="R"))
    assert res.data["number"] == 7
    assert res.data["rationale"] == "mid is safe"
    assert res.public_text is None     # PREDICT produces no public message
```

(If `tests/core/conftest.py` does not export `ScriptedProvider`, it does — confirmed present.)

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/core/test_agent.py::test_predict_phase_parses_number_and_rationale -v`
Expected: FAIL — `PhaseKind` has no `PREDICT`, or parse returns the fallback random number.

- [ ] **Step 3: Implement PREDICT handling**

In `src/core/agent.py`, add `PREDICT` to the enum:

```python
class PhaseKind(Enum):
    TALK = "talk"
    DECIDE = "decide"
    PREDICT = "predict"
```

Add a `PhaseKind.PREDICT` entry to `_CORRECTION` (same JSON shape as DECIDE):

```python
_CORRECTION = {
    PhaseKind.DECIDE: (
        "Respond with ONLY valid JSON, nothing else: "
        '{"number": <integer 0-9>, "rationale": "<short reason>"}'
    ),
    PhaseKind.PREDICT: (
        "Respond with ONLY valid JSON, nothing else: "
        '{"number": <integer 0-9>, "rationale": "<short reason>"}'
    ),
    PhaseKind.TALK: (
        "Respond with ONLY valid JSON, nothing else: "
        '{"message": "<your message>", "ready": <true|false>}'
    ),
}
```

Update `_parse` so PREDICT validates like DECIDE:

```python
def _parse(kind: PhaseKind, text: str) -> dict | None:
    obj = _extract_json_obj(text)
    if obj is None:
        return None
    if kind in (PhaseKind.DECIDE, PhaseKind.PREDICT):
        return _validate_decide(obj)
    if kind is PhaseKind.TALK:
        return _validate_talk(obj)
    return None
```

Update `_fallback` so PREDICT falls back like DECIDE:

```python
def _fallback(kind: PhaseKind) -> dict:
    if kind in (PhaseKind.DECIDE, PhaseKind.PREDICT):
        return {"number": random.randint(0, 9), "rationale": "(unparsed)"}
    return {"message": "", "ready": True}
```

`_result` already returns `public=None` for any non-TALK kind — no change needed.

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/core/test_agent.py -v`
Expected: PASS (all agent tests, including the new one)

- [ ] **Step 5: Commit**

```bash
git add src/core/agent.py tests/core/test_agent.py
git commit -m "feat: add PREDICT phase kind to agent"
```

---

## Task 4: Extract prompt builders into `src/games/prompts.py`

This is a pure refactor (move + rename existing functions, add `predict_context`). Behavior unchanged; all existing game tests stay green.

**Files:**
- Create: `src/games/prompts.py`
- Modify: `src/games/reputation_pd.py`

- [ ] **Step 1: Create the prompts module**

Create `src/games/prompts.py` (move the bodies of the current `_rules_text`, `_talk_context`, `_decide_context` from `reputation_pd.py`, drop the leading underscore, and add `predict_context`):

```python
"""Построители контекста (промптов) для игры — общие для игры и стратегий.

Не импортирует ни игру, ни стратегии, чтобы исключить циклические импорты."""

from __future__ import annotations

from src.core.config import GameCfg


def rules_text(cfg: GameCfg) -> str:
    """Статический текст правил игры (идёт в системный промпт после персоны)."""
    p = cfg.payoffs
    return (
        "You are playing a repeated game. Each round you face one partner.\n"
        "Both of you secretly pick an integer from 0 to 9 at the same time.\n"
        f"- Equal numbers: you each score {p.R:g}.\n"
        f"- Your number exactly one higher than your partner's (0 counts as following 9): "
        f"you score {p.T:g}, your partner {p.S:g}.\n"
        f"- Your partner's number exactly one higher than yours: they score {p.T:g}, you {p.S:g}.\n"
        f"- Any other case: you each score {p.P:g}.\n"
        "Before choosing you may exchange short messages. Messages are not binding; "
        "the final choice is secret and simultaneous. Maximize your own total score."
    )


def talk_context(partner: str, round: int, feed: str) -> str:
    """Контекст хода переговоров (cheap-talk)."""
    feed_block = feed if feed else "(no messages yet)"
    return (
        f"Your partner this round is {partner}. Round {round}.\n"
        f"Negotiation so far:\n{feed_block}\n\n"
        'Send a short message to your partner. Set "ready": true when you have nothing more to say.\n'
        'Respond ONLY as JSON: {"message": "<your message>", "ready": <true|false>}'
    )


def decide_context(partner: str, round: int, feed: str) -> str:
    """Контекст финального выбора числа (стратегия direct)."""
    feed_block = feed if feed else "(no messages were exchanged)"
    return (
        f"Your partner this round is {partner}. Round {round}.\n"
        f"Negotiation:\n{feed_block}\n\n"
        "Now secretly choose your number from 0 to 9.\n"
        'Respond ONLY as JSON: {"number": <0-9>, "rationale": "<short reason>"}'
    )


def predict_context(partner: str, round: int, feed: str) -> str:
    """Контекст предсказания числа партнёра (стратегия prediction)."""
    feed_block = feed if feed else "(no messages were exchanged)"
    return (
        f"Your partner this round is {partner}. Round {round}.\n"
        f"Negotiation:\n{feed_block}\n\n"
        "Predict the number your partner will secretly choose, from 0 to 9.\n"
        'Respond ONLY as JSON: {"number": <0-9>, "rationale": "<short reason>"}'
    )
```

- [ ] **Step 2: Update `reputation_pd.py` to import from prompts**

In `src/games/reputation_pd.py`:
- Add at the top: `from src.games.prompts import decide_context, rules_text, talk_context`
- Delete the now-moved module functions `_rules_text`, `_talk_context`, `_decide_context`.
- In `__init__`, change `_rules_text(cfg)` to `rules_text(cfg)`:
  ```python
  self._rules = rules if rules is not None else rules_text(cfg)
  ```
- In `_cheap_talk`, change `_talk_context(...)` to `talk_context(...)`.
- In `play_pairing`, change the two `_decide_context(...)` calls to `decide_context(...)`.

(`_render_feed`, `_public`, `_sum_usage` stay in `reputation_pd.py`.)

- [ ] **Step 3: Run the game tests to verify nothing broke**

Run: `uv run pytest tests/games/ -v`
Expected: PASS (same tests as before; pure refactor)

- [ ] **Step 4: Run the full non-network suite**

Run: `uv run pytest -q -k "not ollama and not live and not smoke"`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/games/prompts.py src/games/reputation_pd.py
git commit -m "refactor: extract prompt builders into games/prompts.py; add predict_context"
```

---

## Task 5: Strategy package — `Decision`, `DirectStrategy`, `PredictionStrategy`

**Files:**
- Create: `src/strategy/base.py`
- Create: `src/strategy/direct.py`
- Create: `src/strategy/prediction.py`
- Modify: `src/strategy/__init__.py`
- Create: `tests/strategy/conftest.py`
- Test: `tests/strategy/test_direct.py`, `tests/strategy/test_prediction.py`

- [ ] **Step 1: Add the test double and write the failing tests**

Create `tests/strategy/conftest.py`:

```python
from __future__ import annotations

from src.providers.base import Completion


class ScriptedProvider:
    """Тестовый дубль LLMProvider без сети: отдаёт ответы из очереди по порядку."""

    def __init__(self, replies: list[str], *, prompt_tokens: int = 2, completion_tokens: int = 3):
        self._queue = list(replies)
        self._pt = prompt_tokens
        self._ct = completion_tokens
        self.calls: list[tuple[str, list]] = []

    async def complete(self, *, system, messages, temperature, max_tokens) -> Completion:
        self.calls.append((system, messages))
        text = self._queue.pop(0)
        return Completion(text=text, prompt_tokens=self._pt, completion_tokens=self._ct, raw={})

    async def aclose(self) -> None:
        pass
```

Create `tests/strategy/test_direct.py`:

```python
from __future__ import annotations

from conftest import ScriptedProvider

from src.core.agent import Agent, AgentSetup
from src.core.config import ProviderCfg
from src.strategy.direct import DirectStrategy


def _agent(replies):
    cfg = ProviderCfg(base_url="http://x/v1", model="m")
    return Agent("A1", AgentSetup("You are A1.", cfg), ScriptedProvider(replies))


async def test_direct_returns_parsed_number_no_prediction():
    agent = _agent(['{"number": 6, "rationale": "because"}'])
    d = await DirectStrategy().decide(agent, "A2", round=1, feed="", rules="R")
    assert d.number == 6
    assert d.rationale == "because"
    assert d.predicted is None
    assert d.predicted_rationale is None
```

Create `tests/strategy/test_prediction.py`:

```python
from __future__ import annotations

from conftest import ScriptedProvider

from src.core.agent import Agent, AgentSetup
from src.core.config import ProviderCfg
from src.strategy.mappings import get_mapping
from src.strategy.prediction import PredictionStrategy


def _agent(replies):
    cfg = ProviderCfg(base_url="http://x/v1", model="m")
    return Agent("A1", AgentSetup("You are A1.", cfg), ScriptedProvider(replies))


async def test_prediction_maps_predicted_to_final_choice():
    agent = _agent(['{"number": 4, "rationale": "mid"}'])
    d = await PredictionStrategy(get_mapping("one_above")).decide(agent, "A2", 1, "", "R")
    assert d.predicted == 4          # predict-step output
    assert d.number == 5             # one_above mapping applied (4 -> 5)
    assert d.predicted_rationale == "mid"
    assert d.rationale == "mid"      # reasoning carried into the recorded rationale


async def test_prediction_match_mapping_is_identity():
    agent = _agent(['{"number": 8, "rationale": "high"}'])
    d = await PredictionStrategy(get_mapping("match")).decide(agent, "A2", 1, "", "R")
    assert d.predicted == 8 and d.number == 8
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/strategy/test_direct.py tests/strategy/test_prediction.py -v`
Expected: FAIL — `No module named 'src.strategy.base'` / `direct` / `prediction`.

- [ ] **Step 3: Implement the strategy package**

Create `src/strategy/base.py`:

```python
"""Протокол стратегии игры и результат решения."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.core.agent import Agent


@dataclass(frozen=True)
class Decision:
    """Результат решения агента в одной игре.

    Attributes:
        number: Итоговый выбор 0..9, который идёт в подсчёт очков.
        rationale: Обоснование, сохраняемое в память и запись.
        predicted: Предсказанное число партнёра (None для стратегии direct).
        predicted_rationale: Обоснование предсказания (None для direct).
        usage: (prompt_tokens, completion_tokens) для агрегирования в игре.
    """

    number: int
    rationale: str
    predicted: int | None = None
    predicted_rationale: str | None = None
    usage: tuple[int, int] = (0, 0)


class PlayStrategy(Protocol):
    async def decide(self, agent: Agent, partner_id: str, round: int,
                     feed: str, rules: str) -> Decision: ...


def make_strategy(cfg) -> PlayStrategy:
    """Собрать стратегию из конфигурации эпизода (play_strategy/prediction_mapping)."""
    from src.strategy.direct import DirectStrategy
    from src.strategy.mappings import get_mapping
    from src.strategy.prediction import PredictionStrategy

    if cfg.play_strategy == "direct":
        return DirectStrategy()
    if cfg.play_strategy == "prediction":
        return PredictionStrategy(get_mapping(cfg.prediction_mapping))
    raise ValueError(f"неизвестная стратегия игры: {cfg.play_strategy!r}")
```

Create `src/strategy/direct.py`:

```python
"""Прямая стратегия: агент сразу выбирает число через фазу DECIDE."""

from __future__ import annotations

from src.core.agent import Agent, Phase, PhaseKind
from src.games.prompts import decide_context
from src.strategy.base import Decision


class DirectStrategy:
    async def decide(self, agent: Agent, partner_id: str, round: int,
                     feed: str, rules: str) -> Decision:
        res = await agent.act(
            Phase(PhaseKind.DECIDE, decide_context(partner_id, round, feed), rules=rules)
        )
        return Decision(
            number=res.data["number"],
            rationale=res.data["rationale"],
            usage=res.usage,
        )
```

Create `src/strategy/prediction.py`:

```python
"""Стратегия предсказания: предсказать число партнёра, затем отобразить его в выбор."""

from __future__ import annotations

from src.core.agent import Agent, Phase, PhaseKind
from src.games.prompts import predict_context
from src.strategy.base import Decision
from src.strategy.mappings import PredictionMapping


class PredictionStrategy:
    def __init__(self, mapping: PredictionMapping):
        self._mapping = mapping

    async def decide(self, agent: Agent, partner_id: str, round: int,
                     feed: str, rules: str) -> Decision:
        res = await agent.act(
            Phase(PhaseKind.PREDICT, predict_context(partner_id, round, feed), rules=rules)
        )
        predicted = res.data["number"]
        rationale = res.data["rationale"]
        return Decision(
            number=self._mapping(predicted),
            rationale=rationale,
            predicted=predicted,
            predicted_rationale=rationale,
            usage=res.usage,
        )
```

Update `src/strategy/__init__.py`:

```python
from src.strategy.base import Decision, PlayStrategy, make_strategy
from src.strategy.mappings import PredictionMapping, get_mapping

__all__ = ["Decision", "PlayStrategy", "make_strategy", "PredictionMapping", "get_mapping"]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/strategy/ -v`
Expected: PASS (mappings + direct + prediction)

- [ ] **Step 5: Commit**

```bash
git add src/strategy/base.py src/strategy/direct.py src/strategy/prediction.py src/strategy/__init__.py tests/strategy/conftest.py tests/strategy/test_direct.py tests/strategy/test_prediction.py
git commit -m "feat: add PlayStrategy package (Decision, Direct, Prediction)"
```

---

## Task 6: Wire the strategy into the game + persist prediction data

**Files:**
- Modify: `src/games/reputation_pd.py`
- Modify: `src/core/memory.py`
- Modify: `src/games/base.py`
- Test: `tests/games/test_play_pairing.py`, `tests/core/test_memory.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/games/test_play_pairing.py` (reuses the file's existing `_agent` and `_decide` helpers):

```python
async def test_direct_strategy_leaves_predicted_none():
    g = ReputationPD(GameCfg(max_talk_turns=0))   # default DirectStrategy
    a = _agent("A1", [_decide(4)])
    b = _agent("A2", [_decide(4)])
    rec = await g.play_pairing(a, b, 1)
    assert rec.a_predicted is None and rec.b_predicted is None
    assert a.memory.entries[0].my_predicted is None


async def test_prediction_strategy_records_and_remembers_predictions():
    from src.strategy.mappings import get_mapping
    from src.strategy.prediction import PredictionStrategy

    g = ReputationPD(GameCfg(max_talk_turns=0),
                     strategy=PredictionStrategy(get_mapping("one_above")))
    a = _agent("A1", ['{"number": 4, "rationale": "pa"}'])   # predicts 4 -> chooses 5
    b = _agent("A2", ['{"number": 4, "rationale": "pb"}'])   # predicts 4 -> chooses 5
    rec = await g.play_pairing(a, b, 1)
    assert (rec.a_predicted, rec.b_predicted) == (4, 4)
    assert (rec.a_number, rec.b_number) == (5, 5)
    assert rec.outcome == "CC"
    # private scratchpad: the predicted value lives in the acting agent's memory
    assert a.memory.entries[0].my_predicted == 4
    assert "pa" in str(a.memory.entries[0])
    # the partner's prediction reasoning never leaks into a's memory entry
    assert "pb" not in str(a.memory.entries[0])
```

Append to `tests/core/test_memory.py`:

```python
def test_render_includes_prediction_line_when_present():
    from src.core.memory import Memory, MemoryEntry

    m = Memory()
    m.add(MemoryEntry(round=1, partner_id="A2", transcript=[], my_number=5,
                      my_rationale="r", partner_number=5, outcome="CC", payoff=3.0,
                      my_predicted=4))
    rendered = m.render(None)[0].content
    assert "predict" in rendered.lower()
    assert "4" in rendered
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/games/test_play_pairing.py tests/core/test_memory.py -v`
Expected: FAIL — `ReputationPD` has no `strategy` param; `PairingRecord` has no `a_predicted`; `MemoryEntry` has no `my_predicted`.

- [ ] **Step 3: Add the data fields**

In `src/core/memory.py`, add the field to `MemoryEntry` (last field, defaulted):

```python
@dataclass
class MemoryEntry:
    round: int
    partner_id: str
    transcript: list[dict]
    my_number: int
    my_rationale: str
    partner_number: int
    outcome: str
    payoff: float
    my_predicted: int | None = None
```

In `src/core/memory.py`, update `_render_entry` to add the prediction line (insert just before the `reason = ...` line):

```python
    if e.my_predicted is not None:
        lines.append(f"I predicted {e.partner_id} would pick {e.my_predicted}.")
    reason = f" (reason: {e.my_rationale})" if e.my_rationale else ""
```

In `src/games/base.py`, add the two fields to `PairingRecord` (after `usage`, defaulted so existing construction sites stay valid):

```python
    usage: dict                     # {"prompt_tokens", "completion_tokens", "calls"}
    a_predicted: int | None = None  # стратегия prediction; None для direct
    b_predicted: int | None = None
```

- [ ] **Step 4: Delegate the decide step to the strategy**

In `src/games/reputation_pd.py`:

- Update imports: drop `decide_context` from the prompts import (no longer used here); keep `rules_text, talk_context`. Add the `PlayStrategy` type import (safe — `strategy.base` imports nothing from `games`). **Do NOT** add a top-level `from src.strategy.direct import DirectStrategy` — that creates an import cycle (`strategy.direct → games.prompts → games/__init__ → reputation_pd → strategy.direct`). Import it lazily inside `__init__` instead:
  ```python
  from src.games.prompts import rules_text, talk_context
  from src.strategy.base import PlayStrategy
  ```
- Update `__init__` to accept and default the strategy (lazy import breaks the cycle):
  ```python
  def __init__(self, cfg: GameCfg, rules: str | None = None,
               strategy: PlayStrategy | None = None):
      self.cfg = cfg
      self._rules = rules if rules is not None else rules_text(cfg)
      if strategy is None:
          from src.strategy.direct import DirectStrategy  # ленивый импорт: разрывает цикл games<->strategy
          strategy = DirectStrategy()
      self._strategy = strategy
  ```
- Replace the decision block in `play_pairing` (the two `a.act(...)`/`b.act(...)` DECIDE calls through the `return PairingRecord(...)`) with:
  ```python
      da = await self._strategy.decide(a, b.id, round, feed, self._rules)
      db = await self._strategy.decide(b, a.id, round, feed, self._rules)
      x, y = da.number, db.number
      outcome, pa, pb = self.resolve(x, y)
      a.score += pa
      b.score += pb

      public = _public(transcript)
      self._remember(a, b.id, round, public, da, y, outcome, pa)
      self._remember(b, a.id, round, public, db, x, _FLIP[outcome], pb)
      usage = _sum_usage([t["usage"] for t in transcript] + [da.usage, db.usage])
      return PairingRecord(
          round=round, a_id=a.id, b_id=b.id, transcript=public,
          a_number=x, b_number=y,
          a_rationale=da.rationale, b_rationale=db.rationale,
          outcome=outcome, a_payoff=pa, b_payoff=pb, usage=usage,
          a_predicted=da.predicted, b_predicted=db.predicted,
      )
  ```
- Update `_remember` to read `Decision` fields instead of `ActResult`:
  ```python
  def _remember(self, agent, partner_id, round, public_transcript, mine, partner_number, outcome, payoff):
      agent.memory.add(
          MemoryEntry(
              round=round,
              partner_id=partner_id,
              transcript=public_transcript,
              my_number=mine.number,
              my_rationale=mine.rationale,
              partner_number=partner_number,
              outcome=outcome,
              payoff=payoff,
              my_predicted=mine.predicted,
          )
      )
  ```

(`Phase`/`PhaseKind` imports remain — `_cheap_talk` still uses them.)

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/games/ tests/core/test_memory.py -v`
Expected: PASS — including the existing pairing tests (default `DirectStrategy` preserves behavior, `test_usage_aggregated` still sees 4 calls / 8 / 12) and the new prediction tests.

- [ ] **Step 6: Run the full non-network suite**

Run: `uv run pytest -q -k "not ollama and not live and not smoke"`
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add src/games/reputation_pd.py src/core/memory.py src/games/base.py tests/games/test_play_pairing.py tests/core/test_memory.py
git commit -m "feat: delegate game decisions to PlayStrategy; record prediction data"
```

---

## Task 7: Random names replace agent ids

**Files:**
- Modify: `src/population/base.py`
- Modify: `src/population/roster.py`
- Test: `tests/population/test_roster.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/population/test_roster.py` (reuses the file's `_spec`, `_pop_cfg`, and `created` fixture; note `_pop_cfg` builds `PopulationCfg` without pools, so add a pooled variant):

```python
def _pop_cfg_named(n, specs, firsts, lasts):
    return PopulationCfg(kind="roster", n_agents=n, agents=specs,
                         first_name_pool=firsts, last_name_pool=lasts)


def test_names_replace_ids_unique_first_and_last(created):
    firsts = ["Kurisu", "Mayuri", "Itaru", "Moeka"]
    lasts = ["Makise", "Shiina", "Hashida", "Kiryuu"]
    pop = make_population(_pop_cfg_named(3, [_spec("p")], firsts, lasts)).build(random.Random(0))
    ids = pop.ids()
    assert len(ids) == 3
    first_parts = [i.split(" ")[0] for i in ids]
    last_parts = [i.split(" ")[1] for i in ids]
    assert len(set(first_parts)) == 3   # all first names unique
    assert len(set(last_parts)) == 3    # all last names unique
    assert all(f in firsts and l in lasts for f, l in zip(first_parts, last_parts))


def test_name_assignment_is_deterministic_per_seed(created):
    firsts = ["Kurisu", "Mayuri", "Itaru", "Moeka"]
    lasts = ["Makise", "Shiina", "Hashida", "Kiryuu"]
    cfg = _pop_cfg_named(3, [_spec("p")], firsts, lasts)
    ids1 = make_population(cfg).build(random.Random(7)).ids()
    ids2 = make_population(cfg).build(random.Random(7)).ids()
    ids3 = make_population(cfg).build(random.Random(99)).ids()
    assert ids1 == ids2          # same seed -> same assignment
    assert ids1 != ids3          # different seed -> different assignment


def test_empty_pools_fall_back_to_a_ids(created):
    # No pools provided -> A1.. ids (keeps programmatic construction working).
    pop = make_population(_pop_cfg(2, [_spec("p")])).build(random.Random(0))
    assert pop.ids() == ["A1", "A2"]
```

(`test_roster_cycles_personas_and_ids` — the existing test that asserts `["A1".."A5"]` — keeps passing because `_pop_cfg` supplies no pools.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/population/test_roster.py -v`
Expected: new name tests FAIL — names are still `A1..` because sampling isn't implemented; `Population.add` rejects `agent_id`.

- [ ] **Step 3: Allow explicit ids in `Population.add`**

In `src/population/base.py`, update `add`:

```python
    def add(self, setup: AgentSetup, *, agent_id: str | None = None) -> Agent:
        cfg = setup.provider_cfg
        key = (cfg.base_url, cfg.model)
        provider = self._providers.get(key)
        if provider is None:
            provider = make_provider(cfg)
            self._providers[key] = provider
        aid = agent_id if agent_id is not None else self.next_id()
        agent = Agent(aid, setup, provider, context_window=self._window)
        self._agents.append(agent)
        self._by_id[aid] = agent
        return agent
```

- [ ] **Step 4: Sample names in `RosterGenerator.build`**

Replace the body of `src/population/roster.py` with:

```python
from __future__ import annotations

from src.core.agent import AgentSetup
from src.population.base import Population


class RosterGenerator:
    """MVP population generator: an explicit list of specs, cycled up to n_agents."""

    def __init__(self, pop_cfg, *, context_window: int | None = None):
        self._cfg = pop_cfg
        self._window = context_window

    def build(self, rng) -> Population:
        """Собрать популяцию; имена сэмплируются из пулов конфигурации по rng.

        При пустых пулах имён id назначаются как A1, A2, … (резервный режим)."""
        pop = Population(context_window=self._window)
        specs = self._cfg.agents
        names = _sample_names(self._cfg, rng)
        for i in range(self._cfg.n_agents):
            spec = specs[i % len(specs)]              # shorter than n_agents -> cycle
            pop.add(AgentSetup(spec.persona, spec.provider), agent_id=names[i])
        return pop


def _sample_names(cfg, rng) -> list[str | None]:
    """Сэмплировать уникальные 'Имя Фамилия' без повторов; None при пустых пулах."""
    if not cfg.first_name_pool or not cfg.last_name_pool:
        return [None] * cfg.n_agents
    firsts = rng.sample(cfg.first_name_pool, cfg.n_agents)
    lasts = rng.sample(cfg.last_name_pool, cfg.n_agents)
    return [f"{f} {l}" for f, l in zip(firsts, lasts)]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/population/test_roster.py -v`
Expected: PASS (existing A-id tests + new name tests)

- [ ] **Step 6: Commit**

```bash
git add src/population/base.py src/population/roster.py tests/population/test_roster.py
git commit -m "feat: sample random First Last names as agent ids"
```

---

## Task 8: Inject the strategy in the orchestrator

**Files:**
- Modify: `src/core/orchestrator.py`
- Test: `tests/core/test_orchestrator.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_orchestrator.py` (reuses the file's `providers` fixture and `_run` helper; builds a prediction config directly):

```python
def _pred_cfg(n=2, rounds=1, seed=0):
    spec = AgentSpec(persona="p", provider=ProviderCfg(base_url="http://x/v1", model="m"))
    return EpisodeCfg(
        seed=seed,
        rounds=rounds,
        matchmaker="random",
        population=PopulationCfg(kind="roster", n_agents=n, agents=[spec]),
        game=GameCfg(max_talk_turns=0),
        play_strategy="prediction",
        prediction_mapping="one_above",
    )


async def test_prediction_strategy_threaded_through_orchestrator(providers):
    # FixedProvider replies number=4 for every call -> predict 4 -> one_above -> choose 5.
    records = []
    await _run(_pred_cfg(n=2, rounds=1), observer=lambda r, p, recs: records.extend(recs))
    assert len(records) == 1
    rec = records[0]
    assert rec.a_predicted == 4 and rec.b_predicted == 4
    assert rec.a_number == 5 and rec.b_number == 5
    assert rec.outcome == "CC"
```

(The default `FixedProvider` in this file already returns `{"number": 4, "rationale": "r"}`.)

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/core/test_orchestrator.py::test_prediction_strategy_threaded_through_orchestrator -v`
Expected: FAIL — `a_predicted` is `None` because `run_episode` still builds `ReputationPD(cfg.game)` with the default `DirectStrategy`.

- [ ] **Step 3: Build and inject the strategy in `run_episode`**

In `src/core/orchestrator.py`:
- Add the import: `from src.strategy.base import make_strategy`
- Change the game construction line `game = ReputationPD(cfg.game)` to:
  ```python
  game = ReputationPD(cfg.game, strategy=make_strategy(cfg))
  ```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/core/test_orchestrator.py -v`
Expected: PASS (existing direct-run tests + the new prediction test)

- [ ] **Step 5: Run the full non-network suite**

Run: `uv run pytest -q -k "not ollama and not live and not smoke"`
Expected: all pass, 0 failed

- [ ] **Step 6: Commit**

```bash
git add src/core/orchestrator.py tests/core/test_orchestrator.py
git commit -m "feat: select play strategy from config in the orchestrator"
```

---

## Final verification

- [ ] **Run the entire non-network suite**

Run: `uv run pytest -q -k "not ollama and not live and not smoke"`
Expected: all pass, 0 failed (was 1 pre-existing failure before Task 2).

- [ ] **Smoke-run the demo path compiles (optional, no Ollama needed for import)**

Run: `uv run python -c "import examples.orchestrator_demo"`
Expected: no import errors (confirms wiring imports resolve).

- [ ] **Manual config check: a prediction config loads**

Run:
```bash
uv run python -c "from src.core.config import load_episode; c=load_episode('config/example.yaml'); print(c.play_strategy, len(c.population.first_name_pool))"
```
Expected: prints `direct 20`

---

## Notes for the implementer

- **No import cycles:** `games/__init__.py` eagerly imports `reputation_pd`, so importing `src.games.prompts` runs that `__init__`. To avoid a partial-init cycle, `reputation_pd` imports `strategy.base` (safe) at module top but imports `strategy.direct` **lazily inside `__init__`** (see Task 6 Step 4). `games.prompts` imports nothing from `strategy` or `reputation_pd`; `strategy.base`/`strategy.mappings` import nothing from `games`; `strategy.direct`/`strategy.prediction` import `games.prompts` (safe, since `reputation_pd` no longer top-level-imports them). Never add a `reputation_pd` import inside any `strategy` module, and never add a top-level `from src.strategy.direct import ...` to `reputation_pd`.
- **Default strategy:** `ReputationPD(cfg.game)` with no `strategy=` must keep behaving exactly as the old direct path — that is what keeps the pre-existing `tests/games/` suite green. Do not remove the `DirectStrategy()` default.
- **Pools are required in YAML, optional in the dataclass:** validation in `load_episode` enforces presence for config files; `PopulationCfg`'s `default_factory=list` keeps programmatic construction (in orchestrator/roster tests) working with `A1/A2` fallback ids. This split is intentional.
- **Russian vs English:** error messages and docstrings Russian; LLM-facing prompt strings English (must match existing prompts seen by partners).
