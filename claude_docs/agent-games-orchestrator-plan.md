# Анализ и план реализации: слой Оркестратора (MVP, in-memory)

Дата: 2026-06-04. Пятый слой разработки (снизу вверх), стоит на готовых слоях
Провайдера, Агента+Памяти, Игры и Матчинга. Опирается на `agent-games-mvp-arch.md`
§2–§5 (дерево, интерфейсы `population`/`orchestrator`, поток раунда, конфиг).
**При изменениях контракта** (`EpisodeCfg`, `Population`, `run_episode`) — синхронно
править arch. Идём **срезами** (§6).

**Важно — разделение со слоем Логгера.** Хранилище (SQLite, схема таблиц, запись),
**resume** и **replay памяти** в этот слой **НЕ входят**. Они вынесены в **отдельный
слой «Логгер»** (отдельный модуль + отдельный план), который владеет организацией
таблиц и подпишется на оркестратор через шов-`observer` (§1.4, §7). Поэтому
оркестратор сейчас **in-memory**: гоняет эпизод и возвращает результат в память; в БД
ничего не пишет и при падении просто бросает (fail-fast, без resume — resume появится
вместе с Логгером).

**Принятые ранее решения, на которые опирается слой:**
- Матчеру — **собственный** `rng` `Random(f"{seed}:matchmaker")` (M1).
- **fail-fast** при сбое провайдера: исключение валит эпизод (C2). Resume — в Логгере.
- `idle_payoff` — конфигурируемый, дефолт `1` (= `P`) (C3).
- Ориентация пары значима: `play_pairing(a, b)`, `a` открывает; `Game` без `rng`.
- Язык промптов — английский. **Метрик в движке нет** — пост-хок в `analysis/` (D3).
- **CLI не делаем** — точка входа и демонстрация = пример `examples/orchestrator_demo.py` (D5).

## 0. Зачем Оркестратор последним (в MVP)

Слой 5 в `-explained` — «гонять много раундов на популяции». Нижние слои умеют свой
кусок; оркестратор склеивает их в **эпизод**: собрать популяцию из конфига → в каждом
раунде матчер строит пары → независимые пары играются параллельно → начисляются очки →
вернуть результат. Демо `examples/matchmaking_demo.py` уже делает это руками;
оркестратор формализует цикл и добавляет конфиг-файл как единицу эксперимента,
provider-кэш и шов под будущий Логгер.

## 1. Анализ: что нетривиально

Цикл раундов прост. Нетривиальны четыре узла.

1. **Провайдеры кэшируются по `(base_url, model)` (решение D4).** Популяция из N агентов
   на одной модели не должна открывать N HTTP-клиентов. `Population.add(setup)` строит
   провайдер через `make_provider(setup.provider_cfg)` и **кэширует по `(base_url,
   model)`**; агенты с одинаковым конфигом делят один клиент (пул соединений). Следствие:
   **закрывать (`aclose`) надо каждый уникальный провайдер ровно один раз** — поэтому
   жизненный цикл держит `Population` (она их создаёт): `Population.aclose()` закрывает
   все клиенты из кэша в `finally` оркестратора.

2. **Параллельная игра пар безгоночна.** Пары раунда играются через `asyncio.gather`, но
   матчинг — это **разбиение** (гарантия слоя Матчинга) → две пары не трогают одного
   агента → нет гонок по `score`/памяти. Параллелизм ограничивает `Semaphore(max_concurrency)`.

3. **fail-fast (C2).** `gather` по умолчанию пробрасывает первое исключение → эпизод
   падает → `pop.aclose()` в `finally`. Без хранилища «доиграть/возобновить» сейчас
   нечем — это придёт с Логгером (resume). В MVP падение = падение.

4. **Шов под Логгер — `observer` (решение: подтверждён).** `run_episode` принимает
   необязательный колбэк `observer(round, plan, recs)` (sync или async), вызываемый в
   конце каждого раунда. По умолчанию его нет (чистый in-memory прогон). Будущий Логгер
   реализует `observer`, который пишет раунд в БД одной транзакцией — **без переписывания
   цикла**. Так же позже на нём вырастет resume (Логгер отдаёт стартовый раунд, replay
   восстанавливает память — детали в плане Логгера).

5. **`run_episode` ничего не возвращает — драйвер с побочными эффектами.** Единственный
   канал вывода — `observer` (факты по раундам). Финальное состояние (очки/память) живёт
   на агентах: **популяцию строит и держит вызывающий**, передаёт её в `run_episode`,
   после прогона сам читает очки/память и сам закрывает (`aclose`). Один канал вывода,
   явное владение, без дублирующего возврата. (`records` в возврате дублировали бы поток
   `observer` — поэтому их нет.)

6. **Загрузка конфига с YAML-якорями.** `example.yaml` использует `&default`/`*default`
   (один провайдер на всех). `pyyaml` резолвит якоря сам → обычные вложенные `dict`.
   Загрузчик собирает из них вложенные dataclass'ы; валидация лёгкая (нет обязательного
   поля → `KeyError`). `context_window` — глобальный (`EpisodeCfg`, дефолт ∞), доходит до
   каждого `Agent` через `Population`.

## 2. Границы слоя (scope)

**В MVP (оркестратор):**
- `EpisodeCfg` (+ `PopulationCfg`, `AgentSpec`) и `load_episode(path)` в `core/config.py`.
  **Без `db_path`** — он появится в конфиге вместе с Логгером.
- `population/`: `Population` (мутабельный ростер + provider-кэш + `aclose`),
  `PopulationGenerator` (Protocol), `RosterGenerator`, `make_population`.
- `core/orchestrator.py`: `run_episode(cfg, pop, *, observer=None) -> None`, `_guarded`.
  Популяцию строит, инспектирует и закрывает **вызывающий**.
- `config/example.yaml`; `examples/orchestrator_demo.py` (точка входа/демо).
- Тесты: загрузка конфига; ростер + provider-кэш + `aclose`; end-to-end на
  `ScriptedProvider`; Ollama-smoke (мини-эпизод).

**НЕ в этом слое (отдельный слой Логгера + швы, §7):**
- **Storage (SQLite), схема таблиц, запись, `run_id`** — слой Логгера.
- **resume + replay памяти + рефактор `apply_outcome` в Игре (D2)** — слой Логгера.
- **Любые метрики** (доли исходов, gossip, семантика) — `analysis/`, пост-хок (D3).
- **Логирование L2/L3** (каждый вызов LLM / сырой HTTP) — слой Логгера / его шов (D6).
- **CLI** — не делаем; точка входа = пример (D5).
- Мутаторы `Population.remove/replace` + `SelectionPolicy` + эпохи (эволюция).
- Генераторы `homogeneous`/`mixed`; матчинги `scheduled`/`choice`.

## 3. Раскладка файлов

```
src/
  core/
    config.py        # + AgentSpec, PopulationCfg, EpisodeCfg (без db_path), load_episode()
    orchestrator.py  # run_episode(cfg, pop, *, observer) -> None, _guarded()
  population/
    __init__.py
    base.py          # Population, PopulationGenerator (Protocol), make_population
    roster.py        # RosterGenerator
config/
  example.yaml
examples/
  orchestrator_demo.py   # load_episode -> run_episode -> печать таблицы (точка входа)
tests/
  core/        test_config_load.py, test_orchestrator.py
  population/  test_roster.py
  smoke:       test_smoke_orchestrator_ollama.py
```

## 4. Интерфейсы (сигнатуры)

```python
# core/config.py  (добавления)
@dataclass(frozen=True)
class AgentSpec:        persona: str; provider: ProviderCfg
@dataclass(frozen=True)
class PopulationCfg:    kind: str; n_agents: int; agents: list[AgentSpec]
@dataclass(frozen=True)
class EpisodeCfg:
    seed: int; rounds: int; matchmaker: str
    population: PopulationCfg; game: GameCfg
    context_window: int | None = None
    idle_payoff: float = 1.0
    max_concurrency: int = 4
def load_episode(path: str) -> EpisodeCfg: ...     # yaml.safe_load -> вложенные dataclass'ы
```

```python
# population/base.py + roster.py
class Population:
    agents: list[Agent]
    def ids(self) -> list[str]; def get(self, id) -> Agent
    def __iter__(self); def __len__(self)
    def next_id(self) -> str                        # A1, A2, … монотонно, без переиспользования
    def add(self, setup: AgentSetup) -> Agent       # provider из кэша (base_url, model); window из cfg
    async def aclose(self) -> None                  # закрыть КАЖДЫЙ уникальный провайдер один раз
    # --- швы (в MVP не зовутся): remove(id), replace(dead_ids, parent_setups) ---

class PopulationGenerator(Protocol):
    def build(self, rng) -> Population: ...
class RosterGenerator:
    def __init__(self, pop_cfg: PopulationCfg, *, context_window=None): ...
    def build(self, rng) -> Population               # персоны циклически до n_agents -> A1..An
def make_population(pop_cfg, *, context_window=None) -> PopulationGenerator   # "roster" -> RosterGenerator
```

```python
# core/orchestrator.py
Observer = Callable[[int, RoundPlan, list[PairingRecord]], None | Awaitable[None]]
#   единственный канал вывода; Логгер реализует его и пишет раунд в БД

async def run_episode(cfg: EpisodeCfg, pop: Population, *, observer: Observer | None = None) -> None: ...
#   побочные эффекты: мутирует агентов (очки/память) + эмитит раунды в observer.
#   pop строит/инспектирует/закрывает вызывающий (из cfg.population).
```

## 5. Ключевой алгоритм — `run_episode` (in-memory)

```python
async def run_episode(cfg, pop, *, observer=None) -> None:    # pop построил вызывающий
    game = ReputationPD(cfg.game)
    mm   = make_matchmaker(cfg.matchmaker)
    mm.setup(pop.ids(), random.Random(f"{cfg.seed}:matchmaker"), cfg)   # M1: свой rng
    sem  = asyncio.Semaphore(cfg.max_concurrency)
    for r in range(cfg.rounds):
        plan = await mm.plan_round(pop.ids(), r, actor=None)
        recs = await asyncio.gather(*[                              # fail-fast (C2)
            _guarded(game.play_pairing(pop.get(a), pop.get(b), r), sem)
            for a, b in plan.pairings])
        for c in plan.idle:
            pop.get(c).score += cfg.idle_payoff                    # C3
        if observer is not None:                                   # единственный канал вывода
            res = observer(r, plan, recs)
            if inspect.isawaitable(res):
                await res

# вызывающий владеет популяцией (строит, читает очки/память, закрывает):
#   pop = make_population(cfg.population, context_window=cfg.context_window).build(Random(cfg.seed))
#   try:     await run_episode(cfg, pop, observer=...)   # факты раундов — в observer
#   finally: await pop.aclose()
```
`_guarded(coro, sem)`: `async with sem: return await coro`.

`RosterGenerator.build`: `for i in range(n_agents): spec = agents[i % len(agents)];
pop.add(AgentSetup(spec.persona, spec.provider))` → ID `A1..An`, провайдеры из кэша.

## 6. Срезы (порядок реализации)

**Срез 1 — `EpisodeCfg` + `load_episode` (чистый, без LLM/БД).**
- Dataclass'ы (без `db_path`) + загрузчик YAML→dataclass; резолв якорей; дефолты.
- `config/example.yaml`. Тест `test_config_load.py`: грузим пример (поля, общий провайдер
  через якорь, дефолты); минимальный конфиг → дефолты; нет обязательного → `KeyError`.
- **DoD:** пример грузится в `EpisodeCfg`; тесты зелёные.

**Срез 2 — `population/` (ростер + provider-кэш + aclose).**
- `Population` (`add`/`ids`/`get`/`__iter__`/`__len__`/`next_id`/`aclose`),
  `RosterGenerator`, `make_population`.
- Тесты `test_roster.py`: персоны циклически до N, ID `A1..An`; **один провайдер на
  `(base_url, model)`** (идентичность объекта), разные модели → разные; `context_window`
  доходит до агента; `aclose` закрывает каждый уникальный клиент один раз (мок/счётчик).
- **DoD:** ростер детерминирован; кэш и закрытие корректны.

**Срез 3 — `run_episode(cfg, pop) -> None` + observer-шов (end-to-end на `ScriptedProvider`).**
- `_guarded`, `Semaphore`, `idle_payoff`, вызов `observer` по раунду; популяция — у вызывающего.
- Тест `test_orchestrator.py`: крошечный конфиг (N=3, rounds=2) на `ScriptedProvider` →
  записи раундов собраны через `observer`; очки читаются из своего `pop`; `idle_payoff`
  начислен; `observer` вызван по разу на раунд с верными `(r, plan, recs)`; при сбое
  провайдера `run_episode` бросает (fail-fast), вызывающий закрывает провайдеров.
- **DoD:** эпизод гоняется в памяти; параллель безопасна; шов работает.

**Срез 4 — пример + Ollama-smoke.**
- `examples/orchestrator_demo.py`: `load_episode("config/example.yaml")` → `run_episode`
  → печать таблицы очков (точка входа вместо CLI).
- `test_smoke_orchestrator_ollama.py`: мини-эпизод (N=2, rounds=1; skip без сервера) →
  корректные очки (из своего `pop`) и записи (собранные через `observer`).
- **DoD:** `pytest` зелёный; пример гоняет живой мини-эпизод.

## 7. Чистые швы под пост-MVP

- **Слой Логгера (следующий):** реализует `observer(round, plan, recs)` → пишет раунд в
  SQLite одной транзакцией; владеет схемой таблиц (`runs`/`agents`/`rounds`/`pairings`,
  спроектированных под L2) и `run_id` (D1). На нём же **resume** (стартовый раунд из БД)
  + **replay памяти** (восстановить память/очки из `pairings`), для чего в Игре
  выносится `apply_outcome` (единый источник зеркалирования, D2). **L2** (каждый вызов
  LLM) — обёртка-логгер на чокпойнте провайдеров в `Population` + таблица `calls`, теги
  через `contextvar` (D6). Всё это — аддитивно, контракт оркестратора (`observer`) не
  меняется.
- **Эволюция:** `Population.remove/replace` + `population/selection.py` +
  эпоха-обёртка; `AgentSetup` отделён от `Agent` → копируется в потомка.
- **Генераторы/матчинги:** `homogeneous`/`mixed`, `scheduled`/`choice` — по `kind`.
- **Метрики/анализ репутации:** `analysis/` читает БД Логгера (пост-хок).
- **Свип:** обёртка над `run_episode(cfg, pop)` — сетка конфигов + повторы.

## 8. Решения (закрыты 2026-06-04)

- **D3 — метрики вне движка** (только логирование/анализ-пост-хок). ✔
- **D4 — provider lifecycle у `Population`** (кэш `(base_url, model)`, `aclose`). ✔
- **D5 — CLI не делаем**, точка входа = `examples/orchestrator_demo.py`. ✔
- **D7 — нет персистентности в оркестраторе**: in-memory, без storage/resume; всё это —
  отдельный слой Логгера. ✔
- **Перенесены в план Логгера:** D1 (`run_id`), D2 (replay/`apply_outcome`),
  D6 (уровни логирования L2/L3).
