# Архитектура кода — MVP

Дата: 2026-05-28. Это **MVP-срез** полного дизайна из `agent-games-plan.md`.
Оба файла держим в синхроне: любая детализация — правка в обоих.

Цель MVP: один эпизод игры «число 0–9» гоняется end-to-end из конфига и пишет
артефакты. Всё «сложное» (другие матчинги, notes, REFLECT, эпохи, свип-сетка,
анализ репутации) — отложено, но швы оставлены чистыми.

## 1. Что в MVP / что отложено

**В MVP:**
- Игра `reputation_pd`: число 0–9, cheap talk с флагом `ready`, тайный выбор, платежи PD.
- Популяция: пакет `population/` — параметризуемый генератор (как `matchmaking`): `PopulationGenerator` (Protocol) + одна реализация `roster`. Продукт — мутабельный ростер `Population` (устойчивые ID). Плоские раунды (без эпох).
- `Matchmaker` (интерфейс) + одна реализация `random`.
- `Agent.act(phase)` с фазами `TALK`, `DECIDE`. Разделение `AgentSetup` / `Agent`.
- Память: сырая + `context_window` (дефолт ∞). Без notes.
- Один провайдер: `OpenAICompatibleProvider` (`base_url`/`api_key`/`model`), дефолт — локальная Ollama.
- Оркестратор: цикл раундов, параллельные независимые пары, чекпоинт + resume.
- Конфиг-файл = один эпизод. Хранилище: **SQLite** (один файл): таблицы `runs` / `agents` / `rounds` / `pairings`; транскрипты — JSON-колонкой; метрики.

**Отложено (заглушки/чистые швы, не кодим):** матчинги `scheduled`/`choice`,
генераторы популяции `homogeneous`/`mixed`, notes (фаза `NOTES`), `REFLECT`,
эпохи + `SelectionPolicy`, свип-сетка + возобновляемость-по-хешу + параллельные
эпизоды, модуль анализа репутации, нативный Anthropic-провайдер с prompt caching.

## 2. Дерево модулей (MVP-подмножество)

```
src/                     # движок — сам по себе пакет; импорт `from src.…`
  core/
    config.py            # dataclasses + загрузка YAML
    agent.py             # AgentSetup, Agent, Phase, PhaseKind, ActResult
    memory.py            # Memory, MemoryEntry (render с окном)
    orchestrator.py      # run_episode(): цикл раундов, параллель, txn/раунд
    storage.py           # SQLite: схема + запись (txn/раунд) + resume (SQL)
    metrics.py           # доля кооперации + прокси gossip (SQL по pairings)
  providers/
    base.py              # LLMProvider (Protocol), Message, Completion
    openai_compat.py     # OpenAICompatibleProvider (httpx, ретраи)
  population/
    base.py              # Population (мутабельный ростер), PopulationGenerator (Protocol), make_population()
    roster.py            # RosterGenerator (явный список специй, циклически до N)
  games/
    base.py              # Game (Protocol), PairingRecord
    reputation_pd.py     # ReputationPD: cheap talk + DECIDE + правило 0–9 + PD
  matchmaking/
    base.py              # Matchmaker (Protocol), RoundPlan
    random_mm.py         # RandomMatchmaker
  cli.py                 # `run --config ...`
config/example.yaml
pyproject.toml           # httpx, pyyaml ; dev: pytest, pytest-asyncio   (sqlite3 — stdlib)
analysis/                # аналитика — отдельно от движка (импортит src; пост-MVP)
```

> Популяция устроена как `matchmaking`: подключаемый генератор, выбираемый по
> `kind` в конфиге. `population/base.py` держит две вещи — рантайм-структуру
> `Population` (мутабельный ростер: список живых агентов + устойчивые, не
> переиспользуемые ID) и протокол `PopulationGenerator` (`build(rng) ->
> Population`); фабрика `make_population(cfg.population)` выбирает реализацию.
> В MVP реализация одна — `roster` (явный список специй персон/провайдеров,
> циклически до N), остальные (`homogeneous`, `mixed`) — чистый шов. Мутаторы
> `Population` (`add` / `remove` / `next_id`) в MVP не вызываются — через них
> пост-MVP отбор (`SelectionPolicy`, §8) рождает потомков и выбраковывает слабейших.

## 3. Ключевые интерфейсы (сигнатуры)

### providers/base.py
```python
@dataclass
class Message: role: str; content: str          # system | user | assistant

@dataclass
class Completion: text: str; prompt_tokens: int; completion_tokens: int; raw: dict

class LLMProvider(Protocol):
    async def complete(self, *, system: str, messages: list[Message],
                       temperature: float, max_tokens: int) -> Completion: ...
```

### providers/openai_compat.py
```python
class OpenAICompatibleProvider:                 # покрывает Ollama / Cerebras / Gemini / OpenAI / ...
    def __init__(self, base_url: str, api_key: str, model: str): ...
    async def complete(self, *, system, messages, temperature, max_tokens) -> Completion
    # POST {base_url}/chat/completions; backoff-ретраи на 429/5xx; токены из usage
```

### core/agent.py
```python
class PhaseKind(Enum): TALK; DECIDE             # REFLECT/NOTES/CHOOSE_PARTNER/CONSENT — пост-MVP

@dataclass
class Phase: kind: PhaseKind; context: str; rules: str = ""   # context = ситуация + инструкция вывода; rules = свод правил (Игра пишет, Агент кладёт в system)
#   для TALK ситуация = лента переговоров, где КАЖДАЯ реплика выводится со своим флагом `ready`

@dataclass
class ActResult: public_text: str | None; data: dict; usage: tuple[int, int]
#   TALK  -> data = {"message": str, "ready": bool}, public_text = message
#   DECIDE-> data = {"number": int(0..9), "rationale": str}, public_text = None

@dataclass(frozen=True)
class AgentSetup: persona: str; provider_cfg: ProviderCfg   # «генотип» (для эволюции потом)

class Agent:
    id: str; setup: AgentSetup; memory: Memory; score: float; parse_failures: int
    def __init__(self, id, setup, provider: LLMProvider, *, context_window: int | None = None): ...
    async def act(self, phase: Phase) -> ActResult
    #   system = «You are agent {id}» + persona + phase.rules; messages = memory.render(context_window) + [user(phase.context)]
    #   вызывает provider.complete; парсит JSON по phase.kind; кривой ответ → ретрай×2 + безопасный дефолт (parse_failures++)
```

### core/memory.py
```python
@dataclass
class MemoryEntry:
    round: int; partner_id: str
    transcript: list[dict]                      # [{speaker, text, ready}]
    my_number: int; my_rationale: str; partner_number: int
    outcome: str; payoff: float

class Memory:
    entries: list[MemoryEntry]
    def add(self, e: MemoryEntry) -> None
    def render(self, window: int | None) -> list[Message]   # последние window записей (None = все)
```

### population/base.py + roster.py
```python
class Population:                                # рантайм-структура: мутабельный ростер живых агентов
    agents: list[Agent]                          # текущий состав; порядок устойчив
    #   внутри: монотонный счётчик ID + кэш провайдеров по (base_url, model)
    def ids(self) -> list[str]                   # ID в порядке ростера
    def get(self, agent_id: str) -> Agent
    def __iter__(self) -> Iterator[Agent]; def __len__(self) -> int
    def next_id(self) -> str                     # A1, A2, …; счётчик только растёт — ID не переиспользуются
    def add(self, setup: AgentSetup) -> Agent    # id=next_id(); провайдер из setup (кэш base_url+model);
    #                                              память пустая, score=0; append; вернуть нового агента
    # --- мутаторы ниже в MVP не вызываются; чистый шов под отбор (§8) ---
    def remove(self, agent_id: str) -> None      #   убрать из ростера (история/транскрипты остаются в storage)
    def replace(self, dead_ids: list[str], parent_setups: list[AgentSetup]) -> list[Agent]
    #   remove(dead_ids) + add(parent_setups) — связка, которую зовёт пост-MVP SelectionPolicy

class PopulationGenerator(Protocol):             # КАК собирается стартовый состав — сменная стратегия
    def build(self, rng) -> Population: ...       #   rng — для случайных композиций (mixed); roster детерминирован

class RosterGenerator:                           # MVP-реализация: явный список специй, циклически до N
    def __init__(self, pop_cfg): ...             #   pop_cfg = блок `population`: n_agents, agents[] (persona+provider)
    def build(self, rng) -> Population:
    #   pop = Population()
    #   for i in range(pop_cfg.n_agents):
    #     spec = pop_cfg.agents[i % len(pop_cfg.agents)]    # короче N — берём по кругу
    #     pop.add(AgentSetup(spec.persona, spec.provider))  # выдаст A1..An по порядку
    #   return pop                                          # счётчик ID встал на A{n+1}

def make_population(pop_cfg) -> PopulationGenerator:  # по pop_cfg.kind: "roster" -> RosterGenerator(pop_cfg)
    #   ("homogeneous", "mixed") — пост-MVP реализации того же протокола
```

### games/base.py + reputation_pd.py
```python
@dataclass
class PairingRecord:
    round: int; a_id: str; b_id: str
    transcript: list[dict]
    a_number: int; b_number: int; a_rationale: str; b_rationale: str
    outcome: str; a_payoff: float; b_payoff: float; usage: dict

class Game(Protocol):
    async def play_pairing(self, a: Agent, b: Agent, round: int) -> PairingRecord: ...
    #   no rng: ориентацию (кто открывает cheap-talk) задаёт матчер порядком в паре — a открывает

class ReputationPD:                              # реализация Game
    def __init__(self, cfg: GameCfg): ...        # payoffs R,T,P,S; max_talk_turns; talk_stop_rule
    async def play_pairing(self, a, b, round) -> PairingRecord
    def resolve(self, x: int, y: int) -> tuple[str, float, float]
    #   x==y            -> ("CC", R, R)
    #   x==(y+1)%10     -> ("DC", T, S)          # x обманул y
    #   y==(x+1)%10     -> ("CD", S, T)
    #   иначе           -> ("DD", P, P)
```

### matchmaking/base.py + random_mm.py
```python
@dataclass
class RoundPlan: pairings: list[tuple[str, str]]; idle: list[str]; events: list[dict]

class Matchmaker(Protocol):
    def setup(self, agent_ids: list[str], rng, cfg) -> None: ...
    async def plan_round(self, agent_ids: list[str], round: int, actor) -> RoundPlan: ...
    #   actor — колбэк вызова агента (для интерактивных матчингов; в random не используется)

class RandomMatchmaker:                          # перемешать, разбить на пары; нечётный -> один idle
    ...
```

### core/orchestrator.py
```python
async def run_episode(cfg: EpisodeCfg) -> None:
    rng = Random(cfg.seed)
    pop  = make_population(cfg.population).build(rng)   # выбрать генератор по kind → собрать ростер A1..An
    game = ReputationPD(cfg.game)
    mm   = make_matchmaker(cfg.matchmaker)       # "random"
    st   = Storage(cfg.db_path, cfg)             # connect + init schema (WAL); пишет runs/agents
    start = st.resume_point()                    # MAX(round_idx)+1 или 0
    replay_memory(pop, st.iter_pairings(upto=start))          # при resume
    mm.setup(pop.ids(), rng, cfg)
    sem = Semaphore(cfg.max_concurrency)
    for r in range(start, cfg.rounds):
        plan = await mm.plan_round(pop.ids(), r, actor=None)
        recs = await gather(*[guarded(game.play_pairing(pop.get(a), pop.get(b), r), sem)
                              for a, b in plan.pairings])    # порядок пары = кто открывает
        for c in plan.idle: pop.get(c).score += cfg.idle_payoff
        st.write_round(r, plan, recs)            # одна транзакция (план раунда + все пары)
    st.write_scores(pop); st.write_metrics(compute_metrics(st))
```

### core/storage.py
```python
class Storage:
    def __init__(self, db_path: str, cfg: EpisodeCfg): ...   # connect; init schema (WAL); пишет runs/agents
    def resume_point(self) -> int                            # MAX(round_idx)+1 для run_id, иначе 0
    def write_round(self, r: int, plan: RoundPlan,
                    recs: list[PairingRecord]) -> None        # ОДНА транзакция: rounds + pairings
    def iter_pairings(self, upto: int) -> Iterable[PairingRecord]   # для replay_memory при resume
    def write_scores(self, pop: Population) -> None          # итерирует ростер
    def write_metrics(self, m: dict) -> None
```

## 4. Поток данных одного раунда

```
Matchmaker.plan_round → pairings
  для каждой пары (параллельно, независимые пары):
    ReputationPD.play_pairing(a, b):
      cheap talk: a.act(TALK) ⇄ b.act(TALK) ... до стопа (talk_stop_rule, дефолт both_ready_latch) или max_talk_turns
      decision:   a.act(DECIDE), b.act(DECIDE)            # независимо
      resolve(x, y) → outcome, payoffs
      a.memory.add(...); b.memory.add(...); a.score+=; b.score+=
      → PairingRecord
Storage.write_round(...) — одна транзакция (план раунда + все пары) после gather
конец раундов → write_scores + write_metrics
```

Параллелизм безопасен: матчинг — это разбиение, поэтому две пары никогда не трогают
одного агента → гонок по памяти/очкам нет.

## 5. Конфиг (пример `example.yaml`)

```yaml
seed: 42
rounds: 20
matchmaker: random
context_window: null        # ∞
idle_payoff: 1
max_concurrency: 4
db_path: runs/dev.db
game:
  payoffs: {R: 3, T: 5, P: 1, S: 0}
  max_talk_turns: 6
provider_default: &default
  base_url: http://localhost:11434/v1     # локальная Ollama
  api_key_env: OLLAMA_KEY                  # заглушка для локалки
  model: qwen3:8b
  temperature: 0.7
  max_tokens: 512
population:
  kind: roster               # MVP: единственная реализация генератора (как matchmaker: random)
  n_agents: 6
  agents:                    # если короче n_agents — циклически повторяется
    - {persona: "Ты прагматичный игрок.", provider: *default}
    - {persona: "Ты осторожный игрок.",   provider: *default}
```

## 6. Хранилище (MVP) — SQLite

Одна БД SQLite (stdlib, файл `db_path`). Транскрипты — JSON-колонкой (verbatim).
Одна БД держит много прогонов (под будущий свип). Схема:

```sql
runs(run_id PK, config JSON, seed, status, model_info, started_at, finished_at)
agents(run_id, agent_id, setup JSON, final_score,         PRIMARY KEY(run_id, agent_id))
rounds(run_id, round_idx, plan JSON,                       PRIMARY KEY(run_id, round_idx))
pairings(run_id, round_idx, pair_idx, a_id, b_id,
         a_number, b_number, a_rationale, b_rationale,
         outcome, a_payoff, b_payoff, transcript JSON, usage JSON,
         PRIMARY KEY(run_id, round_idx, pair_idx))
```

- **Запись без гонок:** пары раунда играются параллельно, но пишутся одной
  транзакцией после `gather()` (раунды последовательны) → single-writer SQLite не
  конфликтует. WAL-режим для чтения во время анализа.
- **Resume:** `SELECT MAX(round_idx) FROM rounds WHERE run_id=?` → старт со
  следующего; память агентов восстанавливается реплеем `pairings` до этой точки.
- **Метрики (§7)** считаются SQL-запросами по `pairings` (можно кэшировать в `runs`).

## 7. Метрики (MVP)

- Доля исходов CC / off-by-one / DD по раундам и итоговая.
- Прокси gossip: доля реплик cheap talk, упоминающих ID агента вне текущей пары
  (по списку известных ID; без LLM).

## 8. Чистые швы под отложенное

- **Матчинг:** новые реализации `Matchmaker` (`scheduled`, `choice`) — без правок
  оркестратора; `plan_round` уже async и принимает `actor`.
- **Генератор популяции:** новые реализации `PopulationGenerator` (`homogeneous`,
  `mixed`) — выбираются по `population.kind`, без правок оркестратора (он зовёт
  `make_population(...).build(rng)`).
- **Фазы:** `PhaseKind` расширяется (`REFLECT`, `NOTES`, `CHOOSE_PARTNER`,
  `CONSENT`) — `Agent.act` обрабатывает по `kind`.
- **Память:** `notes` добавляется вторым ярусом в `Memory.render`.
- **Эволюция:** пакет `population/` уже выделен (мутабельный `Population` с
  `add` / `remove` / `replace` / `next_id`); добавляются `SelectionPolicy`
  (`population/selection.py`, зовёт эти мутаторы) + эпоха-обёртка над циклом
  раундов. `AgentSetup` отделён от `Agent` → копируется в потомка
  (provider/persona/temp), потомку — свежий ID и пустая память.
- **Свип:** `run_episode(cfg)` оборачивается генератором сетки конфигов + повторы.
- **Провайдер:** нативный `AnthropicProvider` (prompt caching) рядом с
  `OpenAICompatibleProvider` по тому же протоколу.
```
