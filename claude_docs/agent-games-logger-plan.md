# Анализ и план реализации: слой Логгера (MVP, L1)

Дата: 2026-06-05. **Отдельный слой/модуль** над готовыми слоями 1–5 (Провайдер, Агент,
Игра, Матчинг, Оркестратор). Подписывается на оркестратор через шов
`observer(round, plan, recs)` и **владеет организацией таблиц** SQLite. Пересматривает
хранилище из `agent-games-mvp-arch.md` §6 (схема теперь нормализованная, §4 ниже) — при
изменениях контракта (схема, `Storage`) синхронно править arch. Идём **срезами** (§6).

**Принятые решения (2026-06-04/05):**
- Один SQLite-файл, много прогонов. Схема **нормализованная**, кроме конфигов.
- **A:** операционные данные нормализованы (`messages`, `idle`, плоский `usage`);
  **конфиги — JSON** (`runs.config`, `agents.provider`), чтобы эволюция конфига не
  дёргала схему. Персона — обычная колонка.
- **B:** `final_score` храним (авторитетное значение из памяти; пересчёт из
  `pairings`+`idle` — для проверки целостности).
- **C:** `run_id = sha256(каноничный JSON конфига)[:16]` (тот же конфиг+seed → тот же
  `run_id` → resume/дедуп).
- **D:** без `status`; таймстемпы `created_at`/`finished_at` есть.
- **D3:** метрик в движке/логгере нет — `analysis/` считает их пост-хок по этой БД.
- **D6:** сейчас **L1**; **L2** (каждый вызов LLM) — аддитивный шов (§7).
- **resume:** отложен; требует правки матчера (b) — план-функция от `(seed, round)` —
  и выноса `apply_outcome` в Игре (D2). См. §6 (срез R) и §7.

## 0. Зачем отдельным слоем

Оркестратор **in-memory** — ничего не пишет, лишь эмитит раунды в `observer` и мутирует
агентов. Логгер — единственный, кто **персистит**: реализует `observer` (пишет каждый
раунд), плюс пишет `runs`/`agents` в начале и финализирует в конце. **Движок при этом не
меняется** — только подписка на готовый шов. Отдельный модуль — потому что он владеет
схемой таблиц и (позже) resume; это чужая для оркестратора ответственность.

## 1. Анализ: что нетривиально

1. **Одна транзакция на раунд.** `observe(round, plan, recs)` пишет `rounds`+`idle`+
   `pairings`+`messages` **одной транзакцией**. Раунды последовательны → single-writer
   SQLite не конфликтует; частичного раунда в БД не бывает (важно для будущего resume).
   **WAL** — чтобы анализ мог читать во время прогона.
2. **`run_id` = хэш конфига.** Каноничная сериализация: `json.dumps(asdict(cfg),
   sort_keys=True)` → `sha256` → префикс. Стабильно между процессами (в отличие от
   `hash()`). Тот же конфиг+seed → тот же `run_id`.
3. **Что где (§4).** Конфиг/setup-провайдер — JSON; персона — колонка; `usage` — три
   плоские колонки; `transcript` → таблица `messages`; `idle` → своя таблица (idle-платежи
   не лежат в `pairings`, а для очков/анализа нужны).
4. **sqlite3 синхронный под async.** `observe` — синхронная функция; `run_episode`
   умеет sync-`observer`. Короткая блокировка event-loop между раундами на запись — для
   MVP приемлемо (записи мелкие, раунд и так ждёт LLM). Вынос в `run_in_executor` —
   потом, если понадобится.
5. **Соответствие `recs ↔ plan.pairings`.** `run_episode` строит `recs = gather(... for
   a,b in plan.pairings)`, а `gather` сохраняет порядок → `recs[i]` ↔ `plan.pairings[i]`
   → `pair_idx = enumerate(recs)`.
6. **Никаких правок нижних слоёв** для самого логирования — только реализация
   `observer`. (resume — отдельная история, трогает матчер/Игру/оркестратор, §6 срез R.)
7. **Секреты.** `provider.api_key_env` — это **имя** env-переменной, не ключ; в БД
   секретов нет.

## 2. Границы слоя (scope)

**В MVP (логирование L1):**
- `src/storage/`: DDL 6 таблиц + `Storage` (connect/WAL/FK-pragma, `run_id`, запись
  `runs`+`agents` на старте, `observe` — транзакция раунда, `finish` — таймстемп+очки).
- Тест: эпизод на `ScriptedProvider` со `Storage.observe` → все таблицы заполнены верно;
  round-trip; тот же конфиг → тот же `run_id`; FK-целостность.

**НЕ в MVP (швы/позже, §7):**
- **resume + replay** + правка матчера (b) + вынос `apply_outcome` в Игре (D2) —
  отдельный срез/шаг (§6 срез R).
- **L2** (таблица `calls` + `LoggingProvider` + `contextvar`-теги) — D6.
- Таблица `events` (интерактивные матчинги) — пока `plan.events` всегда `[]`.
- Метрики/анализ — `analysis/` поверх БД (D3).
- Другие бэкенды (Postgres/…) — один SQLite.

## 3. Раскладка файлов

```
src/
  storage/
    __init__.py      # экспорт Storage
    schema.py        # DDL (CREATE TABLE …) + init_schema(conn)
    store.py         # class Storage
tests/
  storage/
    test_storage.py  # round-trip на ScriptedProvider-эпизоде + run_id + FK
```

`db_path` приходит **снаружи** (аргумент `Storage(db_path)`); в `EpisodeCfg` его нет
(убрали на слое оркестратора — это ответственность Логгера).

## 4. Схема (DDL) — 6 таблиц, одна БД, много прогонов

```sql
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE runs (
    run_id      TEXT PRIMARY KEY,      -- sha256(каноничный конфиг)[:16]
    config      TEXT NOT NULL,         -- JSON: полный resolved EpisodeCfg
    seed        INTEGER NOT NULL,      -- денормализованная копия (частый фильтр)
    created_at  TEXT NOT NULL,         -- ISO; Шаг 0
    finished_at TEXT                   -- ISO; Шаг F (NULL пока идёт/оборвался)
);

CREATE TABLE agents (
    run_id      TEXT NOT NULL,
    agent_id    TEXT NOT NULL,         -- A1..An
    persona     TEXT NOT NULL,         -- строка персоны
    provider    TEXT NOT NULL,         -- JSON: ProviderCfg
    final_score REAL,                  -- Шаг F
    PRIMARY KEY (run_id, agent_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE rounds (
    run_id     TEXT NOT NULL,
    round_idx  INTEGER NOT NULL,       -- 0..rounds-1; resume_point = MAX(round_idx)+1
    PRIMARY KEY (run_id, round_idx),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE idle (
    run_id     TEXT NOT NULL,
    round_idx  INTEGER NOT NULL,
    agent_id   TEXT NOT NULL,          -- отсидел (получил idle_payoff)
    PRIMARY KEY (run_id, round_idx, agent_id),
    FOREIGN KEY (run_id, round_idx) REFERENCES rounds(run_id, round_idx),
    FOREIGN KEY (run_id, agent_id)  REFERENCES agents(run_id, agent_id)
);

CREATE TABLE pairings (
    run_id      TEXT NOT NULL,
    round_idx   INTEGER NOT NULL,
    pair_idx    INTEGER NOT NULL,      -- индекс пары в раунде (enumerate(recs))
    a_id        TEXT NOT NULL,         -- a открывает cheap-talk
    b_id        TEXT NOT NULL,
    a_number    INTEGER NOT NULL,
    b_number    INTEGER NOT NULL,
    a_rationale TEXT,                  -- приватные обоснования
    b_rationale TEXT,
    a_outcome   TEXT NOT NULL,         -- CC/DC/CD/DD со стороны a (для b — зеркало)
    a_payoff    REAL NOT NULL,
    b_payoff    REAL NOT NULL,
    usage_prompt_tokens     INTEGER NOT NULL,
    usage_completion_tokens INTEGER NOT NULL,
    usage_calls             INTEGER NOT NULL,
    PRIMARY KEY (run_id, round_idx, pair_idx),
    FOREIGN KEY (run_id, round_idx) REFERENCES rounds(run_id, round_idx),
    FOREIGN KEY (run_id, a_id) REFERENCES agents(run_id, agent_id),
    FOREIGN KEY (run_id, b_id) REFERENCES agents(run_id, agent_id)
);

CREATE TABLE messages (
    run_id     TEXT NOT NULL,
    round_idx  INTEGER NOT NULL,
    pair_idx   INTEGER NOT NULL,
    turn_idx   INTEGER NOT NULL,       -- порядок реплики 0..m-1
    speaker    TEXT NOT NULL,
    text       TEXT NOT NULL,
    ready      INTEGER NOT NULL,       -- 0/1
    PRIMARY KEY (run_id, round_idx, pair_idx, turn_idx),
    FOREIGN KEY (run_id, round_idx, pair_idx) REFERENCES pairings(run_id, round_idx, pair_idx),
    FOREIGN KEY (run_id, speaker) REFERENCES agents(run_id, agent_id)
);
```

**Шаги заполнения:** Шаг 0 (старт) → `runs`+`agents`; Шаг R (каждый раунд, одна txn) →
`rounds`+`idle`+`pairings`+`messages`; Шаг F (финал) → `runs.finished_at`+
`agents.final_score`.

## 5. Интерфейс и поток

```python
# src/storage/store.py
class Storage:
    def __init__(self, db_path: str): ...            # connect; WAL; FK pragma; init_schema
    def begin(self, cfg: EpisodeCfg, pop: Population) -> str: ...
    #   run_id=sha256(canon(cfg)); INSERT runs (config JSON, seed, created_at);
    #   INSERT agents (persona, provider JSON) для каждого; запомнить run_id; вернуть его
    def observe(self, round: int, plan: RoundPlan, recs: list[PairingRecord]) -> None: ...
    #   ОДНА транзакция: rounds + idle + pairings + messages (это и есть observer)
    def finish(self, pop: Population) -> None: ...    # UPDATE runs.finished_at; agents.final_score
    def close(self) -> None: ...
```

Сборка у вызывающего (как в примере/тесте) — Логгер просто подписывается на готовый шов:
```python
pop = make_population(cfg.population, context_window=cfg.context_window).build(Random(cfg.seed))
st = Storage(db_path); st.begin(cfg, pop)
try:
    await run_episode(cfg, pop, observer=st.observe)   # движок не меняется
    st.finish(pop)
finally:
    st.close(); await pop.aclose()
```
`observe` синхронна (sqlite3 синхронный); `run_episode` это поддерживает.

## 6. Срезы (порядок реализации)

**Срез 1 — схема + соединение + `run_id` + `begin`.**
- `schema.py` (DDL §4, `init_schema`); `Storage.__init__` (connect/WAL/FK), `begin`
  (run_id, `runs`+`agents`).
- Тест: создать БД, `begin(cfg, pop)` → строки `runs`/`agents` читаются обратно; тот же
  `cfg` → тот же `run_id`; разный конфиг → разный.
- **DoD:** схема создаётся; старт пишется; `run_id` детерминирован.

**Срез 2 — `observe` (транзакция раунда) + `finish`.**
- `observe`: `rounds`+`idle`+`pairings`+`messages` одной транзакцией; `finish`:
  `finished_at`+`final_score`.
- Тест `test_storage.py`: крошечный эпизод (N=3, rounds=2) на `ScriptedProvider` со
  `Storage.observe` → во всех таблицах верные строки (число пар/реплик/idle), FK-целостность,
  `final_score` сошёлся с пересчётом из `pairings`+`idle`.
- **DoD:** полный эпизод пишется корректно; `pytest tests/storage` зелёный.

**Срез 3 — интеграция (пример/end-to-end).**
- Прогнать оркестратор со `Storage` на `ScriptedProvider` (и опц. live-smoke на Ollama),
  проверить, что счётчики строк совпадают с `recs`/транскриптами.
- **DoD:** Логгер реально пишет эпизод; движок не тронут.

**Срез R — resume (отдельный шаг, отложен).**
- Правка матчера (b): `plan_round` — чистая функция от `(seed, round)` (per-round `rng`).
- Вынос `apply_outcome` в Игре (D2): единый «применить исход к обоим», зовётся вживую и
  при replay.
- `Storage.resume_point()` = `MAX(round_idx)+1`; `replay(pop, …)` — восстановить
  память/очки из `pairings`+`idle`; `run_episode` получает `start_round`.
- Тест: K раундов → «перезапуск» (новый `pop`, replay) → продолжение **эквивалентно**
  непрерывному прогону.
- **DoD:** resume эквивалентен непрерывному прогону. (Делаем после L1.)

## 7. Чистые швы под пост-MVP

- **resume** — см. срез R (матчер (b) + `apply_outcome` + `resume_point`/`replay`).
- **L2-логирование** — `LoggingProvider(inner, sink)` на чокпойнте провайдеров в
  `Population` + таблица `calls(run_id, round_idx, pair_idx, agent_id, phase, attempt,
  raw_text, prompt_tokens, completion_tokens)`; теги через `contextvar` (D6).
- **`events`** — таблица под интерактивные матчинги (`plan.events`), когда появятся.
- **Анализ/метрики** — `analysis/` читает БД (WAL): доли исходов, gossip по `messages`.
- **Свип** — одна БД, много `run_id`; обёртка над `run_episode`+`Storage`.
- **Другой бэкенд** — за тем же `Storage`-API можно спрятать Postgres (пост-MVP).

## 8. Решения (закрыты 2026-06-05)

- **A** нормализуем операционное; конфиги — JSON; персона — колонка.
- **B** `final_score` храним.
- **C** `run_id = sha256(каноничный конфиг)[:16]`.
- **D** без `status`; `created_at`/`finished_at` есть.
- **E** анализ — python/pandas, но схема нормализованная (gossip по `messages`).
- **матчер (b)** — per-round `rng`, делаем в срезе R (resume).
