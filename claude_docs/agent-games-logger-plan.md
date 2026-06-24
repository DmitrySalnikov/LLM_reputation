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
    name        TEXT,                  -- опциональная человекочитаемая метка (не входит в run_id)
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
    a_predicted INTEGER,               -- стратегия prediction: догадка a о числе b (NULL для direct)
    b_predicted INTEGER,
    a_reflection TEXT,                 -- пост-игровая рефлексия (NULL при game.reflection=false)
    b_reflection TEXT,
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
`rounds`+`idle`+`pairings`+`messages`(+`llm_calls`); Шаг F (финал) → `runs.finished_at`+
`agents.final_score`.

> **L2 (§9) ревизует этот §4:** `pairings` получает флаг `finished` (результаты становятся
> NULLABLE + `CHECK`), добавляется 7-я таблица `llm_calls` (сырьё каждого HTTP-вызова).
> Актуальные DDL — в §9.4.

## 5. Интерфейс и поток

```python
# src/storage/store.py
class Storage:
    def __init__(self, db_path: str): ...            # connect; WAL; FK pragma; init_schema
    def begin(self, cfg: EpisodeCfg, pop: Population, name: str | None = None) -> str: ...
    #   run_id=sha256(canon(cfg)); INSERT runs (name, config JSON, seed, created_at);
    #   INSERT agents (persona, provider JSON) для каждого; запомнить run_id; вернуть его
    #   name — опциональная метка, в run_id НЕ входит
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
- Корневые инструменты: `experiment.py` (ТОЛЬКО конфиг + точка входа: правишь `CONFIG`,
  опц. имя рана первым аргументом) и `replay.py <run_id>` (read-only: поднимает всю
  историю из нормализованной схемы; `--config` дампит конфиг + читаемую секцию `prompts:`
  с `rules`/`talk_prompt`/`decide_prompt`). Вся логика прогона (build pop → `run_episode`
  → persist+narrate → score) вынесена в `src/runner.py` (`run`/`run_experiment`/
  `narrate_round`) — `experiment.py` импортирует её, не наоборот; `src/` ничего не знает
  про конкретный конфиг. Дедуп по `run_id`. `examples/logger_demo.py` — sweep-демо «one
  DB, many runs».
- Промпты игры (`rules`/`talk_prompt`/`decide_prompt`) — поля `GameCfg` (дефолты в
  `config.py`: `DEFAULT_RULES`/`DEFAULT_TALK_PROMPT`/`DEFAULT_DECIDE_PROMPT`), значит они
  попадают в config-JSON рана и хранятся в БД. Старые раны до этой правки добиты дефолтными
  промптами прямо в `runs.config` (run_id не трогали).
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

## 9. L2 — сырые вызовы LLM (`llm_calls`) + сорванные пары (`finished`)

Реализация шва из §7 («L2»), но богаче исходного наброска. **Одна строка `llm_calls` =
один `provider.complete()`** (включая парс-ретраи `Agent.act`: каждая попытка — отдельная
строка, т.к. в неё доклеивается `_CORRECTION` → это разный вход). Плюс: ловим и **сбойные**
вызовы (не-JSON / HTTP-ошибка / сеть), которые раньше просто роняли эпизод без следа.

### 9.1 Что считаем сырьём

- **Вход** — литеральный `payload`, уходящий в `self._client.post(...)`
  (`providers/openai_compat.py`). Не реконструируем: провайдер сам сообщает отправленное
  через новое поле `Completion.request = payload`.
- **Выход** — всегда `resp.text` (дословное тело строкой, и при успехе, и при сбое;
  `resp.json()` остаётся как распарсенное удобство в `Completion.raw`, но в лог пишем
  именно `resp.text`).

**Гранулярность — один HTTP-запрос = одна строка.** Пишем **каждый** `self._client.post`,
включая сетевые ретраи (429/5xx/таймаут) и тело 5xx. Провайдер — единственный, кто видит
отдельные попытки, поэтому он накапливает их списком `HttpAttempt` и отдаёт наверх (на
`Completion.attempts` при успехе, на `exc.attempts` при сбое); `Agent.act` проставляет
контекст и разворачивает их в строки `LLMCall`. Два измерения ретраев:
- `attempt` — парс-попытка `act` (1..3): после невалидного JSON `act` доклеивает
  `_CORRECTION` и шлёт **новый** `complete()` с другими `messages`;
- `http_attempt` — сетевой ретрай внутри одного `complete()` (1..5): тот же payload.

### 9.2 Два вида сбоя и как их ловим

1. **Парс-сбой фазы** — `Completion` есть, но текст не прошёл валидатор фазы
   (`{"number":15}`, проза). `act` ретраит с `_CORRECTION`, и после 3 попыток (без
   подстановок!) бросает `ActParseError` → пара срывается (`finished=0`), эпизод
   останавливается. Каждая попытка пишется строкой `llm_calls` со `status='parse_error'`.
2. **Транспортный сбой** — `complete()` **бросает**. Ретраятся как транзиентные (тот же
   payload, до 5 раз) все «200, но тело не извлекается»: сеть, 429/5xx, **битый
   JSON-конверт** (`bad_json`) и **кривая форма** валидного JSON без `choices[0].message`
   (`bad_shape`). Гейт в `_post_with_retries` — это сама попытка извлечь контент.
   Терминальны: 4xx (`ProviderHTTPError`) и исчерпание ретраев (`ProviderUnavailable`).
   Любой такой бросок раньше ронял раунд до `observe` → терялся; теперь ловим на
   **границе пары** (см. §9.3).

### 9.3 Перехват и протяжка (одна точка записи — `observe`)

У провайдера нет игровых ключей (run/round/pair/agent/phase/turn) — он только рапортует
байты. Ключи доклеиваются по пути наверх, тем же маршрутом, что и `usage`:

- Провайдер накапливает попытки списком `HttpAttempt` (`status / status_code / request /
  response / response_raw / error / tokens`); отдаёт на `Completion.attempts` (успех) или
  `exc.attempts` (сбой). `Agent.act` разворачивает каждую попытку в `LLMCall`, проставляя
  `agent_id / phase / attempt(парс) / http_attempt(сетевой)` → `ActResult.calls`. На
  исключении провайдера — аннотирует его `agent_id/phase/attempt`, кладёт `exc.calls`,
  **пробрасывает**.
- `ActResult.calls` → `Decision.calls` (strategy) / `transcript[i]["calls"]` (talk) →
  `PairingRecord.llm_calls` (с доклеенным `turn_idx` для talk).
- **`play_pairing` оборачивает тело в `try/except ProviderError`**: на сбое аннотирует
  `round/a_id/b_id/turn`, собирает `LLMCall` из обогащённого исключения, возвращает
  `PairingRecord(finished=False, …)` с числами/исходом = `None`. Раунд **не падает** →
  `gather` доигрывает остальные пары → весь раунд уходит в `observer`.
- `Storage.observe` пишет раунд **обычным путём**: `pairings` (успешные `finished=1`,
  сорванная `finished=0`) + `messages` + `llm_calls` (с живым FK и `pair_idx` из
  `enumerate(recs)`, `call_idx` — порядок в `rec.llm_calls`). `src/` ничего не персистит.
- `run_episode` после `observer` проверяет `finished=0` → **бросает**, останавливая эпизод
  (`finish()` не зовётся → `runs.finished_at = NULL` как маркер «упал»). Соседние пары
  раунда при этом доиграны и записаны (в отличие от прежнего fail-fast `gather`).

**Гранулярность.** Одна строка = один `self._client.post`. Сетевые ретраи (429/5xx/сеть)
пишутся **каждый** (с телом 5xx и `status_code`), различаются `http_attempt`; парс-ретраи
`act` — различаются `attempt`. Финальная успешная попытка несёт `response`/токены; у
ретраев `response=NULL`, токены 0.

### 9.4 Схема: `pairings` + флаг `finished`, новая таблица `llm_calls`

Сорванная пара получает строку в `pairings` (`finished=0`), поэтому у `llm_calls` есть
родитель → восстанавливаем **FK на `pairings`** и реальный `pair_idx`. Результаты пары
становятся NULLABLE; `CHECK` связывает их с флагом («валидация на пустое»):

```sql
-- pairings: + finished, результаты NULLABLE, CHECK по флагу
ALTER семантика (для свежей БД — в DDL):
    finished INTEGER NOT NULL DEFAULT 1,   -- 1 = доиграна; 0 = сорвана (LLM-сбой)
    a_number, b_number, a_outcome, a_payoff, b_payoff,
    usage_prompt_tokens, usage_completion_tokens, usage_calls  -- снять NOT NULL
    CHECK (finished = 0 OR a_number IS NOT NULL),   -- доиграна ⇒ результат есть
    CHECK (finished = 1 OR a_number IS NULL)        -- сорвана  ⇒ результат пуст

CREATE TABLE llm_calls (
    run_id        TEXT    NOT NULL,
    round_idx     INTEGER NOT NULL,
    pair_idx      INTEGER NOT NULL,
    call_idx      INTEGER NOT NULL,   -- порядок вызова внутри пары (порядок исполнения)
    agent_id      TEXT    NOT NULL,   -- кто вызывал
    phase         TEXT    NOT NULL,   -- talk | decide | predict | reflect
    turn_idx      INTEGER,            -- NULL кроме TALK; FK на конкретную реплику messages
    attempt       INTEGER NOT NULL,   -- парс-попытка Agent.act (1..3)
    http_attempt  INTEGER NOT NULL,   -- сетевой ретрай внутри complete() (1..5)
    status        TEXT    NOT NULL,   -- ok | parse_error | bad_json | bad_shape | http_error | server_error | network
    status_code   INTEGER,            -- HTTP-код попытки (NULL при сетевой ошибке)
    request       TEXT    NOT NULL,   -- ДОСЛОВНЫЙ payload (JSON): model, messages, temperature, max_tokens
    response      TEXT,               -- извлечённый текст (только на финальной ok-попытке); иначе NULL
    response_raw  TEXT,               -- ДОСЛОВНОЕ тело resp.text (вкл. тело 5xx); NULL при сетевой ошибке
    error         TEXT,               -- сообщение сбоя
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (run_id, round_idx, pair_idx, call_idx),
    FOREIGN KEY (run_id, round_idx, pair_idx) REFERENCES pairings(run_id, round_idx, pair_idx),
    FOREIGN KEY (run_id, agent_id) REFERENCES agents(run_id, agent_id),
    FOREIGN KEY (run_id, round_idx, pair_idx, turn_idx)
        REFERENCES messages(run_id, round_idx, pair_idx, turn_idx)
);
CREATE INDEX ix_llm_calls_agent  ON llm_calls(run_id, agent_id);
CREATE INDEX ix_llm_calls_status ON llm_calls(run_id, status);
```

`request` — «что реально ушло»; `response_raw` — «что реально пришло» (включая мусор при
сбое); `response` — извлечённый текст; `status`/`http_code`/`error` — диагностика сбоя.

**Поиск, который это даёт:** все вызовы пары/раунда/рана (джойн к `pairings`); всё, что
говорил агент (индекс по `agent_id`); вызов(ы), породивший публичную реплику (FK
`turn_idx → messages`); вызов DECIDE → его исход (джойн к `pairings`); все сбои
(`WHERE status != 'ok'`); все сорванные игры (`pairings.finished = 0`).

### 9.5 Потребители фильтруют `finished=1`

Раз в `pairings` бывают пустые строки, все потребители результата фильтруют `finished=1`:
- проверка целостности очков; `replay.py` (сорванную помечает/пропускает);
- вход судьи (`records` в `runner` отдаёт судье только `finished=1`);
- `narrate_round` (рендер `rec.a_number is None` без падения).

`CREATE TABLE IF NOT EXISTS` **не мигрирует** старые БД (новых колонок/CHECK там нет) — для
ресёрча с пересоздаваемыми БД ок, фиксируем явно.

### 9.6 Вне scope

Судья (`judge_episode`) тоже бьёт в провайдер, но он вне пары — пока не логируется
(отложено). `call_idx` уникальности достигаем порядком в `rec.llm_calls`.

### 9.7 Срезы (порядок реализации, TDD)

1. **provider** — `Completion.request`; обогащение исключений (`request`/`response_raw`/
   `http_code`/`status`) в `complete`/`_post_with_retries`. Тест: payload в `request`;
   не-JSON → `ProviderParseError` с `.response_raw=resp.text`.
2. **agent** — `LLMCall` + `ActResult.calls`; копить по попытке; на исключении провайдера
   аннотировать и пробросить. Тест: успех → 1 call `status=ok`; парс-сбой+успех → 2 call
   `parse_error,ok`; провайдер бросает → исключение с `agent_id/phase`.
3. **strategy** — `Decision.calls` (direct + prediction).
4. **game** — `PairingRecord.finished` + опц. результаты + `llm_calls`; `play_pairing`
   собирает calls, ловит `ProviderError` → `finished=0`, тэгует `turn_idx`. Тест: успех →
   `finished=1` + calls; сбой decide → `finished=0`, числа `None`, сбойный call в списке.
5. **orchestrator** — после `observer` при `finished=0` бросить. Тест: сорванная пара
   останавливает эпизод, предыдущие раунды записаны.
6. **storage** — `pairings` (finished/nullable/CHECK) + таблица `llm_calls`; `observe`
   пишет `finished` и `llm_calls`. Тест: строки, FK, CHECK, джойн `llm_calls→pairings`.
7. **consumers** — `narrate_round`/судья/`replay`/тесты целостности фильтруют `finished=1`.
8. **доки** — синхронизировать `configuration.md`/`architecture.md`.
