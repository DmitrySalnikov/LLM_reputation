# План реализации: слой Агента + Память (MVP)

Дата: 2026-05-29. Второй слой разработки (снизу вверх), стоит на готовом слое
провайдера (`agent-games-provider-plan.md`). Опирается на `agent-games-mvp-arch.md`
§3 (`core/agent.py`, `core/memory.py`) и `agent-games-plan.md` §4–§6.
**При изменениях контракта** (`Phase`/`ActResult`/`AgentSetup`/`Memory`) — синхронно
править arch и plan. Идём **срезами** (§10): каждый срез самодостаточен и тестируем.

## 0. Зачем Агент вторым

Слой 2 в `-explained` — «актёр со своим дневником». Это первый настоящий
потребитель провайдера: `Agent.act` собирает промпт и зовёт `provider.complete`.
Пока его нет — ни Игре, ни генератору популяции не на чём стоять (оба зависят от
`Agent`/`AgentSetup`). Заодно «в бою» проверим провайдер и нашу находку про
reasoning-модели (§8).

## 1. Границы слоя (scope)

**Входит в MVP:**
- `AgentSetup` (генотип: persona + `ProviderCfg`), `Agent` (id, setup, memory, score).
- `PhaseKind` (`TALK`, `DECIDE`), `Phase`, `ActResult`.
- `Agent.act(phase)`: сборка `system`+`messages`, вызов провайдера, **парсинг JSON по фазе**.
- Устойчивый парсер/валидатор JSON-ответа + ретраи на кривой ответ.
- `Memory` / `MemoryEntry`: дневник + `render(window)`.
- Тест-дубль `ScriptedProvider` (без сети) + Ollama-smoke по каждому срезу.

**Реализовано после MVP:** фазы `REFLECT` и `NOTE` + второй ярус памяти (memory notes:
`Memory.notes`/`noted_upto`/`set_notes`; `render` шлёт заметки + буфер новых раундов
вместо полной истории). `NOTE` сворачивает память каждые `game.memory_notes_every`
**сыгранных агентом** раундов (`len(memory.entries)`, idle не считается); его `act`
рендерит память целиком (без окна). См. `architecture.md` §«Pairing flow».

**НЕ входит (чистые швы, §11):**
- `CHOOSE_PARTNER`/`CONSENT`.
- Многоходовый рендер памяти настоящими `assistant`/`user`-репликами (MVP — текстовый блок).
- Логика игры (правила, платежи, лента переговоров) — это слой Игры; Агент её только потребляет.
- Загрузка `AgentSetup` из YAML — придёт со слоем конфига; в тестах строим вручную.

## 2. Раскладка файлов

```
src/core/
  agent.py     # AgentSetup, PhaseKind, Phase, ActResult, Agent (+ парсинг JSON)
  memory.py    # MemoryEntry, Memory
tests/core/
  test_agent.py     # act(DECIDE/TALK), парсинг/валидация/ретраи, сборка промпта (ScriptedProvider)
  test_memory.py    # add/render/window, формат дневника
  test_smoke_agent_ollama.py   # act против реальной Ollama; skip без сервера
```
> `ScriptedProvider` (тест-дубль, реализует протокол `LLMProvider`) живёт в
> `tests/core/` или `tests/conftest.py` — не в `src/`.

## 3. Зависимости

- Только stdlib (`json`, `re`, `enum`, `dataclasses`) + наш слой провайдера
  (`src.providers`, `src.core.config.ProviderCfg`). Новых пакетов нет.
- `httpx` тащится транзитивно через провайдер (Agent сам HTTP не трогает).

## 4. Типы (`core/agent.py`, `core/memory.py`)

```python
class PhaseKind(Enum):
    TALK = "talk"
    DECIDE = "decide"

@dataclass(frozen=True)
class Phase:
    kind: PhaseKind
    context: str          # ситуация + инструкция вывода (станет user-сообщением)
    rules: str = ""       # СТАТИЧНЫЙ свод правил игры; Агент кладёт его в system после персоны

@dataclass(frozen=True)
class ActResult:
    public_text: str | None     # TALK -> message; DECIDE -> None
    data: dict                  # TALK -> {message, ready}; DECIDE -> {number, rationale}
    usage: tuple[int, int]      # (prompt_tokens, completion_tokens), суммарно по всем попыткам

@dataclass(frozen=True)
class AgentSetup:
    persona: str | None          # None -> в system только преамбула (+ правила)
    provider_cfg: ProviderCfg
    identity_prompt: str         # преамбула system; Агент подставляет {id} (общая на популяцию, см. PopulationCfg)

@dataclass
class MemoryEntry:
    round: int; partner_id: str
    transcript: list[dict]               # [{speaker, text, ready}]
    my_number: int; my_rationale: str; partner_number: int
    outcome: str; payoff: float
```

**Отклонения от arch §3 (мелкие, синхронизировать при кодинге):**
1. `Phase` получает поле **`rules`** (в arch было `{kind, context}`). Причина — §5/решение №1.
2. `Agent.__init__` получает **`context_window`** (в arch — `(id, setup, provider)`); см. §5/решение №2.

## 5. `Agent` и `act(phase)` — конвейер

```python
class Agent:
    def __init__(self, id, setup: AgentSetup, provider: LLMProvider,
                 *, context_window: int | None = None):
        self.id = id; self.setup = setup; self.provider = provider
        self.memory = Memory(); self.score = 0.0
        self._window = context_window      # None = ∞

    async def act(self, phase: Phase) -> ActResult:
        system = self.setup.identity_prompt.replace("{id}", self.id) + ("\n\n" + self.setup.persona if self.setup.persona else "") + ("\n\n" + phase.rules if phase.rules else "")
        diary = self.memory.render(self._window)         # [] или [user-сообщение с дневником]
        # ОДНО user-сообщение: дневник + phase.context (+ поправка на ретрае) — склеиваем
        # цикл парс-ретраев (§6): зовём провайдер, парсим по phase.kind
        # суммируем usage по попыткам; на успехе -> ActResult
```

Шаги одного `act`:
1. **`system`** = преамбула (`setup.identity_prompt`; `{id}` подставляет агент) + персона
   (`setup.persona`, опциональна — при `None` блок персоны опускается) + правила
   (`phase.rules`). Преамбула общая на всю популяцию (поле `identity_prompt` блока
   `population`, дефолт `"You are AI agent {id}."`) — это фиксированная рамка эпизода, как
   и правила, а не атрибут отдельного агента; `RosterGenerator` кладёт её в каждый
   `AgentSetup`. Персону агент знает о себе сам; правилами владеет Игра (кладёт их в
   `Phase`); Агент остаётся игронезависимым — просто склеивает.
2. **`messages`** = **одно** user-сообщение: дневник (`memory.render(window)`, §7) +
   `phase.context` (текущая ситуация + инструкция формата вывода — её пишет Игра в
   `context`), склеенные через `\n\n`. Раньше это были два отдельных user-сообщения;
   теперь одно — чтобы не слать подряд два хода одной роли (см. §7 и переносимость
   между провайдерами). После склейки Агент чистит **стыки `<game>`-блоков**: если
   закрывающий `</game>` отделён от следующего открывающего `<game>` лишь пробелами
   (строка результата прошлого раунда ↔ открытие следующего, в т.ч. на границе
   история↔текущий раунд), теги стыка убираются и блоки сливаются в один — транскрипт
   не дёргается лишними закрытием/открытием (`_merge_game_blocks`, regex `</game>\s*<game>`).
3. **Вызов** `provider.complete(system, messages, temperature, max_tokens)` — параметры
   сэмплинга берём из `setup.provider_cfg`.
4. **Парсинг** ответа по `phase.kind` (§6). Успех → `ActResult`; неуспех → ретрай, а после
   всех ретраев — `ActParseError` (пара срывается, без подстановок).

`window` нужен `render`, но это эксперимент-параметр (один на эпизод, не «генотип»),
поэтому он в конструкторе `Agent`, а не в `AgentSetup`. Кто его прокидывает — оркестратор
при сборке популяции (решение №2).

## 6. Парсинг и валидация JSON ответа

Модель возвращает **текст**; нам нужен словарь по схеме фазы. Логика — в Агенте
(она зависит от `PhaseKind`); инструкцию формата пишет Игра в `phase.context`.

**Выемка JSON (lenient):**
- срезать ```/```json-ограждение, если есть;
- найти первый сбалансированный `{...}`-блок (на случай прозы вокруг) и `json.loads`.

**Валидация по фазе:**
- `DECIDE`: `number` — int в `0..9`; `rationale` — str (пустая допустима).
- `TALK`: `message` — str; `ready` — bool (мягкая коэрция: `true/false`, `1/0`,
  `"yes"/"no"`; **отсутствует → `false`**, т.е. «продолжаем говорить» — безопаснее).

**При провале (не распарсилось / не прошло валидацию):**
- до `max_parse_retries` (дефолт **2**) повторных вызовов; со 2-й попытки к тому же
  user-сообщению **дописываем** короткую корректирующую реплику («Ответь СТРОГО валидным
  JSON вида …»), а не шлём её отдельным сообщением;
- если и после ретраев плохо — **никаких подстановок**: `act` бросает `ActParseError`
  (+ счётчик `parse_failures`), пара срывается (`finished=0`) и эпизод останавливается —
  ровно как при сбое провайдера. (Это отменяет прежнее «решение №3» о безопасном дефолте:
  тихие случайные ходы засоряли результаты, поэтому теперь честно роняем партию.)
- **Пустой `content`** (reasoning-модель не успела, `finish_reason="length"`, §8)
  трактуем как провал парсинга → тот же ретрай-путь.

`usage` в `ActResult` — сумма `(prompt, completion)` по **всем** сделанным попыткам.

## 7. Память (`core/memory.py`)

```python
class Memory:
    def __init__(self): self.entries: list[MemoryEntry] = []
    def add(self, e: MemoryEntry) -> None: self.entries.append(e)
    def render(self, window: int | None) -> list[Message]:
        # последние window записей (None = все) -> ОДИН Message("user", diary_text)
        # пусто -> []
```

- **MVP — текстовый блок.** `render` сводит последние `window` записей в один
  `Message("user", diary_text)`. Роли почти везде `"user"` (см. обсуждение: внутри
  партии лента переговоров тоже идёт текстом в `context`, не реальными `assistant`-
  репликами). Структурный многоходовый рендер — шов на потом (решение №5).
- **Формат записи** (с июня 2026 — игровой транскрипт, не сводка): каждый прошлый раунд
  отрисовывается теми же тегами, что объявлены в правилах — `<game>` (реплики игры),
  `<you>` (свои), `<Имя>` (партнёра). Блок раунда: открытие чата (с пометкой, кто говорил
  первым — видно по первому `speaker`), реплики cheap-talk, закрытие чата (с причиной —
  лимит реплик или обоюдное `finish`), своё секретное число строкой `<you>N</you>`, и
  вскрывающая строка результата (оба числа, **обе выплаты**, накопленный счёт **после**
  раунда). Сырой код исхода (`outcome`) **не** рендерится. Шаблоны строк живут в `GameCfg`
  (`history_*`, `msg_*`, `opener_*`, `reason_*`), не в коде; `render_turns` (в `memory.py`)
  рендерит реплики и для истории, и для живого `{feed}` текущего раунда — прошлое и
  настоящее читаются одинаково (та же строка закрытия с `{reason}`, что и в живой фазе
  DECIDE; тот же `opener_self`, с которого начинается живой `talk_open_prompt`). Пример блока:
  ```
  <game>Round 3 · opponent A5
  The chat has been open. A5 starts first:</game>
  <A5>let us both take 4</A5>
  <you>ok, 4</you>
  <game>The chat has been closed as both players agreed to stop. Choose the number.</game>
  <you>4</you>
  <game>The choice has been accepted. A5 chose 4. Payoffs: you = 4, A5 = 4.
  Your total score after round 3 is 16 points.</game>
  ```
  Приватные следы выключенных фич (rationale/prediction/reflection) добавляются хвостом
  строкой `<you>(...)</you>` — только если присутствуют. Шаблоны транскрипта едут в
  `render` через `Phase.game_cfg` (как `rules` — от игры), дефолт = `GameCfg()`.
- `window` в раундах: `None` = вся история; `k` = последние k записей. Токен-кап —
  пост-MVP предохранитель (§11).

## 8. max_tokens и reasoning-модели (следствие слоя провайдера)

Из находки провайдера (provider-plan §11): qwen3 кладёт `<think>` в `message.reasoning`,
а ответ — в `content`; при малом `max_tokens` `content` пуст. Для Агента это значит:
- держать `max_tokens` в `ProviderCfg` **с запасом** (для qwen3 простой ход ≈ 200+
  токенов уходит на reasoning перед JSON);
- пустой `content` Агент уже ловит как провал парсинга (§6) → ретрай;
- smoke-тесты Агента: либо `max_tokens` ≥ 512, либо не-reasoning модель (`llama3:8b`)
  для скорости.

## 9. Тест-дубль и тест-план

**`ScriptedProvider`** — реализует протокол `LLMProvider`, без сети:
```python
class ScriptedProvider:
    def __init__(self, replies: list[str]):
        self._q = list(replies); self.calls = []     # calls: list[(system, messages)]
    async def complete(self, *, system, messages, temperature, max_tokens) -> Completion:
        self.calls.append((system, messages))
        return Completion(self._q.pop(0), prompt_tokens=1, completion_tokens=1, raw={})
    async def aclose(self): ...
```
Отдаёт заранее заданные тексты по очереди и **запоминает** (system, messages) —
позволяет проверять и парсинг, и сборку промпта.

**Unit (без сети):**
- DECIDE: чистый JSON; JSON в ```-блоке; JSON среди прозы; `number` вне `0..9` → ретрай;
  невалидный дважды → 3 вызова; стойкий провал → дефолт + `parse_failures`.
- TALK: `{message, ready}`; коэрция `ready` (true/"yes"/1); `ready` отсутствует → false.
- Сборка промпта: `system` содержит персону и правила; единственное сообщение —
  `user`, и при непустой памяти его текст = дневник + `\n\n` + `context` (склейка).
- `usage` суммируется по ретраям.

**Smoke (Ollama, skip без сервера):** реальный `act(DECIDE)` → `number` в `0..9`,
`rationale` непустой; реальный `act(TALK)` → есть `message` и `ready: bool`.

Запуск: `pytest tests/core -q`.

## 10. Срезы (порядок реализации)

**Срез 1 — типы + `act(DECIDE)` + парсинг (без памяти).**
- `PhaseKind`, `Phase`, `ActResult`, `AgentSetup`; минимальный `Memory` (пустой,
  `render → []`); `Agent.__init__` + `act` только для `DECIDE`.
- Парсер/валидатор JSON + ретраи; после ретраев — `ActParseError` (§6).
- Тесты: DECIDE-кейсы + сборка промпта (ScriptedProvider) + Ollama-smoke на число 0–9.
- **DoD:** `pytest tests/core/test_agent.py` зелёный; smoke даёт число 0–9.

**Срез 2 — `act(TALK)`.**
- Ветка `TALK` в парсинге: `{message, ready}`, коэрция `ready`, `public_text=message`.
- Тесты: TALK-кейсы + smoke (message + ready).
- **DoD:** оба `kind` работают; тесты зелёные.

**Срез 3 — Память.**
- `MemoryEntry` (полный), `Memory.add`, `render(window)` (диалоговый блок, окно).
- Подключить `memory.render(window)` в `act`.
- Тесты: render с N записями, `window=k` vs `None`, формат блока; интеграция —
  `act` с непустой памятью прокидывает дневник в промпт (через `ScriptedProvider.calls`).
- **DoD:** `pytest tests/core` зелёный; агент «помнит» прошлые партии в промпте.

После среза 3 слой Агента закрыт → можно браться за Игру (`reputation_pd`), которая
дёргает `agent.act(TALK/DECIDE)`.

## 11. Чистые швы под пост-MVP

- **Фазы `REFLECT` / `NOTES`**: новые значения `PhaseKind` + ветки парсинга в `act`
  (REFLECT — свободный текст; NOTES — обновление заметок). Промпты держим нейтральными
  (plan §6: не «запиши, кто тебя кинул»).
- **Второй ярус памяти (notes)**: `Memory.render` отдаёт `[notes] + [последние W сырьём]`.
- **Структурный рендер памяти**: `assistant`/`user`-реплики вместо текстового блока.
- **Матчинг-фазы** `CHOOSE_PARTNER` / `CONSENT`: те же `PhaseKind` + схемы вывода.
- **Токен-кап памяти** как предохранитель сверх `window`.
- **Эволюция**: `AgentSetup` уже отделён → копируется в потомка (provider-plan/arch §8).

## 12. Открытые решения (подтвердить до/в ходе кодинга)

1. **Откуда правила игры:** поле `Phase.rules` (Игра пишет, Агент кладёт в `system`).
   Альтернатива — целиком в `phase.context`. Предлагаю `Phase.rules` (правила в `system`,
   как в arch; Агент остаётся игронезависимым). **Расширяет `Phase` в arch.**
2. **Куда прокинуть `context_window`:** параметр конструктора `Agent` (не в `AgentSetup`,
   т.к. это эксперимент-ось, а не генотип). Прокидывает оркестратор. **Расширяет ctor в arch.**
3. **Политика провала парсинга:** ретрай ×2 с корректирующей репликой → безопасный
   дефолт + счётчик `parse_failures` (vs бросать исключение и ронять партию). Предлагаю
   дефолт+метрику — устойчивее для длинных прогонов; провалы видны в метриках.
4. **Коэрция `ready`:** принимать `bool`/`true,false`/`1,0`/`yes,no`; отсутствие → `false`.
5. **Рендер памяти:** один текстовый `user`-блок в MVP (vs структурные реплики). Предлагаю блок.
6. **Владение форматом вывода:** инструкцию «верни JSON вида …» пишет Игра в
   `phase.context`; Агент только парсит (и добавляет корректирующий суффикс на ретрае).
   Альтернатива — Агент сам всегда добавляет формат-инструкцию по `kind`.
