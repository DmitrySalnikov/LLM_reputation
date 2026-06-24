# Анализ и план реализации: слой Игры `reputation_pd` (MVP)

Дата: 2026-05-29. Третий слой разработки (снизу вверх), стоит на готовых слоях
провайдера и Агента. Опирается на `agent-games-mvp-arch.md` §3 (`games/base.py`,
`reputation_pd.py`) и `agent-games-plan.md` §5 (детальный протокол партии).
**При изменениях контракта** (`PairingRecord`/`Game`/`resolve`/`GameCfg`) — синхронно
править arch и plan. Идём **срезами** (§12).

## 0. Зачем Игра третьей

Слой 3 в `-explained` — «правила одной партии двух агентов». Это первый слой, где
агенты реально играют друг против друга: Игра дёргает `agent.act(TALK)` в цикле
переговоров и `agent.act(DECIDE)`, считает исход/платежи, пишет `PairingRecord` и
пополняет память обоих. Тут впервые появляется собственно предмет исследования —
обмен репликами (потенциальный gossip) и его влияние на выбор.

## 1. Анализ: что в этом слое нетривиально

Самое сложное — **не** правило платежей (оно тривиально, §6), а **протокол
переговоров** и **корректное обновление состояния двух агентов**. Разберём узлы.

1. **Стоп-правило cheap-talk (`both_ready_latch`).** Агенты говорят по очереди;
   каждая реплика несёт флаг `ready`. «Сказал готов — вышел из разговора (latch),
   дозревает второй». Тонкости: (а) после latch агент молчит и НЕ реагирует на
   последующие реплики партнёра — это известное ограничение latch (альтернативы —
   шов §13); (б) **каждый говорит ≥1 раз** даже если первый сразу готов — это
   эмерджентный инвариант latch (выход требует обоих `ready`, а `ready` ставится
   только после реплики), отдельный параметр не нужен; (в) **кто начинает — задаёт
   матчер** порядком в паре (`play_pairing(a, b)` → `a` открывает; Игра монетку не
   кидает, `rng` ей не нужен); (г) жёсткий потолок `max_talk_turns` обрывает в
   любом случае. Нужен аккуратный цикл без зацикливания (§8).

2. **Идентичность: свой ID — в `system`, ID партнёра — в `context`.** Свой ID
   агент знает сам (`self.id`) и кладёт в `system` («You are AI agent {id}» + персона).
   ID партнёра и номер раунда — per-pairing, их даёт Игра в `phase.context`. Правила
   игры — статичные, идут в `phase.rules`. (Дневник памяти при этом метит свои реплики
   как `me`, чужие — `partner_id`.)

3. **Относительность исхода для памяти.** `resolve(x, y)` направленна: `DC` значит
   «x обманул y». В память агента A пишем исход с его точки зрения, агенту B —
   **зеркальный** (`DC↔CD`). Иначе B прочитает «DC» как «это я обманул». См. §10.

4. **Приватность.** Чужое `rationale` partner'у не раскрываем никогда. `MemoryEntry`
   хранит только `my_rationale`. Публичный текст (реплики) и приватные обоснования
   в `PairingRecord` лежат **раздельно** — это и нужно для будущего анализа репутации
   (gossip — в публичном тексте, ground-truth действий — в числах).

5. **Нейтральность промптов.** Правила и инструкции держим нейтральными: НЕ «предупреди
   других о кидалах», НЕ «сообщи, кому не доверять» — иначе мы наполовину построим
   институт репутации за агентов и сместим результат (plan §6/§16). Промпт — только про
   механику игры и формат вывода. Это исследовательски критично.

6. **Суммирование usage.** Партия = много вызовов LLM (реплики + 2 решения). Токены
   агрегируем со всех `ActResult.usage` в `PairingRecord.usage`.

7. **Сбой агента в середине партии.** `agent.act` сам гасит провалы парсинга
   (дефолт, не бросает). Но `provider` после исчерпания ретраев бросает
   `ProviderUnavailable`. В MVP — пробрасываем из `play_pairing` (оркестратор решит);
   ловить и «доигрывать» партию — пост-MVP (решение №4).

8. **Параллелизм.** Внутри партии всё последовательно (реплики чередуются; решения
   независимы — можно `gather`, но в MVP делаем последовательно для простоты).
   Независимые ПАРЫ параллелит оркестратор, не Игра.

9. **Свойство игры (не баг, а фича).** Чтобы получить T, надо угадать число партнёра
   и взять ровно +1 (mod 10). Если оба наивно пытаются обмануть — берут `k+1` → снова
   равенство → (R,R): взаимное предательство схлопывается в кооперацию (beauty-contest
   mod 10). На код не влияет, но объясняет динамику; фиксируем в `resolve`-тестах
   (off-by-one не бывает взаимным → исход однозначен).

## 2. Границы слоя (scope)

**В MVP:**
- `GameCfg` (+ `Payoffs`) в `core/config.py`; `Game` (Protocol), `PairingRecord` в `games/base.py`.
- `ReputationPD`: `resolve(x,y)` + `play_pairing(a,b,round)` (cheap-talk `both_ready_latch`
  + DECIDE + платежи + сборка `PairingRecord` + обновление score/памяти обоих).
- Нейтральные промпт-билдеры (rules / talk-context / decide-context).
- Тесты: исчерпывающие на `resolve`; `ScriptedProvider` на цикл/решение/память; Ollama-smoke на партию.

**НЕ в MVP (швы, §13):**
- Фаза `REFLECT` (`reflect_after`) и приватная пост-игровая рефлексия в `PairingRecord`.
- Другие `talk_stop_rule` (`both_ready_revocable` / `either` / `fixed_k`).
- Любой матчинг (кто с кем) — это слой 4; Игра получает готовую пару.
- Анализ репутации/gossip (отдельный модуль, отложен) — Игра лишь пишет данные в удобном виде.

## 3. Раскладка файлов

```
src/
  core/config.py        # + Payoffs, GameCfg
  games/
    __init__.py         # реэкспорт: Game, PairingRecord, ReputationPD
    base.py             # Game (Protocol), PairingRecord
    reputation_pd.py    # ReputationPD: resolve + play_pairing + промпт-билдеры
tests/games/
    test_resolve.py         # исчерпывающе по правилу 0–9 + инварианты платежей
    test_play_pairing.py    # ScriptedProvider: цикл talk, решение, запись, память
    test_smoke_game_ollama.py  # реальная партия двух агентов; skip без сервера
```

## 4. Зависимости

- Только наш код: `src.core.agent` (Agent, Phase, PhaseKind, ActResult),
  `src.core.memory` (MemoryEntry), `src.core.config`. Новых пакетов нет.

## 5. Типы

```python
# core/config.py
@dataclass(frozen=True)
class Payoffs:
    R: float = 3.0; T: float = 5.0; P: float = 1.0; S: float = 0.0   # T>R>P>S, 2R>T+S

@dataclass(frozen=True)
class GameCfg:
    payoffs: Payoffs = field(default_factory=Payoffs)
    max_talk_turns: int = 6            # жёсткий потолок ВСЕХ реплик в партии (сумма по обоим)
    talk_stop_rule: str = "both_ready_latch"   # MVP — единственное значение

# games/base.py
@dataclass
class PairingRecord:
    round: int; a_id: str; b_id: str
    transcript: list[dict]             # [{speaker, text, ready}] — публичный текст
    a_number: int; b_number: int
    a_rationale: str; b_rationale: str # приватные; partner'у НЕ раскрываются (только в артефакт)
    outcome: str                       # с точки зрения A: CC/DC/CD/DD
    a_payoff: float; b_payoff: float
    usage: dict                        # {"prompt_tokens","completion_tokens","calls"}

class Game(Protocol):
    async def play_pairing(self, a: Agent, b: Agent, round: int) -> PairingRecord: ...
    #   no rng: первого говорящего фиксирует матчер порядком в паре (a открывает) — решение №6
```

`max_talk_turns` = суммарный потолок реплик (дефолт 6 → ~3 на каждого). Альтернатива —
«на агента» (решение №2).

## 6. `resolve(x, y)` — правило и инварианты

```python
def resolve(self, x: int, y: int) -> tuple[str, float, float]:
    R,T,P,S = payoffs
    if x == y:            return ("CC", R, R)
    if x == (y + 1) % 10: return ("DC", T, S)   # x обманул y (на +1 по кругу, 0 бьёт 9)
    if y == (x + 1) % 10: return ("CD", S, T)   # y обманул x
    return ("DD", P, P)
```
- Направленный цикл `1>0, 2>1, …, 0>9`: каждое число эксплуатирует ровно одно (на 1 ниже).
- Off-by-one **не бывает взаимным** → максимум один получает T, исход однозначен.
- Инварианты платежей: `T>R>P>S` и `2R>T+S` (строгое PD) — проверим тестом.

## 7. `play_pairing` — конвейер

```python
async def play_pairing(self, a, b, round) -> PairingRecord:
    transcript = await self._cheap_talk(a, b, round)              # §8
    feed = render_feed(transcript)
    ra = await a.act(Phase(DECIDE, decide_context(a.id, b.id, round, feed), rules))
    rb = await b.act(Phase(DECIDE, decide_context(b.id, a.id, round, feed), rules))
    x, y = ra.data["number"], rb.data["number"]
    outcome, pa, pb = self.resolve(x, y)
    a.score += pa; b.score += pb
    self._remember(a, b, round, transcript, ra, rb, outcome, pa)  # §10
    self._remember(b, a, round, transcript, rb, ra, _flip(outcome), pb)
    return PairingRecord(round, a.id, b.id, transcript, x, y,
                         ra.data["rationale"], rb.data["rationale"],
                         outcome, pa, pb, usage=_sum_usage(transcript, ra, rb))
```
- Решения **независимы и вслепую**: `decide_context` содержит только ленту переговоров,
  не выбор партнёра. Последовательно (можно `gather` — не критично).
- `usage` агрегирует токены всех `act` (реплики хранят usage в transcript-метаданных или
  суммируем по ходу — детали в коде).

## 8. Cheap-talk: `both_ready_latch` (детально)

```python
async def _cheap_talk(self, a, b, round):
    transcript = []
    ready = {a.id: False, b.id: False}
    order = [a, b]                                      # a открывает; ориентацию задал матчер
    i = 0
    while len(transcript) < self.cfg.max_talk_turns:
        cur = order[i % 2]; oth = order[(i + 1) % 2]; i += 1
        if ready[cur.id]:
            if ready[oth.id]:
                break                                  # оба готовы — конец
            continue                                   # latched — молчит, дозревает второй
        ctx = talk_context(cur.id, oth.id, round, render_feed(transcript))
        res = await cur.act(Phase(TALK, ctx, self._rules))
        transcript.append({"speaker": cur.id, "text": res.public_text, "ready": res.data["ready"]})
        ready[cur.id] = res.data["ready"]
        if ready[a.id] and ready[b.id]:
            break
    return transcript
```
- **Без зацикливания:** пропуск latched-агента (`continue`) не добавляет реплику, но `i`
  растёт → следующий ход у другого; если оба latched → выходим (`break`). Между
  двумя `continue` подряд всегда вклинивается говорящий.
- **Каждый говорит ≥1 раз — это эмерджентный инвариант latch, не параметр:** выход
  требует, чтобы ОБА были `ready`, а `ready` ставится только после реплики. Поэтому
  отдельный `min_talk_turns_each` не нужен (убран).
- **Потолок** `max_talk_turns` режет по числу реплик в `transcript`.

## 9. Промпты (нейтральные)

Билдеры в `reputation_pd.py`. Язык — английский (портируемо на не-русские модели в свипе).
- **`rules`** (статично → `system` после персоны): механика числа 0–9, матрица платежей,
  «слова ни к чему не обязывают», «выбор тайный и одновременный». Без призывов про доверие/слухи.
- **`talk_context`**: «Your partner this round is {partner}. Round {r}. Negotiation
  so far:\n{feed}\nSend a short message; set ready=true when you have nothing to add. JSON
  {message, ready}.» (свой ID агент уже знает из `system`.)
- **`decide_context`**: «Your partner this round is {partner}. Round {r}. Negotiation:\n{feed}\n
  Now SECRETLY choose your number 0–9. JSON {number, rationale}.»
- `feed` = лента: на реплику строка `{speaker}: {text} (ready=...)`.
- Память агента (дневник прошлых партий) подмешивает сам `Agent.act` — Игра её не трогает.

## 10. Память после партии (`_remember`)

Для каждого агента — `MemoryEntry` с **его** точки зрения:
```python
MemoryEntry(round, partner_id=other.id, transcript=transcript,
            my_number=mine.number, my_rationale=mine.rationale,
            partner_number=other.number, outcome=relative_outcome, payoff=my_payoff)
```
- `transcript` общий (публичный) — одинаков у обоих.
- `relative_outcome`: для A — `outcome` (resolve звался с x=a); для B — `_flip(outcome)`
  (`DC↔CD`, `CC`/`DD` без изменений).
- **Чужое `rationale` НЕ попадает** в память агента (приватность).

## 11. Тест-план

**`test_resolve.py` (без LLM, исчерпывающе):**
- Все 100 пар `(x,y)`: ровно при `x==y` → CC; при `x==(y+1)%10` → DC; зеркально CD; иначе DD.
- Обёртка `0` бьёт `9` (`0 == (9+1)%10`).
- Платежи: `(R,R)`/`(T,S)`/`(S,T)`/`(P,P)`; инварианты `T>R>P>S`, `2R>T+S`.
- Off-by-one не взаимен (нет пары с двумя T).

**`test_play_pairing.py` (`ScriptedProvider`, 2 агента):**
- Скриптуем talk до latch обоих → проверяем длину/порядок `transcript`, флаги `ready`.
- Потолок `max_talk_turns` обрывает; каждый говорит ≥1 раз (инвариант latch); первый — порядок аргументов (`play_pairing(a, b)` → `a`; swap → `b`).
- Решения → `resolve` → корректные `outcome`/платежи; `score` обоих обновлён.
- Память: у A и B по записи; `outcome` зеркальный; `my_rationale` не протёк к партнёру.
- `PairingRecord` заполнен; `usage` агрегирован.
- latched-агент после готовности молчит (не появляется новых его реплик).

**`test_smoke_game_ollama.py` (skip без сервера):** реальная партия 2 агентов на
`llama3:8b` → `PairingRecord` с числами 0–9, валидным `outcome`, платежами; у обоих
agents пополнилась память.

Запуск: `pytest tests/games -q`.

## 12. Срезы (порядок реализации)

**Срез 1 — `resolve` + типы (чистое ядро, без LLM).**
- `Payoffs`, `GameCfg` (config); `PairingRecord`, `Game` (base); `ReputationPD.__init__` + `resolve`.
- Тесты `test_resolve.py` (исчерпывающе). **DoD:** правило и инварианты зелёные.

**Срез 2 — решение + платёж + запись + память (talk выключен).**
- `play_pairing` для DECIDE-части при `max_talk_turns=0`: 2×`act(DECIDE)` → `resolve` →
  `score` → `PairingRecord` → `_remember` обоих (зеркальный outcome, приватность).
- Промпт-билдеры `rules`/`decide_context`. Тесты на `ScriptedProvider`.
- **DoD:** партия без переговоров считается и пишется верно; память корректна.

**Срез 3 — cheap-talk `both_ready_latch`.**
- `_cheap_talk` (§8) + `talk_context` + `render_feed`; подключить перед DECIDE.
- Тесты: latch, потолок, min-one-each, порядок, первый — по порядку аргументов.
- **DoD:** переговоры идут и корректно останавливаются; `transcript` полон.

**Срез 4 — нейтральные промпты + Ollama-smoke end-to-end.**
- Финализировать формулировки rules/контекстов (ревью на нейтральность, §1.5).
- `test_smoke_game_ollama.py`: реальная партия двух агентов.
- **DoD:** `pytest tests/games` зелёный; живая партия даёт осмысленный `PairingRecord`.

После среза 4 слой Игры закрыт → дальше слой Матчинга (кто с кем) и оркестратор.

## 13. Чистые швы под пост-MVP

- **`REFLECT`**: после вскрытия — приватная рефлексия каждого (`reflect_after`); добавляется
  в `PairingRecord` и в память. Новая ветка в `act` (слой Агента уже расширяем по `PhaseKind`).
- **Другие `talk_stop_rule`**: `both_ready_revocable` / `either` / `fixed_k` — ветки в `_cheap_talk`.
- **`max_talk_turns` на агента** вместо суммарного — параметр.
- **Анализ репутации**: `PairingRecord` уже разделяет публичный текст и приватные
  обоснования + хранит числа (ground truth) → модуль анализа подключается без правок Игры.

## 14. Решения (все закрыты на 2026-06-04)

1. **Язык промптов:** РЕШЕНО — **английский** (механика; персоны/реплики — как заданы).
2. **`max_talk_turns`:** РЕШЕНО — **суммарный** потолок реплик (реализовано).
3. **Решения внутри партии:** РЕШЕНО — **последовательно** (реализовано). Кросс-парный
   `gather` — на слое Оркестратора; воспроизводимость *внутри* партии не преследуем
   (LLM недетерминирован при temperature>0), поэтому игре `rng` не нужен (см. 6).
4. **Сбой агента (provider исчерпал ретраи):** РЕШЕНО — **fail-fast**: пробрасываем из
   `play_pairing`, оркестратор валит эпизод, восстановление через `resume`.
5. **`MemoryEntry.outcome`:** РЕШЕНО — **относительный** (зеркалим для B; реализовано).
6. **Кто первый говорит:** РЕШЕНО — **порядок в паре от матчера**: `play_pairing(a, b)`,
   `a` открывает. `rng` из контракта `Game` убран (матчер — единственный источник
   структурной случайности, его `rng` воспроизводим). Реализовано.
