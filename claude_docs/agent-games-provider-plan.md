# План реализации: слой провайдера LLM (MVP)

Дата: 2026-05-29. Детальный план **первого** слоя разработки (снизу вверх).
Опирается на `agent-games-mvp-arch.md` §3 (`providers/base.py`, `providers/openai_compat.py`)
и `agent-games-plan.md` §7. Этот файл детализирует те интерфейсы до уровня реализации.
**При изменениях контракта** (`Message`/`Completion`/`LLMProvider`) — синхронно править arch и plan.

## 0. Почему провайдер первым

Слой 1 в `-explained` — «телефон до модели». Это нижний слой: на нём стоит Агент
(Слой 2), который дёргает `provider.complete(...)`. Реализуем и тестируем его
изолированно (без Агента/Игры), end-to-end против локальной Ollama — тогда у
верхних слоёв есть твёрдая опора, а отладка сети/ретраев/токенов не размазана по
агенту.

## 1. Границы слоя (scope)

**Входит в MVP:**
- Протокол `LLMProvider.complete(...)` (async) + типы `Message`, `Completion`.
- `OpenAICompatibleProvider`: HTTP-вызов `chat/completions`, ретраи с backoff,
  таймауты, извлечение usage, разбор ответа.
- `ProviderCfg` (датакласс) + разрешение API-ключа из env.
- Фабрика `make_provider(cfg) -> LLMProvider`.
- Иерархия ошибок провайдера.
- Тесты: unit на `httpx.MockTransport` + smoke против Ollama.

**НЕ входит (оставляем чистые швы):**
- Нативный `AnthropicProvider` + prompt caching (пост-MVP).
- Streaming, tool-calls, `response_format`/JSON-режим.
- **Кэширование инстансов** провайдера по `(base_url, model)` — живёт в `Population`
  (`population/base.py`), не здесь. Здесь только «построить один провайдер».
- **Парсинг JSON** ответа модели (`{message, ready}` / `{number, rationale}`) — это
  слой Агента. Провайдер возвращает сырой текст.
- Учёт стоимости/`estimate` — слой свипа; провайдер лишь отдаёт `prompt/completion` токены.

## 2. Раскладка файлов

```
LLM_reputation/                # этот репозиторий (ветка game)
  pyproject.toml               # deps: httpx ; dev: pytest, pytest-asyncio ; + конфиг pytest (asyncio_mode=auto)
  config/example.yaml
  src/                         # движок — сам по себе пакет; импорт `from src.…`
    __init__.py
    core/
      __init__.py
      config.py                # (частично) ProviderCfg + загрузка блока provider_*
    providers/
      __init__.py              # реэкспорт публичного: Message, Completion, LLMProvider, make_provider
      base.py                  # Message, Completion, LLMProvider (Protocol), ошибки
      openai_compat.py         # OpenAICompatibleProvider + make_provider
  tests/
    providers/
      test_openai_compat.py    # unit: запрос/ответ/usage/ретраи/ошибки (MockTransport)
      test_smoke_ollama.py     # smoke: реальный вызов; skip, если сервер недоступен
  analysis/                    # аналитика — отдельно от движка (в слое провайдера не трогаем)
```

> `src/core/config.py` общий с будущими слоями — сейчас заводим в нём **только**
> `ProviderCfg` и разбор provider-блока YAML; остальные датаклассы конфига придут
> со своими слоями. **Вариант A:** `src/` — это сам пакет; в каждой папке —
> `__init__.py`, импорт вида `from src.providers.openai_compat import
> OpenAICompatibleProvider`, запуск и тесты — из корня репо. Аналитика
> (`analysis/`) импортирует движок (`from src.…`), не наоборот.

## 3. Зависимости и среда

- **Python 3.11+** (Protocol, `X | None`, dataclasses, `asyncio`).
- **httpx** — единственная рантайм-зависимость слоя (async-клиент).
- **dev:** `pytest`, `pytest-asyncio` (режим `asyncio_mode=auto`).
- Ретраи/backoff пишем **руками** (без `tenacity`) — минимизируем зависимости.
- Unit-тесты — на встроенном `httpx.MockTransport` (без `respx`).

## 4. Типы данных (`providers/base.py`)

```python
@dataclass(frozen=True)
class Message:
    role: str          # "user" | "assistant"  (system передаётся отдельно — см. §5)
    content: str

@dataclass(frozen=True)
class Completion:
    text: str          # choices[0].message.content (или "" при пустом)
    prompt_tokens: int    # usage.prompt_tokens (0, если провайдер не отдал usage)
    completion_tokens: int
    raw: dict          # полный JSON ответа — для логов/отладки/анализа
```

Поля `Completion` — ровно как в arch.md §3 (`text; prompt_tokens; completion_tokens; raw`).
Кандидат на добавление (см. §13/§15): `finish_reason`, `model` — потребует правки arch.

## 5. Протокол `LLMProvider`

```python
class LLMProvider(Protocol):
    async def complete(self, *, system: str, messages: list[Message],
                       temperature: float, max_tokens: int) -> Completion: ...
    async def aclose(self) -> None: ...     # закрыть нижележащий HTTP-клиент
```

Семантика:
- **`system` — отдельный параметр**, не элемент `messages`. Реализация сама ставит
  его первым сообщением с ролью `system`. `messages` — это чередование `user`/`assistant`.
- `temperature`/`max_tokens` приходят **в вызов** (их источник — `ProviderCfg` агента,
  §6), провайдер их не хранит.
- Успех → `Completion`. Неуспех после ретраев → исключение из иерархии §8.
- Без побочных эффектов, кроме HTTP-запроса. Идемпотентность не гарантируется (LLM).

## 6. `ProviderCfg` и ключи (`src/core/config.py`)

```python
@dataclass(frozen=True)
class ProviderCfg:
    base_url: str                 # напр. http://localhost:11434/v1
    model: str                    # напр. qwen3:8b
    api_key_env: str = ""         # имя env-переменной с ключом; пусто → локалка
    temperature: float = 0.7      # дефолт сэмплинга (читает Агент, передаёт в complete)
    max_tokens: int = 512
    timeout_s: float = 120.0      # read-таймаут на вызов (LLM медленный)
```

- `temperature`/`max_tokens` тут — **дефолты для Агента**, не состояние провайдера.
  `AgentSetup` держит `ProviderCfg`; `Agent.act` берёт из неё значения и кладёт в
  `complete(...)`. Это сохраняет провайдер «тонким» (только endpoint+auth+model).
- **Разрешение ключа** (в `make_provider`): `key = os.environ.get(api_key_env)`;
  если пусто — подставляем заглушку (Ollama ключ игнорирует). Политика для облака
  (fail-fast при отсутствии ключа) — открытое решение §15.

## 7. `OpenAICompatibleProvider` — устройство

```python
class OpenAICompatibleProvider:
    def __init__(self, base_url: str, api_key: str, model: str,
                 *, timeout_s: float = 120.0, client: httpx.AsyncClient | None = None):
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"}
        self._model = model
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=timeout_s, write=30.0, pool=10.0))
        self._owns_client = client is None

    async def complete(self, *, system, messages, temperature, max_tokens) -> Completion:
        payload = {
            "model": self._model,
            "messages": [{"role": "system", "content": system},
                         *({"role": m.role, "content": m.content} for m in messages)],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = await self._post_with_retries(payload)          # см. §8
        choice = data["choices"][0]
        return Completion(
            text=(choice["message"].get("content") or ""),
            prompt_tokens=(data.get("usage") or {}).get("prompt_tokens", 0),
            completion_tokens=(data.get("usage") or {}).get("completion_tokens", 0),
            raw=data,
        )

    async def aclose(self):
        if self._owns_client:
            await self._client.aclose()
```

- **URL:** `base_url` нормализуем (`rstrip('/')`) и добавляем `/chat/completions`.
- **Запрос:** стандартный OpenAI chat-completions JSON. `system` — первым сообщением.
- **Ответ:** `text = choices[0].message.content`; usage → два инта; `raw` — весь JSON.
- **Пустой/обрезанный ответ:** `content` может быть `None` → `""`. `finish_reason ==
  "length"` (упёрлись в `max_tokens`) — пока только в `raw`; явный сигнал — §13.

## 8. Ошибки и ретраи (`_post_with_retries`)

Иерархия (`providers/base.py`):
```python
class ProviderError(Exception): ...                 # база
class ProviderHTTPError(ProviderError): ...         # неретраибельный HTTP (4xx, кроме 429)
class ProviderUnavailable(ProviderError): ...       # исчерпаны ретраи (429/5xx/сеть/таймаут)
class ProviderParseError(ProviderError): ...        # ответ не разобрался (нет choices и т.п.)
```

Политика:
- **Ретраим:** HTTP `429`, `5xx`, и сетевые `httpx.TransportError` / `httpx.TimeoutException`.
- **Не ретраим (fail fast):** прочие `4xx` (`400` bad request, `401/403` auth,
  `404` нет модели) → `ProviderHTTPError` с кодом и телом.
- **Backoff:** экспонента с джиттером —
  `delay = min(cap, base * 2**attempt) + random()*base`, `base=1.0s`, `cap=30.0s`,
  `max_attempts=5`. На `429`/`503` — уважать заголовок `Retry-After`, если есть.
- После `max_attempts` → `ProviderUnavailable` (с последней причиной в `__cause__`).
- Парсинг: если нет `choices[0].message` → `ProviderParseError` (не ретраим — ответ
  пришёл, но кривой; ретрай вряд ли поможет, лучше упасть с диагностикой).

Псевдокод:
```python
async def _post_with_retries(self, payload) -> dict:
    base, cap, max_attempts = 1.0, 30.0, 5
    for attempt in range(max_attempts):
        try:
            r = await self._client.post(self._url, json=payload, headers=self._headers)
            if r.status_code == 429 or 500 <= r.status_code < 600:
                raise _Retryable(r)               # уйдём в backoff (учесть Retry-After)
            if r.status_code >= 400:
                raise ProviderHTTPError(r.status_code, r.text)
            return r.json()
        except (httpx.TransportError, httpx.TimeoutException, _Retryable) as e:
            if attempt == max_attempts - 1:
                raise ProviderUnavailable(...) from e
            await asyncio.sleep(_backoff(attempt, e))   # Retry-After важнее формулы
```

## 9. HTTP-клиент и конкурентность

- Один `httpx.AsyncClient` на инстанс провайдера (пул соединений переиспользуется
  между вызовами). Можно **инжектить** общий клиент (`client=...`) — тогда провайдер
  его не закрывает (`_owns_client=False`).
- **Жизненный цикл:** провайдер даёт `aclose()`. Закрытие — обязанность владельца
  ростера: `Population.aclose_all()` (пост-задача того слоя) или `AsyncExitStack` в
  оркестраторе. Для одиночного smoke-скрипта — `async with`/ручной `aclose()`.
- **Конкурентность** (семафор `max_concurrency`) — на уровне оркестратора, не здесь.
  Провайдер обязан быть безопасным при параллельных `complete()` с одного инстанса —
  `httpx.AsyncClient` это поддерживает.

## 10. Фабрика `make_provider`

```python
def make_provider(cfg: ProviderCfg, *, client: httpx.AsyncClient | None = None) -> LLMProvider:
    key = os.environ.get(cfg.api_key_env) or "sk-noauth"     # Ollama ключ игнорирует
    return OpenAICompatibleProvider(cfg.base_url, key, cfg.model,
                                    timeout_s=cfg.timeout_s, client=client)
```

- Чистый конструктор: `ProviderCfg → LLMProvider`. **Без кэша** — кэширование по
  `(base_url, model)` делает `Population.add` (arch §3). Опциональный общий `client`
  — чтобы Population мог переиспользовать один пул на все провайдеры.

## 11. Совместимость провайдеров (заметки)

- **Ollama** (дефолт MVP): endpoint `http://localhost:11434/v1/chat/completions`,
  auth игнорируется (шлём заглушку), модель напр. `qwen3:8b` (должна быть `ollama
  pull`-нута). Свежие версии отдают `usage`; если нет — токены = 0 (не падаем).
- **⚠️ Reasoning-модели (qwen3 и т.п.) — важно для слоя Агента.** Проверено
  2026-05-29 на Ollama: qwen3 кладёт `<think>`-рассуждение в **отдельное поле
  `message.reasoning`**, а собственно ответ — в `message.content`. Пока модель не
  закончила думать, `content` пуст и `finish_reason == "length"`. На дефолтном
  `max_tokens=512` весь бюджет уходит на reasoning → агент получит **пустой ответ**.
  Следствия: (а) провайдер корректен — читает `content` (это финальный ответ);
  (б) слою Агента нужен **запас `max_tokens`** (для qwen3 простой ответ ≈ 200 ток.
  на reasoning), иначе JSON не успеет сгенерироваться; (в) поле `reasoning` пока
  игнорируем (лежит в `Completion.raw`), но это кандидат в приватную «рефлексию».
- **Cerebras / Gemini(OpenAI-эндпоинт) / OpenAI / Groq / OpenRouter**: тот же формат;
  меняется `base_url` + `model` + реальный ключ из env. Различия — в наличии полей
  `usage`, лимитах и кодах rate-limit (поэтому ретраи на 429/5xx обязательны).
- **Принцип устойчивости:** к ответу относимся защитно — `usage` может отсутствовать,
  `content` быть `None`; ничего из этого не должно ронять слой (кроме отсутствия
  `choices` → `ProviderParseError`).

## 12. Тест-план

**Unit (`httpx.MockTransport`, без сети):**
1. Форма запроса: `model`, `messages` начинается с `system`, далее user/assistant в
   порядке; `temperature`/`max_tokens` проброшены; заголовок `Authorization`.
2. Разбор ответа: `text` = content; `prompt/completion_tokens` из `usage`; `raw` = весь JSON.
3. `usage` отсутствует → токены `0`, без исключения.
4. `content: null` → `text == ""`.
5. Ретраи: `503, 503, 200` → один успешный `Completion`, ровно 3 запроса.
6. `429` с `Retry-After: 1` → выждать и повторить.
7. `400`/`401` → `ProviderHTTPError`, **без** ретраев (1 запрос).
8. Исчерпание ретраев (постоянный `503`) → `ProviderUnavailable`, `max_attempts` запросов.
9. Нет `choices` → `ProviderParseError`.
10. `aclose()` закрывает собственный клиент и НЕ закрывает инжектированный.

> Backoff в тестах — патчим `asyncio.sleep` (или `_backoff` → 0), чтобы не ждать.

**Smoke (`test_smoke_ollama.py`):** реальный `complete(system="Отвечай одним
словом", messages=[Message("user","Привет")], temperature=0, max_tokens=16)` против
локальной Ollama. `pytest.skip`, если `OLLAMA` недоступен (быстрый коннект-чек) —
чтобы CI без сервера был зелёным. Проверяем: непустой `text`, `prompt_tokens > 0`
(если версия отдаёт usage).

Запуск: `pytest tests/providers -q`.

## 13. Чистые швы под пост-MVP

- **`AnthropicProvider`**: тот же `LLMProvider`-протокол, нативный endpoint + prompt
  caching. Подключается рядом с `OpenAICompatibleProvider`, `make_provider`
  диспетчит по типу (поле `kind`/`base_url`).
- **`response_format` / JSON-режим**: опциональный kwarg в `complete` или поле cfg
  (`{"type":"json_object"}` у OpenAI, `format:"json"` у Ollama). Сейчас не нужно —
  JSON обеспечивает Агент промптом + парсингом.
- **`extra_headers` / `extra_body`**: проброс провайдер-специфичных параметров.
- **`finish_reason` / `model` в `Completion`**: для детекта обрезки и логов (правка arch).
- **Streaming**: не требуется (полный ответ).

## 14. Пошаговый план реализации (чеклист)

1. `pyproject.toml` (httpx + конфиг pytest), скелет `src/` с `__init__.py` (+ `src/core/`, `src/providers/`) и `tests/providers/`.
2. `src/providers/base.py`: `Message`, `Completion`, `LLMProvider` (Protocol), иерархия ошибок.
3. `src/core/config.py`: `ProviderCfg` (+ минимальный разбор provider-блока YAML позже, в слое конфига).
4. `providers/openai_compat.py`: конструктор + `complete` (happy path), без ретраев.
5. Unit-тесты 1–4 (запрос/ответ/usage/пустой content) на `MockTransport` — зелёные.
6. Добавить `_post_with_retries` (backoff, Retry-After, классификация кодов) + `aclose`.
7. Unit-тесты 5–10 (ретраи/ошибки/lifecycle) — зелёные.
8. `make_provider` + разрешение ключа из env.
9. Smoke против Ollama (с graceful-skip).
10. `providers/__init__.py` — публичный реэкспорт.

**Definition of done слоя:** `pytest tests/providers` зелёный (unit без сети);
smoke проходит при поднятой Ollama; `make_provider(cfg).complete(...)` отдаёт
осмысленный `Completion` с токенами; на 429/5xx — ретраи, на 4xx — мгновенная ошибка.

## 15. Открытые решения (подтвердить до/в ходе кодинга)

1. **[РЕШЕНО 2026-05-29] Размещение кода:** этот репозиторий
   (`/home/d/LLM_reputation`, ветка `game`); движок — в `src/` (без вложенного
   пакета `agentgames`), аналитика — в отдельной `analysis/`. Импорт — `from src.…`
   (вариант A: `src` сам по себе пакет, запуск и тесты из корня репо).
2. **Отсутствие ключа для облака:** fail-fast (бросать понятную ошибку) или мягко
   (заглушка → 401 от сервера)? Предлагаю fail-fast для не-localhost.
3. **Общий `httpx.AsyncClient` на все провайдеры** (Population инжектит один пул) vs
   по клиенту на провайдер? Предлагаю общий — меньше сокетов, проще закрывать.
4. **`max_attempts`/`base`/`cap` ретраев** — дефолты 5/1с/30с ок? Делать ли их полем `ProviderCfg`?
5. **Расширять ли `Completion`** полями `finish_reason`/`model` сейчас (правка arch) или оставить шов?
