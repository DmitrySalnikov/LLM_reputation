# TODO

убрать кириллицу

описать, как работает

подумать над ручной подменой конкретных шагов и спланировать эксперимент

rationale для ризонинг моделей - писать thinking

показывать агентам их ризонинг с прошлых раундов

# LLM_reputation

Исследование самостоятельного возникновения института репутации в группах ИИ-агентов.

Популяция LLM-агентов раунд за раундом играет в координационную игру с
предварительными переговорами (cheap-talk). Цель — проверить, складывается ли между
агентами репутация без внешних правил, только из повторяющихся взаимодействий.

## Как это устроено

Один **эпизод** = один YAML-файл конфигурации. Оркестратор строит популяцию агентов,
матчмейкер разбивает их на пары каждый раунд, а в каждой паре агенты обмениваются
короткими сообщениями и затем тайно выбирают число 0–9. Выигрыш зависит от того,
совпали ли числа, оказалось ли одно ровно на единицу больше (по модулю 10) или нет.

Игровая стратегия выбирается в конфигурации:
- `direct` — агент сразу выбирает число;
- `prediction` — агент предсказывает число партнёра, после чего отображение
  (`match` / `one_above`) превращает предсказание в собственный выбор.

## Установка и запуск

```bash
uv sync --extra dev          # установить зависимости (включая pytest)
```

Создайте `.env` с ключом провайдера (по умолчанию используется Together.ai,
OpenAI-совместимый эндпоинт):

```
TOGETHER_API_KEY=ваш_ключ
```

Запуск эпизода:

```bash
uv run python examples/orchestrator_demo.py                          # config/example.yaml
uv run python examples/orchestrator_demo.py config/example_prediction.yaml
```

Эксперимент с сохранением в SQLite: конфигурация лежит в YAML (по умолчанию
`config/experiment.yaml`), `experiment.py` её только загружает и запускает. Каждый
запуск дописывается в общую базу `experiment.db` (повторный запуск той же
конфигурации пропускается). `replay.py` воспроизводит сохранённый запуск раунд за
раундом без обращений к LLM:

```bash
uv run python experiment.py                       # config/experiment.yaml
uv run python experiment.py config/example.yaml   # другой эпизод
uv run python experiment.py config/experiment.yaml "имя запуска"
uv run python replay.py                # список запусков в базе
uv run python replay.py <run_id>       # воспроизвести запуск
uv run python replay.py <run_id> -c    # + промпты, ростер и параметры конфига
```

Чтобы включить **LLM-судью** — отдельную модель, которая после эпизода решает, возник
ли институт репутации, — добавьте блок `judge:` в YAML-конфигурацию (см.
`config/example.yaml`, закомментированный пример) или раскомментируйте `JUDGE` в
`experiment.py` (используя `JudgeCfg`). Вердикт печатается в конце эпизода,
сохраняется в базе данных, а `replay.py` подсвечивает процитированные сообщения
жёлтым цветом и добавляет раздел JUDGE VERDICT.

**Судья настраивается полностью независимо от агентов.** У блока `judge` свой
`provider` (отдельные `base_url`, `model`, `api_key_env`), поэтому агенты и судья
могут жить на разных эндпоинтах: например, агенты — на внешнем API (Together.ai), а
судья — на локальной модели через Ollama (или наоборот). Любая OpenAI-совместимая
комбинация работает.

```yaml
# Агенты — внешний API (Together.ai), судья — локальная модель через Ollama.
provider_default: &default
  base_url: https://api.together.xyz/v1   # внешний API
  api_key_env: TOGETHER_API_KEY           # ключ из .env
  model: Qwen/Qwen2.5-7B-Instruct-Turbo

judge:
  provider:
    base_url: http://localhost:11434/v1   # локальный Ollama, OpenAI-совместимый
    model: qwen2.5:72b                     # api_key_env не нужен (по умолчанию sk-noauth)
  # prompt: необязательная замена дефолтного английского промпта судьи ({transcript})

population:
  provider: *default                       # агенты используют внешний провайдер
  # ... остальная популяция
```

Чтобы поменять их местами (агенты — локально, судья — на внешнем API), задайте
`base_url: http://localhost:11434/v1` в `provider_default`, а во `judge.provider` —
внешний эндпоинт с `api_key_env`.

### Оценка судьёй задним числом (backfill)

`judge_runs.py` прогоняет LLM-судью по уже сохранённым в `experiment.db` запускам:
восстанавливает публичный cheap-talk, зовёт судью и пишет вердикт в базу (его потом
подсвечивает `replay.py`). Модель судьи берётся из блока `judge:` конфига (по умолчанию
`config/experiment.yaml`) — задайте её там (через `judge.provider` или якорь `*provider`,
чтобы судья ходил той же моделью, что и агенты). Уже оценённые запуски пропускаются, если
не передан `--force`.

```bash
uv run python judge_runs.py                            # оценить все завершённые запуски
uv run python judge_runs.py --force                    # переоценить, в т.ч. уже оценённые
uv run python judge_runs.py --config config/example.yaml   # взять судью из другого конфига
uv run python judge_runs.py --design <HASH>            # только один дизайн (флаг повторяемый)
uv run python judge_runs.py --exclude-name <LABEL>     # исключить запуски по имени
```

Запускам нужен доступный провайдер судьи; ключ читается из `.env`. Фильтры
(`--design` / `--exclude-design` / `--name` / `--exclude-name`) выбирают, какие запуски
оценивать.

### Детерминированный судья: упоминания термина

`keyword_judge.py` — альтернатива LLM-судье без LLM. Для каждого выбранного запуска ищет
ТЕРМИН (число или слово) в тексте публичных реплик и считает число РАЗНЫХ говорящих,
упомянувших его (имена говорящих не учитываются — сопоставляется только текст реплики).
Поиск подстроки с учётом регистра. Результат пишется в базу (таблица `keyword_counts`,
upsert по `(run_id, term)`), в CSV и на экран.

```bash
uv run python keyword_judge.py 123                     # упоминания «123» во всех завершённых запусках
uv run python keyword_judge.py 123 --db research.db    # другая база (по умолчанию experiment.db)
uv run python keyword_judge.py trust --csv out.csv     # слово-термин + другой путь CSV
uv run python keyword_judge.py 7 --design <HASH>       # только один дизайн (флаг повторяемый)
uv run python keyword_judge.py 7 --exclude-name <LABEL>    # исключить запуски по имени
```

`keyword_judge.py` принимает те же фильтры выбора запусков, что и `judge_runs.py`
(`--design` / `--exclude-design` / `--name` / `--exclude-name`); учитываются только
завершённые запуски. LLM не нужен — провайдер и ключ не требуются.

### Сбор статистики

`collect_stats.py` агрегирует вердикты судьи по дизайнам (`config_hash`): доля запусков, где
институт репутации возник, с 95% доверительным интервалом Вилсона. Печатает таблицу в консоль
и пишет `stats.json` + `stats.csv`.

```bash
uv run python collect_stats.py                         # все оценённые запуски -> stats.json + stats.csv
uv run python collect_stats.py --design <HASH>         # только выбранные дизайны (флаг повторяемый)
uv run python collect_stats.py --out s.json --csv s.csv    # другие пути артефактов
```

`collect_stats.py` принимает те же фильтры выбора запусков, что и `judge_runs.py`. Учитываются
только запуски, у которых уже есть вердикт судьи, — сначала прогоните `judge_runs.py`.

Для отладки можно включить трассировку точного входа LLM перед выбором числа
(флаг можно задать и в `.env`):

```bash
LLM_TRACE=1 uv run python examples/orchestrator_demo.py
```

Тесты:

```bash
uv run pytest
```

Юнит-тесты не ходят в сеть (LLM подменяется заглушкой). Smoke-тесты обращаются к
локальному Ollama и автоматически пропускаются, если он недоступен.

## Документация

Архитектура и устройство слоёв — в `docs/`: англоязычные обзоры
(`architecture.md`, `configuration.md`, `testing.md`, `conventions.md`) и подробные
проектные документы по каждому слою (`agent-games-*-plan.md`, на русском).

## Команда

[Андрей Серяков](https://github.com/AndreySeryakov)
[Крупкина Екатерина](https://github.com/ktchka)
[Быстров Андрей](https://github.com/Shougakusei)
[Сальников Дима](https://github.com/DmitrySalnikov)
