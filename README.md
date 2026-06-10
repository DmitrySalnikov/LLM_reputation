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
PYTHONPATH=. uv run python examples/orchestrator_demo.py                          # config/example.yaml
PYTHONPATH=. uv run python examples/orchestrator_demo.py config/example_prediction.yaml
```

Эксперимент с сохранением в SQLite: конфигурация лежит в YAML (по умолчанию
`config/experiment.yaml`), `experiment.py` её только загружает и запускает. Каждый
запуск дописывается в общую базу `experiment.db` (повторный запуск той же
конфигурации пропускается). `replay.py` воспроизводит сохранённый запуск раунд за
раундом без обращений к LLM:

```bash
PYTHONPATH=. uv run python experiment.py                       # config/experiment.yaml
PYTHONPATH=. uv run python experiment.py config/example.yaml   # другой эпизод
PYTHONPATH=. uv run python experiment.py config/experiment.yaml "имя запуска"
PYTHONPATH=. uv run python replay.py                # список запусков в базе
PYTHONPATH=. uv run python replay.py <run_id>       # воспроизвести запуск
PYTHONPATH=. uv run python replay.py <run_id> -c    # + промпты, ростер и параметры конфига
```

Для отладки можно включить трассировку точного входа LLM перед выбором числа
(флаг можно задать и в `.env`):

```bash
LLM_TRACE=1 PYTHONPATH=. uv run python examples/orchestrator_demo.py
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
