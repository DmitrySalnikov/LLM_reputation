# MVP — диаграмма последовательности (поток одного раунда)

UML sequence diagram потока одного раунда. Сопровождает `agent-games-mvp-explained.md`
(словесное описание по слоям) и `agent-games-mvp-arch.md` (интерфейсы/код).
Рендерится в Obsidian и на GitHub. При изменениях правим вместе с остальными.

Ключевое: **Оркестратор — единственный «дирижёр»**. Он зовёт Матчинг, получает
план обратно, и **отдельно** зовёт Игру; Матчинг Игру не вызывает. Реальная
вложенность вызовов — `Оркестратор → Игра → Агент → Провайдер`.

```mermaid
sequenceDiagram
    participant O as Оркестратор
    participant M as Матчинг
    participant G as Игра
    participant A as Агент A
    participant B as Агент B
    participant P as Провайдер→LLM
    participant DB as SQLite

    O->>M: plan_round(agent_ids, r)
    activate M
    M-->>O: RoundPlan(пары, idle)
    deactivate M

    Note over O,B: для каждой пары — параллельно
    O->>G: play_pairing(A, B, r)
    activate G

    Note over G,B: переговоры
    loop пока оба ready (latch) или лимит
        G->>A: act(TALK, ситуация)
        activate A
        A->>P: complete(system, дневник+ситуация)
        activate P
        P-->>A: {message, ready}
        deactivate P
        A-->>G: реплика A
        deactivate A
        G->>B: act(TALK, ситуация)
        activate B
        B->>P: complete(...)
        activate P
        P-->>B: {message, ready}
        deactivate P
        B-->>G: реплика B
        deactivate B
    end

    Note over G,B: тайный выбор (независимо, вслепую)
    G->>A: act(DECIDE)
    activate A
    A->>P: complete(...)
    activate P
    P-->>A: {number, rationale}
    deactivate P
    A-->>G: x (втайне)
    deactivate A
    G->>B: act(DECIDE)
    activate B
    B->>P: complete(...)
    activate P
    P-->>B: {number, rationale}
    deactivate P
    B-->>G: y (втайне)
    deactivate B

    Note over G: resolve(x,y) → исход, очки; запись в дневники
    G-->>O: PairingRecord
    deactivate G

    O->>DB: write_round(plan, records) — одна транзакция
    activate DB
    DB-->>O: ok
    deactivate DB
```

Что показано:
- **O** — корень: сам зовёт `M`, сам зовёт `G`; `M` и `G` — соседи, не вложены.
- Сплошные стрелки — вызовы с параметрами; пунктирные — ответы (`RoundPlan`,
  `{message, ready}`, `x`/`y`, `PairingRecord`).
- Activation-бары — истинная глубина вызовов: `O → G → A → P`.
- Тайный выбор: `G` спрашивает `A` и `B` по отдельности; `x`/`y` уходят только в
  `G` и друг другу не видны.
