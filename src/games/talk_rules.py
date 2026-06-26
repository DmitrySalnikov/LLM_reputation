from __future__ import annotations

from typing import Protocol


class TalkStopRule(Protocol):
    """Стоп-правило cheap-talk: как завершается обмен репликами в паре.

    Цикл переговоров ходит по очереди (a, b, a, b…) до жёсткого потолка max_talk_turns.
    Правило управляет тремя микрорешениями на каждом ходу — это раскладывает поведение на
    два НЕЗАВИСИМЫХ признака (молчать ли после finish; липкий ли finish) и оставляет место
    будущим вариантам (`either`, `fixed_k`):

      skip_turn  — должен ли уже-готовый говорящий промолчать в этот ход (молчит / говорит дальше);
      next_ready — как обновить флаг готовности по новой реплике (липкий / отзываемый finish);
      is_over    — пора ли завершить переговоры (после очередной реплики).

    `ready` — карта {agent_id: готов ли агент закончить}; next_ready решает, перезаписывать её
    сигналом (отзываемо) или защёлкивать (`prev or signal`).
    """

    def skip_turn(self, speaker_id: str, ready: dict[str, bool]) -> bool: ...

    def next_ready(self, prev: bool, signal: bool) -> bool: ...

    def is_over(self, ready: dict[str, bool]) -> bool: ...


class BothReadyLatch:
    """Защёлка: выставил finish — замолкаешь и ждёшь, пока дозреет второй.

    Признаки: молчит после finish + finish липкий. Переговоры рвутся, когда finish у ОБОИХ;
    готовый агент больше не получает слова (его finish уже не изменить)."""

    def skip_turn(self, speaker_id: str, ready: dict[str, bool]) -> bool:
        return ready[speaker_id]

    def next_ready(self, prev: bool, signal: bool) -> bool:
        return prev or signal

    def is_over(self, ready: dict[str, bool]) -> bool:
        return all(ready.values())


class BothReadyRevocable:
    """Отзываемый finish: продолжаешь говорить даже выставив finish, и можешь его снять.

    Признаки: говорит дальше + finish отзываемый. Флаг ready перезаписывается каждой репликой,
    поэтому finish=false снимает прежнее согласие. Стоп — когда finish у ОБОИХ одновременно."""

    def skip_turn(self, speaker_id: str, ready: dict[str, bool]) -> bool:
        return False

    def next_ready(self, prev: bool, signal: bool) -> bool:
        return signal

    def is_over(self, ready: dict[str, bool]) -> bool:
        return all(ready.values())


class BothReadyCommitted:
    """Липкий finish + продолжение диалога: выставил finish — назад не отыграешь, но говоришь дальше.

    Признаки: говорит дальше (как revocable) + finish липкий (как latch). Агент берёт ходы и
    после finish, но ready защёлкнут (`prev or signal`) — отозвать нельзя. Стоп — когда finish
    хоть раз выставил КАЖДЫЙ."""

    def skip_turn(self, speaker_id: str, ready: dict[str, bool]) -> bool:
        return False

    def next_ready(self, prev: bool, signal: bool) -> bool:
        return prev or signal

    def is_over(self, ready: dict[str, bool]) -> bool:
        return all(ready.values())


_RULES = {
    "both_ready_latch": BothReadyLatch,
    "both_ready_revocable": BothReadyRevocable,
    "both_ready_committed": BothReadyCommitted,
}


def make_talk_rule(name: str) -> TalkStopRule:
    """Собрать стоп-правило cheap-talk по имени (единый реестр; см. также _validate в config).

    Args:
        name: Имя правила ("both_ready_latch" | "both_ready_revocable").

    Returns:
        Экземпляр стоп-правила.

    Raises:
        ValueError: Если имя правила не распознано.
    """
    try:
        return _RULES[name]()
    except KeyError:
        raise ValueError(f"неизвестное talk_stop_rule: {name!r}")
