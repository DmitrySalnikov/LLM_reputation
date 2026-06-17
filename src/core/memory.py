from __future__ import annotations

from dataclasses import dataclass

from src.core.config import GameCfg
from src.providers.base import Message

# Запасной набор шаблонов транскрипта, если рендеру не передали GameCfg (прямые вызовы
# render() в тестах, фазы без игрового конфига). Совпадает с дефолтами GameCfg.
_DEFAULT_GAME = GameCfg()


@dataclass
class MemoryEntry:
    round: int
    my_id: str              # id самого агента (для метки "<my_id> (you)" в дневнике)
    partner_id: str
    transcript: list[dict]  # [{speaker, text, ready}]
    my_number: int
    my_rationale: str
    partner_number: int
    outcome: str
    payoff: float            # выигрыш самого агента в этом раунде
    partner_payoff: float    # выигрыш партнёра (для симметричной строки "Payoffs: ...")
    score: float = 0.0       # накопленный счёт агента ДО этого раунда (как в фазовом хедере)
    my_predicted: int | None = None  # стратегия prediction; None для direct
    my_reflection: str | None = None  # пост-игровая рефлексия; None, если выключена


class Memory:
    def __init__(self) -> None:
        self.entries: list[MemoryEntry] = []
        self.notes: str | None = None   # сжатая память (memory notes); None = заметок ещё нет
        self.noted_upto: int = 0         # сколько записей уже свёрнуто в notes (граница буфера)

    def add(self, entry: MemoryEntry) -> None:
        self.entries.append(entry)

    def set_notes(self, notes: str) -> None:
        """Запомнить свежие заметки и сдвинуть границу: всё сыгранное на сейчас — свёрнуто."""
        self.notes = notes
        self.noted_upto = len(self.entries)

    def render(self, window: int | None, cfg: GameCfg | None = None) -> list[Message]:
        # Прошлые раунды отрисовываются как один игровой транскрипт (теги <game>/<you>/<имя>);
        # шаблоны берутся из cfg (или дефолтных, если конфиг не передали).
        cfg = cfg or _DEFAULT_GAME
        # Без заметок — обычный транскрипт прошлых раундов (с учётом окна).
        if self.notes is None:
            entries = _window(self.entries, window)
            if not entries:
                return []
            body = "\n".join(_render_entry(e, cfg) for e in entries)
            return [Message("user", body)]
        # С заметками: сжатые заметки + сырой буфер раундов, сыгранных после последней
        # консолидации (вместо полной истории). Окно ограничивает только буфер.
        parts = ["Your notes from earlier rounds:\n" + self.notes]
        buffer = _window(self.entries[self.noted_upto:], window)
        if buffer:
            parts.append("Your rounds since those notes:\n" + "\n".join(
                _render_entry(e, cfg) for e in buffer
            ))
        return [Message("user", "\n\n".join(parts))]


def render_turns(transcript: list[dict], me_id: str, msg_self: str, msg_partner: str) -> str:
    """Отрисовать реплики cheap-talk тегами <you>/<имя> — общий код для истории и живого фида.

    Args:
        transcript: Реплики раунда (каждая со `speaker` и `text`).
        me_id: Идентификатор зрителя (его реплики идут как <you>).
        msg_self: Шаблон собственной реплики ({text}).
        msg_partner: Шаблон реплики партнёра ({partner}/{text}).

    Returns:
        Реплики, склеенные через перевод строки (пустая строка, если их нет).
    """
    lines = []
    for t in transcript:
        speaker = t.get("speaker")
        text = t.get("text", "")
        if speaker == me_id:
            lines.append(msg_self.replace("{text}", text))
        else:
            lines.append(msg_partner.replace("{partner}", speaker).replace("{text}", text))
    return "\n".join(lines)


def pick_opener(transcript: list[dict], me_id: str, partner_id: str,
                opener_self: str, opener_partner: str) -> str:
    """Фраза {opener}: кто открыл раунд с точки зрения зрителя `me_id`.

    Первым говорит тот, кто стоит первым в transcript. Если это сам зритель — берём
    opener_self, иначе opener_partner с подставленным именем партнёра. Единый источник
    истины для строки открытия — и в истории (memory), и в живом talk_prompt (reputation_pd)."""
    if transcript and transcript[0].get("speaker") == me_id:
        return opener_self
    return opener_partner.replace("{partner}", partner_id)


def _window(entries: list[MemoryEntry], window: int | None) -> list[MemoryEntry]:
    if window is None:
        return entries
    if window <= 0:
        return []
    return entries[-window:]


def _render_entry(e: MemoryEntry, cfg: GameCfg) -> str:
    # Один прошлый раунд как кусок транскрипта: открытие чата, реплики, закрытие, своё
    # секретное число (<you>), вскрывающая строка результата. Партнёр назван по имени, свои
    # реплики — <you>; кто открыл раунд видно по первому говорящему в transcript.
    opener = pick_opener(e.transcript, e.my_id, e.partner_id, cfg.opener_self, cfg.opener_partner)
    lines = [
        cfg.history_round_prompt
        .replace("{round}", str(e.round))
        .replace("{partner}", e.partner_id)
        .replace("{opener}", opener)
    ]
    if e.transcript:
        lines.append(render_turns(e.transcript, e.my_id, cfg.msg_self, cfg.msg_partner))
    reason = cfg.reason_agreed if _both_agreed(e) else cfg.reason_limit
    lines.append(cfg.history_close_prompt.replace("{reason}", reason))
    lines.append(cfg.msg_self.replace("{text}", str(e.my_number)))   # своё секретное число
    lines.append(
        cfg.history_result_prompt
        .replace("{round}", str(e.round))
        .replace("{partner}", e.partner_id)
        .replace("{my_number}", str(e.my_number))
        .replace("{partner_number}", str(e.partner_number))
        .replace("{payoff}", f"{e.payoff:g}")
        .replace("{partner_payoff}", f"{e.partner_payoff:g}")
        .replace("{total}", f"{e.score + e.payoff:g}")
    )
    # Приватные следы выключенных фич (rationale/prediction/reflection) — только если есть.
    if e.my_predicted is not None:
        lines.append(f"<you>(I predicted {e.partner_id} would pick {e.my_predicted})</you>")
    if e.my_rationale:
        lines.append(f"<you>(my reasoning: {e.my_rationale})</you>")
    if e.my_reflection:
        lines.append(f"<you>(my takeaway: {e.my_reflection})</you>")
    return "\n".join(lines)


def both_agreed(transcript: list[dict], a_id: str, b_id: str) -> bool:
    """Закрылся ли чат по согласию: оба участника хоть раз выставили finish/ready=true.

    Иначе чат упёрся в лимит реплик. Единый источник истины для строки закрытия — и в
    истории прошлых раундов (memory), и в живой фазе DECIDE (reputation_pd)."""
    ready_speakers = {t.get("speaker") for t in transcript if t.get("ready")}
    return {a_id, b_id} <= ready_speakers


def _both_agreed(e: MemoryEntry) -> bool:
    return both_agreed(e.transcript, e.my_id, e.partner_id)
