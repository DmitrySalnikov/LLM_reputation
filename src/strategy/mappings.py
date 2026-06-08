"""Реестр чистых отображений предсказания партнёра в итоговый выбор."""

from __future__ import annotations

from typing import Callable

# Чистая функция: предсказанное число партнёра -> собственный выбор (0..9).
PredictionMapping = Callable[[int], int]

_REGISTRY: dict[str, PredictionMapping] = {
    "match": lambda p: p,
    "one_above": lambda p: (p + 1) % 10,
}


def get_mapping(name: str) -> PredictionMapping:
    """Вернуть функцию отображения по имени.

    Args:
        name: Имя отображения, зарегистрированное в реестре.

    Returns:
        Чистая функция отображения предсказания в выбор.

    Raises:
        ValueError: Если имя отображения не зарегистрировано.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"неизвестное отображение предсказания: {name!r}; "
            f"доступны: {sorted(_REGISTRY)}"
        ) from None
