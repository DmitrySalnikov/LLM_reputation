"""Registry of pure mappings from a partner prediction to the final choice."""

from __future__ import annotations

from typing import Callable

# Pure function: partner's predicted number -> own choice (0..9).
PredictionMapping = Callable[[int], int]

_REGISTRY: dict[str, PredictionMapping] = {
    "match": lambda p: p,
    "one_above": lambda p: (p + 1) % 10,
}


def get_mapping(name: str) -> PredictionMapping:
    """Return the mapping function by name.

    Args:
        name: Name of the mapping registered in the registry.

    Returns:
        Pure function mapping a prediction to a choice.

    Raises:
        ValueError: If the mapping name is not registered.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown prediction mapping: {name!r}; "
            f"available: {sorted(_REGISTRY)}"
        ) from None
