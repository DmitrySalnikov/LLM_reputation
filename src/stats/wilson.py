from __future__ import annotations

import math


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Доверительный интервал Вилсона для доли k/n (по умолчанию 95%, z=1.96).

    Корректен у краёв 0/1 и при малом n, в отличие от нормального приближения (Wald).

    Args:
        k: Число «успехов» (прогонов, где институт репутации возник).
        n: Общее число испытаний (оценённых прогонов). Должно быть > 0.
        z: z-квантиль (1.96 ≈ 95%).

    Returns:
        Пара (lo, hi), обрезанная в [0, 1].

    Raises:
        ValueError: n <= 0 или k вне диапазона [0, n].
    """
    if n <= 0:
        raise ValueError("n должно быть > 0")
    if not 0 <= k <= n:
        raise ValueError("k должно быть в диапазоне [0, n]")
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, center - half), min(1.0, center + half)
