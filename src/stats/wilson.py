from __future__ import annotations

import math


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson confidence interval for the proportion k/n (95% by default, z=1.96).

    Accurate near the 0/1 edges and for small n, unlike the normal (Wald) approximation.

    Args:
        k: Number of "successes" (runs where the reputation institution emerged).
        n: Total number of trials (scored runs). Must be > 0.
        z: z-quantile (1.96 ≈ 95%).

    Returns:
        A pair (lo, hi), clipped to [0, 1].

    Raises:
        ValueError: n <= 0 or k outside the range [0, n].
    """
    if n <= 0:
        raise ValueError("n must be > 0")
    if not 0 <= k <= n:
        raise ValueError("k must be within [0, n]")
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, center - half), min(1.0, center + half)
