"""Independent S3/S4 helper equations for golden comparisons."""

from math import exp
from typing import Sequence


def depth_stability(depths: Sequence[float], near: float, far: float) -> float:
    if far <= near:
        raise ValueError("far must exceed near")
    if len(depths) < 2:
        return 1.0
    values = [(depth - near) / (far - near) for depth in depths]
    return exp(-sum(abs(right - left) for left, right in zip(values, values[1:])) / (len(values) - 1))
