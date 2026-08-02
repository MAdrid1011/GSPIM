"""Independent explicit-4D covariance and S1 selection equations."""

from math import log
from typing import Mapping


def covariance_4d(record: Mapping[str, object]) -> tuple[tuple[float, float, float, float], ...]:
    rotation = record["temporal_rotation"]
    scale = tuple(float(value) for value in record["temporal_scale"])  # type: ignore[arg-type]
    return tuple(
        tuple(
            sum(float(rotation[row][axis]) * scale[axis] ** 2 * float(rotation[column][axis]) for axis in range(4))  # type: ignore[index]
            for column in range(4)
        )
        for row in range(4)
    )


def explicit_geometry(record: Mapping[str, object], timestamp: float) -> Mapping[str, object]:
    """Independently condition a native 4D Gaussian on one frame timestamp."""

    covariance = covariance_4d(record)
    sigma_tt = covariance[3][3]
    if sigma_tt <= 0.0:
        raise ValueError("temporal variance must be positive")
    mean = tuple(float(value) for value in record["mean"])  # type: ignore[arg-type]
    delta = float(timestamp) - float(record["temporal_mean"])
    spatial_temporal = tuple(covariance[axis][3] for axis in range(3))
    conditional_mean = tuple(mean[axis] + spatial_temporal[axis] * delta / sigma_tt for axis in range(3))
    conditional_covariance = tuple(
        tuple(covariance[row][column] - spatial_temporal[row] * spatial_temporal[column] / sigma_tt for column in range(3))
        for row in range(3)
    )
    return {"mean": conditional_mean, "covariance": conditional_covariance, "opacity": record["opacity"], "color": record["color"]}


def explicit_active(record: Mapping[str, object], start: float, end: float, lsb: float = 1.0 / 65536.0) -> bool:
    sigma_tt = covariance_4d(record)[3][3]
    mean = float(record["temporal_mean"])
    distance = max(start - mean, 0.0, mean - end)
    return sigma_tt * log(20.0) - 0.5 * distance * distance >= -lsb
