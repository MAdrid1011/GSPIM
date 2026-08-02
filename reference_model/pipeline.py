"""Independent five-stage fixture oracle for all three paper workload classes."""

from __future__ import annotations

from math import exp
from typing import Callable, Mapping, Sequence

from .contracts import ExpectedWindow
from .renderer import render_checksum
from .selection import explicit_active, explicit_geometry


def _score(depths: Sequence[float], near: float, far: float) -> float:
    normalized = [(value - near) / (far - near) for value in depths]
    return 1.0 if len(normalized) < 2 else exp(-sum(abs(right - left) for left, right in zip(normalized, normalized[1:])) / (len(normalized) - 1))


def _render_order(stable: Sequence[int], unstable: Sequence[int], stable_keys: Mapping[int, float], unstable_keys: Mapping[int, float]) -> tuple[int, ...]:
    output: list[int] = []
    stable_index = 0
    unstable_index = 0
    while stable_index < len(stable) and unstable_index < len(unstable):
        stable_id, unstable_id = stable[stable_index], unstable[unstable_index]
        if (stable_keys[stable_id], stable_id) <= (unstable_keys[unstable_id], unstable_id):
            output.append(stable_id)
            stable_index += 1
        else:
            output.append(unstable_id)
            unstable_index += 1
    return tuple(output + list(stable[stable_index:]) + list(unstable[unstable_index:]))


def _run_window(
    selected: Mapping[int, Mapping[str, object]],
    geometry_at: Callable[[int, Mapping[str, object], float], Mapping[str, object]],
    camera: Mapping[str, object],
    timestamps: Sequence[float],
    capacity: int,
) -> ExpectedWindow:
    geometry_by_frame = tuple({primitive_id: geometry_at(primitive_id, record, timestamp) for primitive_id, record in selected.items()} for timestamp in timestamps)
    matrix = camera["world_to_camera"]  # type: ignore[assignment]
    depths = {
        primitive_id: tuple(
            sum(float(matrix[2][column]) * (float(geometry[primitive_id]["mean"][0]), float(geometry[primitive_id]["mean"][1]), float(geometry[primitive_id]["mean"][2]), 1.0)[column] for column in range(4))  # type: ignore[index]
            for geometry in geometry_by_frame
        )
        for primitive_id in selected
    }
    scores = {primitive_id: _score(values, float(camera["near"]), float(camera["far"])) for primitive_id, values in depths.items()}
    stable = {primitive_id for primitive_id, score in scores.items() if score > 0.98 + 1.0 / 65536.0}
    unstable_ids = set(selected) - stable
    minimum = {primitive_id: min(values) for primitive_id, values in depths.items()}
    stable_order = tuple(sorted(stable, key=lambda primitive_id: (minimum[primitive_id], primitive_id)))
    rols: list[tuple[int, ...]] = []
    for frame_index in range(len(timestamps)):
        unstable = tuple(sorted(unstable_ids, key=lambda primitive_id: (depths[primitive_id][frame_index], primitive_id)))
        rols.append(_render_order(stable_order, unstable, minimum, {primitive_id: depths[primitive_id][frame_index] for primitive_id in unstable}))
    attribute_bytes = sum(int(record["attribute_bytes"]) for record in selected.values())
    entries = sum(len(rol) for rol in rols)
    total = attribute_bytes + entries * 4 + entries * 16
    if total > capacity:
        raise ValueError("golden fixture batch exceeds requested capacity")
    checksums = tuple(render_checksum(rol, geometry, camera) for rol, geometry in zip(rols, geometry_by_frame))
    return ExpectedWindow(set(selected), stable, unstable_ids, tuple(rols), (total,), checksums)


def run_explicit_window(
    records: Sequence[Mapping[str, object]],
    camera: Mapping[str, object],
    timestamps: Sequence[float],
    window_start: float,
    window_end: float,
    capacity: int,
) -> ExpectedWindow:
    if window_end <= window_start:
        raise ValueError("reference window end must exceed its start")
    selected = {int(record["primitive_id"]): record for record in records if explicit_active(record, window_start, window_end)}
    return _run_window(selected, lambda _primitive_id, record, timestamp: explicit_geometry(record, timestamp), camera, timestamps, capacity)


def _ex_geometry(_primitive_id: int, record: Mapping[str, object], timestamp: float) -> Mapping[str, object]:
    keys = record["keyframes"]  # type: ignore[assignment]
    if bool(record["is_static"]):
        return dict(keys[0])  # type: ignore[arg-type,index]
    if timestamp < float(keys[0]["timestamp"]) or timestamp > float(keys[-1]["timestamp"]):  # type: ignore[index]
        raise ValueError("dynamic Ex4DGS geometry is outside its declared keyframe support")
    if timestamp == float(keys[0]["timestamp"]):  # type: ignore[index]
        return dict(keys[0])  # type: ignore[arg-type,index]
    if timestamp == float(keys[-1]["timestamp"]):  # type: ignore[index]
        return dict(keys[-1])  # type: ignore[arg-type,index]
    for left, right in zip(keys, keys[1:]):  # type: ignore[arg-type]
        left_time, right_time = float(left["timestamp"]), float(right["timestamp"])
        if left_time <= timestamp <= right_time:
            ratio = (timestamp - left_time) / (right_time - left_time)
            mean = tuple((1.0 - ratio) * float(a) + ratio * float(b) for a, b in zip(left["mean"], right["mean"]))
            covariance = tuple(tuple((1.0 - ratio) * float(left["covariance"][row][column]) + ratio * float(right["covariance"][row][column]) for column in range(3)) for row in range(3))
            color = tuple((1.0 - ratio) * float(a) + ratio * float(b) for a, b in zip(left["color"], right["color"]))
            return {"mean": mean, "covariance": covariance, "opacity": (1.0 - ratio) * float(left["opacity"]) + ratio * float(right["opacity"]), "color": color}
    raise AssertionError("timestamp must be bracketed")


def run_ex4dgs_window(
    records: Sequence[Mapping[str, object]],
    camera: Mapping[str, object],
    timestamps: Sequence[float],
    window_start: float,
    window_end: float,
    capacity: int,
) -> ExpectedWindow:
    if window_end <= window_start:
        raise ValueError("reference window end must exceed its start")
    selected = {
        int(record["primitive_id"]): record
        for record in records
        if bool(record["is_static"]) or (float(record["keyframes"][0]["timestamp"]) < window_end and float(record["keyframes"][-1]["timestamp"]) >= window_start)
    }
    for primitive_id, record in sorted(selected.items()):
        if bool(record["is_static"]):
            continue
        keys = record["keyframes"]
        start, end = float(keys[0]["timestamp"]), float(keys[-1]["timestamp"])
        unsupported = tuple(timestamp for timestamp in timestamps if timestamp < start or timestamp > end)
        if unsupported:
            raise ValueError(
                f"dynamic Ex4DGS record {primitive_id} does not support every requested S2 frame: {unsupported}"
            )
    return _run_window(selected, _ex_geometry, camera, timestamps, capacity)


def _anchor_geometry(_primitive_id: int, record: Mapping[str, object], timestamp: float) -> Mapping[str, object]:
    for frame in record["frame_geometry"]:  # type: ignore[union-attr]
        if float(frame["timestamp"]) == float(timestamp):  # type: ignore[index]
            return {
                "mean": frame["mean"],  # type: ignore[index]
                "covariance": frame["covariance"],  # type: ignore[index]
                "opacity": frame["opacity"],  # type: ignore[index]
                "color": frame["color"],  # type: ignore[index]
            }
    raise ValueError("anchor fixture has no generated geometry for the requested timestamp")


def run_anchored_window(
    anchors: Sequence[Mapping[str, object]],
    payloads: Mapping[int, Mapping[str, object]],
    camera: Mapping[str, object],
    timestamps: Sequence[float],
    window_start: float,
    window_end: float,
    capacity: int,
) -> ExpectedWindow:
    if window_end <= window_start:
        raise ValueError("reference window end must exceed its start")
    selected: dict[int, Mapping[str, object]] = {}
    for anchor in anchors:
        if float(anchor["start"]) < window_end and float(anchor["end"]) >= window_start:
            for primitive_id in range(int(anchor["binding_start"]), int(anchor["binding_start"]) + int(anchor["binding_count"])):
                selected[primitive_id] = payloads[primitive_id]
    for primitive_id, payload in sorted(selected.items()):
        available = {float(frame["timestamp"]) for frame in payload["frame_geometry"]}  # type: ignore[union-attr,index]
        missing = tuple(timestamp for timestamp in timestamps if timestamp not in available)
        if missing:
            raise ValueError(
                f"anchor payload {primitive_id} does not support every requested S2 frame: {missing}"
            )
    return _run_window(selected, _anchor_geometry, camera, timestamps, capacity)
