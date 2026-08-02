"""S2--S4 math: depth stability, differentiated sorting, and bounded batches."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite
from struct import pack, unpack
from typing import Mapping, Sequence


def depth_stability_score(depths: Sequence[float], near: float, far: float) -> float:
    """Compute the paper's exponential normalized mean absolute depth variation."""

    if far <= near:
        raise ValueError("far must exceed near")
    if len(depths) < 2:
        return 1.0
    normalized = [(float(depth) - near) / (far - near) for depth in depths]
    variation = sum(abs(right - left) for left, right in zip(normalized, normalized[1:]))
    return exp(-variation / (len(normalized) - 1))


@dataclass(frozen=True)
class RenderOrderPlan:
    stable_order: tuple[int, ...]
    unstable_order_by_frame: tuple[tuple[int, ...], ...]
    rols: tuple[tuple[int, ...], ...]


def _sortable_depth_key(depth: float) -> int:
    """Map an IEEE-754 depth to an unsigned key with numeric ordering."""

    value = float(depth)
    if not isfinite(value):
        raise ValueError("sorting depth must be finite")
    bits = unpack(">Q", pack(">d", value))[0]
    return (~bits & ((1 << 64) - 1)) if bits >> 63 else bits ^ (1 << 63)


def fixed_width_radix_sort(ids: Sequence[int], keys: Mapping[int, float]) -> tuple[int, ...]:
    """Stable 64-bit radix sort with primitive ID as a deterministic tie-breaker."""

    output = list(sorted(ids))
    for shift in range(0, 64, 8):
        buckets: list[list[int]] = [[] for _ in range(256)]
        for primitive_id in output:
            buckets[(_sortable_depth_key(keys[primitive_id]) >> shift) & 0xFF].append(primitive_id)
        output = [primitive_id for bucket in buckets for primitive_id in bucket]
    return tuple(output)


def _merge(stable: Sequence[int], unstable: Sequence[int], stable_keys: Mapping[int, float], unstable_keys: Mapping[int, float]) -> tuple[int, ...]:
    output: list[int] = []
    stable_index = 0
    unstable_index = 0
    while stable_index < len(stable) and unstable_index < len(unstable):
        left, right = stable[stable_index], unstable[unstable_index]
        if (stable_keys[left], left) <= (unstable_keys[right], right):
            output.append(left)
            stable_index += 1
        else:
            output.append(right)
            unstable_index += 1
    return tuple(output + list(stable[stable_index:]) + list(unstable[unstable_index:]))


def _merge_path_partition(
    diagonal: int,
    stable: Sequence[int],
    unstable: Sequence[int],
    stable_keys: Mapping[int, float],
    unstable_keys: Mapping[int, float],
) -> tuple[int, int]:
    """Return the merge-path co-rank for one diagonal, favoring stable ties."""

    low = max(0, diagonal - len(unstable))
    high = min(diagonal, len(stable))
    while low <= high:
        stable_count = (low + high) // 2
        unstable_count = diagonal - stable_count
        stable_left = (stable_keys[stable[stable_count - 1]], stable[stable_count - 1]) if stable_count else None
        stable_right = (stable_keys[stable[stable_count]], stable[stable_count]) if stable_count < len(stable) else None
        unstable_left = (unstable_keys[unstable[unstable_count - 1]], unstable[unstable_count - 1]) if unstable_count else None
        unstable_right = (unstable_keys[unstable[unstable_count]], unstable[unstable_count]) if unstable_count < len(unstable) else None
        if stable_left is not None and unstable_right is not None and stable_left > unstable_right:
            high = stable_count - 1
        elif unstable_left is not None and stable_right is not None and unstable_left > stable_right:
            low = stable_count + 1
        else:
            return stable_count, unstable_count
    raise AssertionError("merge-path partition search failed")


def merge_path(
    stable: Sequence[int],
    unstable: Sequence[int],
    stable_keys: Mapping[int, float],
    unstable_keys: Mapping[int, float],
    partitions: int = 4,
) -> tuple[int, ...]:
    """Partition a sorted merge by merge-path, then emit deterministic segments.

    The partitions express the paper's parallel merge-path decomposition. This
    functional artifact emits the independent segments serially and makes no
    throughput claim.
    """

    total = len(stable) + len(unstable)
    if total == 0:
        return ()
    lanes = max(1, min(int(partitions), total))
    ranks = tuple(_merge_path_partition(total * lane // lanes, stable, unstable, stable_keys, unstable_keys) for lane in range(lanes + 1))
    return tuple(
        primitive_id
        for (stable_start, unstable_start), (stable_end, unstable_end) in zip(ranks, ranks[1:])
        for primitive_id in _merge(
            stable[stable_start:stable_end],
            unstable[unstable_start:unstable_end],
            stable_keys,
            unstable_keys,
        )
    )


def plan_render_orders(depths: Mapping[int, Sequence[float]], stable_ids: set[int]) -> RenderOrderPlan:
    """Sort stable IDs once and unstable IDs for every frame, then merge ROLs."""

    if not depths:
        return RenderOrderPlan((), (), ())
    width = len(next(iter(depths.values())))
    if width < 1 or any(len(values) != width for values in depths.values()):
        raise ValueError("depth sequences must have one nonempty window width")
    if not stable_ids.issubset(depths):
        raise ValueError("stable IDs must be active IDs")
    unstable_ids = set(depths) - stable_ids
    minimum = {primitive_id: min(values) for primitive_id, values in depths.items()}
    stable = fixed_width_radix_sort(tuple(stable_ids), minimum)
    unstable_orders: list[tuple[int, ...]] = []
    rols: list[tuple[int, ...]] = []
    for frame_index in range(width):
        unstable = fixed_width_radix_sort(
            tuple(unstable_ids),
            {primitive_id: depths[primitive_id][frame_index] for primitive_id in unstable_ids},
        )
        unstable_orders.append(unstable)
        rols.append(merge_path(stable, unstable, minimum, {primitive_id: depths[primitive_id][frame_index] for primitive_id in unstable}))
    return RenderOrderPlan(stable, tuple(unstable_orders), tuple(rols))


@dataclass(frozen=True)
class BatchFootprint:
    attribute_bytes: int
    local_index_bytes: int
    raster_intermediate_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.attribute_bytes + self.local_index_bytes + self.raster_intermediate_bytes


@dataclass(frozen=True)
class Batch:
    batch_id: int
    stable_ids: tuple[int, ...]
    attribute_ids: tuple[int, ...]
    frame_rols: tuple[tuple[int, ...], ...]
    footprint: BatchFootprint


class CapacityError(ValueError):
    """A required batch segment cannot fit without violating the L2 contract."""


def _segments_for_groups(plan: RenderOrderPlan, groups: Sequence[tuple[int, ...]]) -> list[list[list[int]]]:
    membership = {primitive_id: group_id for group_id, group in enumerate(groups) for primitive_id in group}
    output: list[list[list[int]]] = [[[] for _ in plan.rols] for _ in groups]
    for frame_index, rol in enumerate(plan.rols):
        group_id = 0
        for primitive_id in rol:
            if primitive_id in membership:
                group_id = membership[primitive_id]
            output[group_id][frame_index].append(primitive_id)
    return output


def _footprint(frame_rols: Sequence[Sequence[int]], attribute_bytes: Mapping[int, int], index_bytes: int, raster_bytes: int) -> tuple[tuple[int, ...], BatchFootprint]:
    attribute_ids = tuple(sorted(set().union(*(set(rol) for rol in frame_rols))))
    missing = [primitive_id for primitive_id in attribute_ids if primitive_id not in attribute_bytes]
    if missing or any(attribute_bytes[primitive_id] < 1 for primitive_id in attribute_ids):
        raise ValueError("every batch primitive needs a positive exact attribute size")
    entries = sum(len(rol) for rol in frame_rols)
    return attribute_ids, BatchFootprint(sum(attribute_bytes[primitive_id] for primitive_id in attribute_ids), entries * index_bytes, entries * raster_bytes)


def make_batch_plan(
    order_plan: RenderOrderPlan,
    attribute_bytes: Mapping[int, int],
    l2_capacity: int,
    local_index_bytes: int = 4,
    raster_intermediate_bytes: int = 16,
) -> tuple[Batch, ...]:
    """Choose maximal shared stable intervals while accounting all stored bytes."""

    if l2_capacity < 1 or local_index_bytes < 1 or raster_intermediate_bytes < 1:
        raise ValueError("capacity accounting parameters must be positive")
    if not order_plan.rols:
        return ()
    stable = order_plan.stable_order
    if not stable:
        attributes, footprint = _footprint(order_plan.rols, attribute_bytes, local_index_bytes, raster_intermediate_bytes)
        if footprint.total_bytes > l2_capacity:
            raise CapacityError("unstable-only ROL cannot fit in the effective L2")
        return (Batch(0, (), attributes, order_plan.rols, footprint),)

    groups: list[tuple[int, ...]] = []
    start = 0
    while start < len(stable):
        chosen: tuple[int, ...] | None = None
        for end in range(start + 1, len(stable) + 1):
            candidate = tuple(stable[start:end])
            probe_groups = groups + [candidate] + [(primitive_id,) for primitive_id in stable[end:]]
            segments = _segments_for_groups(order_plan, probe_groups)[len(groups)]
            _, footprint = _footprint(segments, attribute_bytes, local_index_bytes, raster_intermediate_bytes)
            if footprint.total_bytes <= l2_capacity:
                chosen = candidate
            else:
                break
        if chosen is None:
            raise CapacityError(f"stable primitive {stable[start]} and its required local records cannot fit in the effective L2")
        groups.append(chosen)
        start += len(chosen)
    segments = _segments_for_groups(order_plan, groups)
    batches: list[Batch] = []
    for batch_id, (group, per_frame) in enumerate(zip(groups, segments)):
        attributes, footprint = _footprint(per_frame, attribute_bytes, local_index_bytes, raster_intermediate_bytes)
        if footprint.total_bytes > l2_capacity:
            raise AssertionError("greedy batch construction violated capacity")
        batches.append(Batch(batch_id, group, attributes, tuple(tuple(rol) for rol in per_frame), footprint))
    return tuple(batches)
