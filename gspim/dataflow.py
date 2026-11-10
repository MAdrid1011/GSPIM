"""S2--S4 math: depth stability, differentiated sorting, and bounded batches."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite
from struct import pack, unpack
from typing import Mapping, Sequence


def depth_stability_score(depths: Sequence[float], near: float, far: float) -> float:
    """Compute exponential normalized mean absolute depth variation."""

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

    Co-ranks divide the output into independent segments. Each segment merges
    its corresponding slices with deterministic primitive-ID tie breaking.
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


def cuda_render_orders(means, view, near: float, far: float, operators, statistics=None):
    """Classify the native window and merge CUB-sorted stable/unstable IDs.

    Sorting keys for the stable sequence are window minima, including during
    the merge. Primitive ID breaks equal-depth ties in both paths.
    """
    import torch

    if not isfinite(near) or not isfinite(far) or far <= near:
        raise ValueError("depth stability requires finite, increasing camera bounds")
    if statistics is None:
        depth = (means @ view[0, 2, :3] + view[0, 2, 3]).contiguous()
        score, minimum = operators.window_statistics(depth, far - near)
    else:
        depth, score, minimum = statistics.run(means, view, far - near)
    width, count = depth.shape
    stable, unstable, next_delta = operators.classify(score)

    def sort(values, keys):
        # CUB transforms IEEE float keys internally, including negative depths.
        return operators.radix(keys.contiguous(), values.contiguous(), 0, 32)[1]

    stable = sort(stable, minimum[stable.long()])
    if len(stable) == count:
        orders = stable.long()[None].expand(width, -1)
    else:
        unstable = torch.stack([sort(unstable, depth[frame, unstable.long()]) for frame in range(width)]).contiguous()
        orders = operators.merge(stable, unstable, minimum.contiguous(), depth)
    return orders, next_delta, stable, depth


def cuda_batch_boundaries(orders, stable_ids, projected, shared_bytes: int, l2_bytes: int, image_width: int, image_height: int, operators=None, shared_tiles=False, layout=None, resident_blocks=None, cost_prefix=None):
    """Bound batch inputs after reserving all potentially resident tile state."""
    import torch

    width, count = orders.shape
    device = orders.device
    radii, xy = projected[:2]
    properties = torch.cuda.get_device_properties(device)
    tile_width, tile_height = (image_width + 15) // 16, (image_height + 15) // 16
    masked = shared_tiles and count > 0 and count.bit_length() + width + 8 <= 32
    if resident_blocks is None:
        if operators is None:
            raise ValueError("reference planning requires the device's resident block limit")
        resident_blocks = operators.raster_blocks(xy, masked)
    if resident_blocks < 1:
        raise ValueError("resident block limit must be positive")
    tiles = tile_width * tile_height
    # C/T occupies registers while a CTA executes; conservatively also charge
    # its backing storage. A signed T carries termination for multi-frame paths.
    parts = 2 if masked else 1
    resident_tiles = min(properties.multi_processor_count * resident_blocks, width * tiles * parts)
    reserved = resident_tiles * (256 // parts) * (16 if width > 1 or masked else 17)
    reserved += width * tiles * (4 if masked else 1)
    # Shared jobs have one tile list, while independent lists have one per frame.
    reserved += (1 if shared_tiles else width) * tiles * (8 + 4) + 4
    reserved += 16 if layout is not None else 0
    budget = l2_bytes - reserved
    if budget <= 0:
        raise ValueError("L2 cannot hold the resident raster state")
    if count == 0:
        return orders.new_empty((width, 0)), 1, reserved
    if shared_tiles and len(stable_ids) != count:
        raise ValueError("shared tile lists require a common stable order")
    if cost_prefix is not None and (operators is None or not shared_tiles or len(stable_ids) != count):
        raise ValueError("actual tile costs require native shared-order batching")
    if operators is not None and len(stable_ids) == count:
        # All ROLs then share one sequence. Reduce exact per-record costs
        # directly, avoiding W copies of ranks, gathers, and scatter atomics.
        prefix = cost_prefix if cost_prefix is not None else operators.stable_batch_prefix(
            orders, xy, radii, tile_width, tile_height, shared_bytes, shared_tiles,
            orders.new_empty(0) if layout is None else layout,
        )
        # Search the identical greedy boundaries on-device; only two scalar
        # planning results cross to the host instead of the complete prefix.
        assignment, batches, maximum = operators.bounded_batches(prefix, budget, width)
        return assignment, batches, maximum + reserved
    else:
        lower = ((xy - radii).div(16).floor()).clamp_min(0)
        upper = ((xy + radii).div(16).ceil()).clamp_min(0)
        lower[..., 0].clamp_max_(tile_width)
        lower[..., 1].clamp_max_(tile_height)
        upper[..., 0].clamp_max_(tile_width)
        upper[..., 1].clamp_max_(tile_height)
        visible = (radii > 0).all(-1)
        if shared_tiles:
            union_lower = torch.where(visible[..., None], lower,
                                      lower.new_tensor((tile_width, tile_height))).amin(0)
            union_upper = torch.where(visible[..., None], upper, 0).amax(0)
            tile_bytes = (union_upper - union_lower).prod(-1).to(torch.int64) * 4
            # Nine FP32 fields in every frame, plus one shared inverse ID.
            attributes = width * 36 if layout is None else (layout[2, 1:] - layout[2, :-1]) * 4
            costs = torch.where(visible.any(0), attributes + shared_bytes + 4 + tile_bytes, 0)
            costs = costs[orders[0]]
            bucket = torch.arange(count, device=device)
        else:
            intersections = ((upper - lower).prod(-1).to(torch.int64) * visible)
            intersections = intersections.gather(1, orders)
            # Radix scratch and keys die before composition. Only int32 tile
            # IDs remain alongside the frame attributes and local orders.
            local_bytes = 8 + 8 + 4 + 12 + 12 + 4 + 4
            occurrence_bytes = local_bytes * visible.gather(1, orders) + intersections * 4
            if len(stable_ids) == 0:
                # No stable key can delimit a shared interval. Split the
                # independent ROLs at rank boundaries, charging every frame's
                # source attributes separately and retaining each full order.
                bucket = torch.arange(count, device=device)
                shared = shared_bytes * visible.gather(1, orders)
                costs = (occurrence_bytes + shared).sum(0)
            else:
                rank_by_id = torch.full((count,), -1, device=device, dtype=torch.int64)
                rank_by_id[stable_ids.long()] = torch.arange(len(stable_ids), device=device)
                rank = rank_by_id[orders]
                bucket = rank.cummax(dim=1).values.clamp_min(0)
                shared = torch.where((rank < 0) & visible.gather(1, orders), shared_bytes, 0)
                shared[0] = shared[0] + torch.where(
                    (rank[0] >= 0) & visible.any(0)[orders[0]], shared_bytes, 0)
                costs = torch.zeros(len(stable_ids), device=device, dtype=torch.int64)
                costs.scatter_add_(0, bucket.flatten(), (occurrence_bytes + shared).flatten())
        prefix = torch.cat((costs.new_zeros(1), costs.cumsum(0))).cpu().numpy()
    from bisect import bisect_right
    boundaries = [0]
    maximum = 0
    interval_count = len(prefix) - 1
    while boundaries[-1] < interval_count:
        start = boundaries[-1]
        end = bisect_right(prefix, prefix[start] + budget, lo=start + 1) - 1
        if end <= start:
            raise ValueError("one render interval exceeds L2; reduce the window")
        maximum = max(maximum, prefix[end] - prefix[start])
        boundaries.append(end)
    limits = torch.tensor(boundaries[1:], device=device, dtype=torch.int64)
    assignment = torch.searchsorted(limits, bucket, right=True)
    if assignment.ndim == 1:
        assignment = assignment[None].expand(width, -1)
    return assignment.contiguous(), len(boundaries) - 1, int(maximum) + reserved
