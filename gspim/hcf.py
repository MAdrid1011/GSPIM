"""Functional HCF with FIFO mask intake, aligned gathers, and live ranges."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Mapping


@dataclass(frozen=True)
class RecordLocation:
    primitive_id: int
    die: int
    bank: int
    block: int = 0
    slot: int = 0


@dataclass(frozen=True)
class HCFRecord:
    primitive_id: int
    location: RecordLocation
    payload: object
    # An anchor is itself the PPIM-selected source record.  HCF, rather than
    # host code, expands its validated binding-table entries into payloads.
    bound_records: tuple["HCFRecord", ...] = ()


class ReorgPurpose(str, Enum):
    ACTIVE = "active"
    STABLE = "stable"
    UNSTABLE = "unstable"
    BATCH = "batch"


@dataclass(frozen=True)
class LiveRange:
    region: str
    start: int
    count: int


@dataclass(frozen=True)
class BlockCompletion:
    die: int
    bank: int
    block: int
    output_range: LiveRange
    output_ids: tuple[int, ...]
    input_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class HCFResult:
    purpose: ReorgPurpose
    output_ids: tuple[int, ...]
    packed_records: tuple[HCFRecord, ...]
    die_ranges: Mapping[int, LiveRange]
    block_completions: tuple[BlockCompletion, ...]
    overflow_ids: tuple[int, ...]
    allocation_id: str | None = None
    gpu_fallbacks: tuple["GPUFallback", ...] = ()


@dataclass(frozen=True)
class GPUFallback:
    """One original physical block handed to the explicit GPU fallback path."""

    die: int
    bank: int
    block: int
    source_records: tuple[HCFRecord, ...]
    overflow_ids: tuple[int, ...]


class ActiveMapFIFO:
    """A visible FIFO of bank-block masks consumed by the die compactor."""

    def __init__(self) -> None:
        self._entries: deque[tuple[int, int, int, tuple[int, ...]]] = deque()

    def push(self, die: int, bank: int, block: int, selected_ids: tuple[int, ...]) -> None:
        if selected_ids:
            self._entries.append((die, bank, block, selected_ids))

    def pop(self) -> tuple[int, int, int, tuple[int, ...]]:
        return self._entries.popleft()

    def __bool__(self) -> bool:
        return bool(self._entries)


class HierarchicalCompactionFabric:
    """L1 selection and L2 gather/writeback semantics without a timing model."""

    def __init__(self, reserved_capacity_per_die: int) -> None:
        if reserved_capacity_per_die < 1:
            raise ValueError("reserved_capacity_per_die must be positive")
        self.reserved_capacity_per_die = reserved_capacity_per_die
        self._free_ranges: dict[int, list[tuple[int, int]]] = {}
        self._allocations: dict[str, Mapping[int, LiveRange]] = {}
        self._next_allocation = 0

    def _free_for_die(self, die: int) -> list[tuple[int, int]]:
        return self._free_ranges.setdefault(die, [(0, self.reserved_capacity_per_die)])

    def _reserve(self, die: int, count: int) -> tuple[int, int]:
        """Reserve one contiguous interval from the shared reserved-bank space."""

        if count < 1:
            return 0, 0
        free = self._free_for_die(die)
        if not free:
            return 0, 0
        index, (start, available) = max(enumerate(free), key=lambda item: item[1][1])
        granted = min(count, available)
        if granted:
            if granted == available:
                free.pop(index)
            else:
                free[index] = (start + granted, available - granted)
        return start, granted

    def release(self, result: HCFResult) -> None:
        """Return a completed consumer's HCF range to its die's shared workspace."""

        if result.allocation_id is None:
            return
        ranges = self._allocations.pop(result.allocation_id, None)
        if ranges is None:
            raise ValueError(f"HCF allocation {result.allocation_id} was already released or is unknown")
        for die, live_range in ranges.items():
            free = self._free_for_die(die)
            free.append((live_range.start, live_range.count))
            free.sort()
            merged: list[tuple[int, int]] = []
            for start, count in free:
                if merged and merged[-1][0] + merged[-1][1] == start:
                    previous_start, previous_count = merged[-1]
                    merged[-1] = (previous_start, previous_count + count)
                else:
                    merged.append((start, count))
            self._free_ranges[die] = merged

    def reorganize(self, records: Mapping[int, HCFRecord], selected_ids: set[int], purpose: ReorgPurpose) -> HCFResult:
        unknown = selected_ids - set(records)
        if unknown:
            raise ValueError(f"HCF selection refers to missing payloads: {sorted(unknown)}")
        fifo = ActiveMapFIFO()
        by_block: dict[tuple[int, int, int], list[HCFRecord]] = {}
        for primitive_id in sorted(selected_ids):
            record = records[primitive_id]
            location = record.location
            by_block.setdefault((location.die, location.bank, location.block), []).append(record)
        for (die, bank, block), block_records in sorted(by_block.items()):
            ordered = tuple(record.primitive_id for record in sorted(block_records, key=lambda item: (item.location.slot, item.primitive_id)))
            fifo.push(die, bank, block, ordered)

        queued_blocks: list[tuple[int, int, int, tuple[HCFRecord, ...]]] = []
        requested_by_die: dict[int, int] = {}
        while fifo:
            die, bank, block, ids = fifo.pop()
            sources = tuple(records[primitive_id] for primitive_id in ids)
            block_records = tuple(
                bound
                for source in sources
                for bound in (source.bound_records or (source,))
            )
            if len({record.primitive_id for record in block_records}) != len(block_records):
                raise ValueError("HCF binding expansion produced duplicate primitive payloads")
            queued_blocks.append((die, bank, block, block_records))
            requested_by_die[die] = requested_by_die.get(die, 0) + len(block_records)
        reservations = {die: self._reserve(die, count) for die, count in requested_by_die.items()}
        accepted_by_die: dict[int, list[HCFRecord]] = {}
        overflow: list[int] = []
        fallbacks: list[GPUFallback] = []
        block_inputs: list[tuple[int, int, int, tuple[HCFRecord, ...]]] = []
        for die, bank, block, block_records in queued_blocks:
            available = reservations[die][1] - len(accepted_by_die.get(die, []))
            accepted = block_records[:max(0, available)]
            rejected = block_records[len(accepted):]
            if accepted:
                accepted_by_die.setdefault(die, []).extend(accepted)
            overflow.extend(record.primitive_id for record in rejected)
            if rejected:
                fallbacks.append(GPUFallback(
                    die,
                    bank,
                    block,
                    block_records,
                    tuple(record.primitive_id for record in rejected),
                ))
            block_inputs.append((die, bank, block, accepted))

        ranges: dict[int, LiveRange] = {}
        for die in sorted(accepted_by_die):
            count = len(accepted_by_die[die])
            ranges[die] = LiveRange(purpose.value, reservations[die][0], count)
        completions: list[BlockCompletion] = []
        positions = {die: 0 for die in accepted_by_die}
        for (die, bank, block, input_records), (_, _, _, accepted) in zip(queued_blocks, block_inputs):
            reservation_start = reservations[die][0]
            output_range = LiveRange(
                purpose.value,
                reservation_start + positions.get(die, 0),
                len(accepted),
            )
            positions[die] = positions.get(die, 0) + len(accepted)
            completions.append(BlockCompletion(
                die,
                bank,
                block,
                output_range,
                tuple(record.primitive_id for record in accepted),
                tuple(record.primitive_id for record in input_records),
            ))
        packed = tuple(record for die in sorted(accepted_by_die) for record in accepted_by_die[die])
        allocation_id = None
        if ranges:
            allocation_id = f"{purpose.value}:{self._next_allocation}"
            self._next_allocation += 1
            self._allocations[allocation_id] = dict(ranges)
        return HCFResult(
            purpose,
            tuple(record.primitive_id for record in packed),
            packed,
            ranges,
            tuple(completions),
            tuple(overflow),
            allocation_id,
            tuple(fallbacks),
        )
