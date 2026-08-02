"""Task descriptors, visibility protocol, and real completion metadata."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from .hcf import BlockCompletion, LiveRange, ReorgPurpose
from .ppim import ProgramId
from .trace import TaskTrace


class PIMTaskKind(str, Enum):
    SELECT = "PIM_SELECT"
    REORG = "PIM_REORG"


@dataclass(frozen=True)
class MemoryRegion:
    name: str


@dataclass(frozen=True)
class PIMTaskDescriptor:
    task_id: str
    kind: PIMTaskKind
    die: int
    source: MemoryRegion
    destination: MemoryRegion
    program: ProgramId | None
    blocks: tuple[int, ...]
    purpose: ReorgPurpose
    # A die-level command may cover several PIM banks.  Keeping the bank scope
    # explicit prevents a block number from being mistaken for a physical block.
    banks: tuple[int, ...] = ()
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskOutcome:
    output_count: int
    ranges: tuple[LiveRange, ...]
    block_completions: tuple[BlockCompletion, ...]
    overflow_ids: tuple[int, ...]


@dataclass(frozen=True)
class CompletionEvent:
    task_id: str
    die: int
    complete: bool
    output_count: int = 0
    ranges: tuple[LiveRange, ...] = ()
    block_completions: tuple[BlockCompletion, ...] = ()
    overflow_ids: tuple[int, ...] = ()


class VisibilityError(RuntimeError):
    pass


class Runtime:
    """Functional command protocol with no elapsed-time or throughput model."""

    def __init__(self, dies: int = 32) -> None:
        if dies < 1:
            raise ValueError("at least one die is required")
        self.dies = dies
        self._visibility: dict[str, str] = {}
        self._tasks: dict[str, PIMTaskDescriptor] = {}
        self._events: dict[str, CompletionEvent] = {}
        self._live_at_completion: dict[str, tuple[str, ...]] = {}
        self._die_order: dict[int, list[str]] = {die: [] for die in range(dies)}
        self.trace = TaskTrace()

    def gpu_write(self, region: MemoryRegion) -> None:
        self._visibility[region.name] = "gpu_dirty"

    def writeback(self, region: MemoryRegion) -> None:
        state = self._visibility.get(region.name)
        if state in (None, "gpu_dirty", "gpu_readable"):
            self._visibility[region.name] = "memory_visible"

    def invalidate(self, region: MemoryRegion) -> None:
        if self._visibility.get(region.name) != "pim_written":
            raise VisibilityError(f"cannot invalidate {region.name} before PIM writes it")
        self._visibility[region.name] = "invalidated"

    def submit(self, descriptor: PIMTaskDescriptor) -> CompletionEvent:
        if not 0 <= descriptor.die < self.dies:
            raise ValueError("task die is out of range")
        if descriptor.task_id in self._tasks:
            raise ValueError(f"duplicate task ID: {descriptor.task_id}")
        if not descriptor.blocks:
            raise ValueError("task needs at least one block")
        if not descriptor.banks:
            raise ValueError("task needs at least one PIM-bank scope entry")
        if any(bank < 0 for bank in descriptor.banks):
            raise ValueError("task bank scope must be nonnegative")
        if descriptor.kind is PIMTaskKind.SELECT and descriptor.program is None:
            raise ValueError("PIM_SELECT requires a PPIM program")
        if descriptor.kind is PIMTaskKind.REORG and descriptor.program is not None:
            raise ValueError("PIM_REORG does not execute a PPIM program")
        if self._visibility.get(descriptor.source.name) == "gpu_dirty":
            raise VisibilityError(f"writeback required before {descriptor.kind.value} reads {descriptor.source.name}")
        if any(dependency not in self._events for dependency in descriptor.depends_on):
            raise ValueError("all task dependencies must be submitted first")
        # Command ordering is represented by the descriptor's explicit data
        # dependency.  A hidden die-wide predecessor would reintroduce the
        # window barrier that hybrid block scheduling is intended to remove.
        dependencies = tuple(dict.fromkeys(descriptor.depends_on))
        self._tasks[descriptor.task_id] = descriptor
        self._events[descriptor.task_id] = CompletionEvent(descriptor.task_id, descriptor.die, False)
        self._die_order[descriptor.die].append(descriptor.task_id)
        self.trace.add(descriptor.task_id, descriptor.kind.value, dependencies)
        return self._events[descriptor.task_id]

    def submit_gpu_stage(self, task_id: str, kind: str, dependencies: tuple[str, ...]) -> CompletionEvent:
        """Make a GPU stage live; completion is separate so overlap is observable."""

        if task_id in self._events or any(dependency not in self._events for dependency in dependencies):
            raise ValueError("GPU stage ID or dependency is invalid")
        self.trace.add(task_id, kind, dependencies)
        event = CompletionEvent(task_id, 0, False)
        self._events[task_id] = event
        return event

    def complete_gpu_stage(self, task_id: str, kind: str | None = None, dependencies: tuple[str, ...] | None = None) -> CompletionEvent:
        """Complete a live GPU stage after all of its submitted dependencies finish."""

        if task_id not in self._events:
            if kind is None or dependencies is None:
                raise ValueError("a new GPU stage needs kind and dependencies")
            self.submit_gpu_stage(task_id, kind, dependencies)
        event = self._events[task_id]
        node = self.trace.nodes[task_id]
        if not all(self._events[dependency].complete for dependency in node.dependencies):
            raise RuntimeError("GPU stage dependencies are incomplete")
        self._live_at_completion[task_id] = tuple(candidate for candidate, candidate_event in self._events.items() if candidate != task_id and not candidate_event.complete)
        complete = CompletionEvent(task_id, event.die, True)
        self._events[task_id] = complete
        return complete

    def complete(self, task_id: str, outcome: TaskOutcome) -> CompletionEvent:
        if task_id not in self._tasks:
            raise ValueError(f"unknown task: {task_id}")
        descriptor = self._tasks[task_id]
        dependencies = self.trace.nodes[task_id].dependencies
        if not all(self._events[dependency].complete for dependency in dependencies):
            raise RuntimeError("cannot complete a task before its dependencies")
        if outcome.output_count < 0:
            raise ValueError("output count must not be negative")
        self._live_at_completion[task_id] = tuple(candidate for candidate, candidate_event in self._events.items() if candidate != task_id and not candidate_event.complete)
        event = CompletionEvent(task_id, descriptor.die, True, outcome.output_count, outcome.ranges, outcome.block_completions, outcome.overflow_ids)
        self._events[task_id] = event
        self._visibility[descriptor.destination.name] = "pim_written"
        return event

    def complete_block(
        self,
        task_id: str,
        completion: BlockCompletion,
        overflow_ids: tuple[int, ...] = (),
    ) -> CompletionEvent:
        """Expose one HCF-produced physical block before its die task finishes.

        The host-visible PIM_REORG command retains its final completion and
        aggregate count.  This event is the paper's earlier block-range
        notification that releases the dependent GPU S2/S3 work.
        """

        if task_id not in self._tasks:
            raise ValueError(f"unknown task: {task_id}")
        descriptor = self._tasks[task_id]
        if descriptor.kind is not PIMTaskKind.REORG:
            raise ValueError("only PIM_REORG exposes HCF block completions")
        if self._events[task_id].complete:
            raise RuntimeError("cannot expose a block after final task completion")
        dependencies = self.trace.nodes[task_id].dependencies
        if not all(self._events[dependency].complete for dependency in dependencies):
            raise RuntimeError("cannot expose a block before reorganization dependencies")
        block_id = f"{task_id}:bank{completion.bank}-b{completion.block}"
        if block_id in self._events:
            raise ValueError(f"duplicate HCF block completion: {block_id}")
        self.trace.add(block_id, "PIM_REORG_BLOCK", dependencies)
        self._live_at_completion[block_id] = tuple(
            candidate
            for candidate, candidate_event in self._events.items()
            if candidate != block_id and not candidate_event.complete
        )
        event = CompletionEvent(
            block_id,
            descriptor.die,
            True,
            len(completion.output_ids),
            (completion.output_range,),
            (completion,),
            overflow_ids,
        )
        self._events[block_id] = event
        self._visibility[descriptor.destination.name] = "pim_written"
        return event

    def event(self, task_id: str) -> CompletionEvent:
        return self._events[task_id]

    @property
    def events(self) -> Mapping[str, CompletionEvent]:
        """Immutable-view completion evidence exposed by a completed window."""

        return dict(self._events)

    @property
    def live_at_completion(self) -> Mapping[str, tuple[str, ...]]:
        """Tasks live when each completion was emitted; used to prove overlap."""

        return dict(self._live_at_completion)

    def gpu_acquire(self, region: MemoryRegion, task_id: str) -> None:
        event = self.event(task_id)
        if not event.complete:
            raise VisibilityError(f"completion required before GPU reads {region.name}")
        if self._visibility.get(region.name) != "invalidated":
            raise VisibilityError(f"invalidate required before GPU reads {region.name}")
        self._visibility[region.name] = "gpu_readable"
