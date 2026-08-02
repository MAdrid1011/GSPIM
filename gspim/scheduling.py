"""Hybrid block, batch, and cross-window dependency construction."""

from __future__ import annotations

from typing import Generic, TypeVar


_BufferItem = TypeVar("_BufferItem")


class PingPongBatchBuffers(Generic[_BufferItem]):
    """Two fixed buffers with explicit producer and renderer ownership."""

    def __init__(self) -> None:
        self._slots: list[_BufferItem | None] = [None, None]
        self._next_pack = 0

    @property
    def next_slot(self) -> int:
        """Slot reserved for the next producer without mutating ownership."""

        return self._next_pack

    def pack(self, batch: _BufferItem) -> int:
        slot = self._next_pack
        if self._slots[slot] is not None:
            raise RuntimeError("next Batch Buffer is still occupied")
        self._slots[slot] = batch
        self._next_pack = 1 - slot
        return slot

    def consume(self, slot: int) -> _BufferItem:
        batch = self._slots[slot]
        if batch is None:
            raise RuntimeError("renderer consumed an empty Batch Buffer")
        self._slots[slot] = None
        return batch
