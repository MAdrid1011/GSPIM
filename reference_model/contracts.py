"""Independent types used by the reference oracle only."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExpectedWindow:
    active_ids: set[int]
    stable_ids: set[int]
    unstable_ids: set[int]
    rols: tuple[tuple[int, ...], ...]
    batch_bytes: tuple[int, ...]
    checksums: tuple[str, ...]
