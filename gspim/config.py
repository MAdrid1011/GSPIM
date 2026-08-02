"""Artifact configuration for topology, fixed point, buffering, and rendering."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ArchitectureConfig:
    packages: int = 8
    dies_per_package: int = 4
    banks_per_die: int = 16
    pim_banks_per_die: int = 12
    reserved_banks_per_die: int = 4
    l2_capacity_bytes: int = 4 * 1024 * 1024
    min_window_size: int = 2
    max_window_size: int = 5
    tau_stability: float = 0.98
    tau_window: float = 0.98
    fixed_fractional_bits: int = 16
    block_records: int = 16
    local_index_bytes: int = 4
    raster_intermediate_bytes: int = 16

    def __post_init__(self) -> None:
        if self.pim_banks_per_die + self.reserved_banks_per_die != self.banks_per_die:
            raise ValueError("PIM and reserved banks must cover each die")
        if not 2 <= self.min_window_size <= self.max_window_size <= 5:
            raise ValueError("window bounds must remain within the paper's 2--5 frame range")
        if self.l2_capacity_bytes < 1 or self.block_records < 1:
            raise ValueError("capacity and block_records must be positive")
        if self.local_index_bytes < 1 or self.raster_intermediate_bytes < 1:
            raise ValueError("batch accounting parameters must be positive")

    @property
    def total_dies(self) -> int:
        return self.packages * self.dies_per_package

    @property
    def reserved_record_capacity_per_die(self) -> int:
        return self.reserved_banks_per_die * self.block_records
