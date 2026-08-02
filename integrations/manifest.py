"""Pinned provenance for optional external workload code."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Upstream:
    name: str
    url: str
    commit: str


EX4DGS = Upstream(
    "Ex4DGS",
    "https://github.com/juno181/Ex4DGS.git",
    "7fac64997164fde64732c80261e56adf66ca14df",
)
