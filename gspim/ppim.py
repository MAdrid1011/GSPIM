"""Executable PPIM programs with the paper's fixed-point decision rules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import floor, log
from typing import Mapping


class ProgramId(str, Enum):
    TEMP_ACTIVITY = "TEMP_ACTIVITY"
    KEYFRAME_RANGE = "KEYFRAME_RANGE"
    ANCHOR_OVERLAP = "ANCHOR_OVERLAP"
    DEPTH_STABLE = "DEPTH_STABLE"


class PPIMOperation(str, Enum):
    ROT_SLICE = "ROT_SLICE"
    SCALE = "SCALE"
    DOT = "DOT"
    WINDOW_DIST = "WINDOW_DIST"
    CMP_LE = "CMP_LE"
    LOAD = "LOAD"
    CMP_LT = "CMP_LT"
    AND = "AND"
    CMP_GT = "CMP_GT"


@dataclass(frozen=True)
class PPIMProgram:
    program_id: ProgramId
    operations: tuple[PPIMOperation, ...]


PROGRAMS = {
    ProgramId.TEMP_ACTIVITY: PPIMProgram(ProgramId.TEMP_ACTIVITY, (PPIMOperation.ROT_SLICE, PPIMOperation.SCALE, PPIMOperation.DOT, PPIMOperation.WINDOW_DIST, PPIMOperation.CMP_LE)),
    ProgramId.KEYFRAME_RANGE: PPIMProgram(ProgramId.KEYFRAME_RANGE, (PPIMOperation.LOAD, PPIMOperation.CMP_LT, PPIMOperation.CMP_LT, PPIMOperation.AND)),
    ProgramId.ANCHOR_OVERLAP: PPIMProgram(ProgramId.ANCHOR_OVERLAP, (PPIMOperation.LOAD, PPIMOperation.CMP_LT, PPIMOperation.CMP_LT, PPIMOperation.AND)),
    ProgramId.DEPTH_STABLE: PPIMProgram(ProgramId.DEPTH_STABLE, (PPIMOperation.CMP_GT,)),
}


@dataclass(frozen=True)
class FixedPoint26:
    """Signed 26-bit operand with a rounded 52-bit multiplication result."""

    raw: int
    fractional_bits: int = 16
    bits: int = 26

    @classmethod
    def from_float(cls, value: float, fractional_bits: int = 16) -> "FixedPoint26":
        scaled = float(value) * (1 << fractional_bits)
        rounded = floor(scaled + 0.5) if scaled >= 0.0 else -floor(-scaled + 0.5)
        return cls(cls._wrap(rounded), fractional_bits)

    @classmethod
    def from_raw(cls, raw: int, fractional_bits: int = 16) -> "FixedPoint26":
        return cls(cls._wrap(raw), fractional_bits)

    @staticmethod
    def _wrap(raw: int) -> int:
        """Match a 26-bit two's-complement register-file writeback."""

        bits = 26
        value = int(raw) & ((1 << bits) - 1)
        return value - (1 << bits) if value >= (1 << (bits - 1)) else value

    def to_float(self) -> float:
        return self.raw / float(1 << self.fractional_bits)

    def multiply(self, other: "FixedPoint26") -> "FixedPoint26":
        if self.fractional_bits != other.fractional_bits:
            raise ValueError("fixed-point operands need the same binary point")
        product = self.raw * other.raw
        rounding = 1 << (self.fractional_bits - 1)
        rounded = -((-product + rounding) >> self.fractional_bits) if product < 0 else (product + rounding) >> self.fractional_bits
        return FixedPoint26(self._wrap(rounded), self.fractional_bits)

    def add(self, other: "FixedPoint26") -> "FixedPoint26":
        if self.fractional_bits != other.fractional_bits:
            raise ValueError("fixed-point operands need the same binary point")
        return FixedPoint26(self._wrap(self.raw + other.raw), self.fractional_bits)


@dataclass(frozen=True)
class ProgramResult:
    mask: bool
    registers: Mapping[str, float | int | bool]


class PPIMInterpreter:
    """Program-level PPIM semantics used by the runtime and layout adapters."""

    def __init__(self, fractional_bits: int = 16) -> None:
        if not 1 <= fractional_bits < 26:
            raise ValueError("fractional_bits must fit in a signed 26-bit operand")
        self.fractional_bits = fractional_bits

    @property
    def lsb(self) -> float:
        return 1.0 / (1 << self.fractional_bits)

    def run(
        self,
        program: ProgramId,
        fields: Mapping[str, object],
        window_start: float | None = None,
        window_end: float | None = None,
        tau: float | None = None,
    ) -> ProgramResult:
        if program is ProgramId.TEMP_ACTIVITY:
            return self._temp_activity(fields, self._window(window_start, window_end))
        if program in (ProgramId.KEYFRAME_RANGE, ProgramId.ANCHOR_OVERLAP):
            return self._interval(program, fields, self._window(window_start, window_end))
        if program is ProgramId.DEPTH_STABLE:
            if tau is None or "score" not in fields:
                raise ValueError("DEPTH_STABLE requires score and tau")
            score = FixedPoint26.from_float(float(fields["score"]), self.fractional_bits)
            threshold = FixedPoint26.from_float(float(tau), self.fractional_bits)
            return ProgramResult(
                score.raw > threshold.raw + 1,
                {
                    "score": score.to_float(),
                    "score_raw": score.raw,
                    "threshold": threshold.to_float(),
                    "threshold_raw": threshold.raw,
                },
            )
        raise ValueError(f"unsupported PPIM program: {program}")

    @staticmethod
    def _window(start: float | None, end: float | None) -> tuple[float, float]:
        # The public pipeline rejects empty windows. The interpreter also
        # accepts an ordered zero-width probe for fixed-point microprogram
        # boundary tests.
        if start is None or end is None or end < start:
            raise ValueError("program requires an ordered temporal window")
        return float(start), float(end)

    def _temp_activity(self, fields: Mapping[str, object], window: tuple[float, float]) -> ProgramResult:
        required = ("temporal_mean", "temporal_rotation", "temporal_scale")
        if any(key not in fields for key in required):
            raise ValueError("TEMP_ACTIVITY requires temporal_mean, temporal_rotation, and temporal_scale")
        rotation = tuple(tuple(float(item) for item in row) for row in fields["temporal_rotation"])  # type: ignore[arg-type]
        scale = tuple(float(item) for item in fields["temporal_scale"])  # type: ignore[arg-type]
        if len(rotation) != 4 or any(len(row) != 4 for row in rotation) or len(scale) != 4:
            raise ValueError("TEMP_ACTIVITY requires a 4x4 rotation and four scales")
        # T1 is the scaled 4D rotation factor.  Its temporal row dotted with
        # itself is Sigma_tt of R diag(s^2) R^T, the value used by both S1 and
        # the S2 conditional slice.  Products round through the documented
        # 26-bit/52-bit register-file boundary before the accumulation.
        temporal_row = rotation[3]
        scaled_row = tuple(
            FixedPoint26.from_float(component, self.fractional_bits)
            .multiply(FixedPoint26.from_float(scale[index], self.fractional_bits))
            for index, component in enumerate(temporal_row)
        )
        sigma_tt = FixedPoint26.from_raw(0, self.fractional_bits)
        for component in scaled_row:
            sigma_tt = sigma_tt.add(component.multiply(component))
        mean = FixedPoint26.from_float(float(fields["temporal_mean"]), self.fractional_bits)
        start, end = (FixedPoint26.from_float(value, self.fractional_bits) for value in window)
        distance_raw = max(start.raw - mean.raw, 0, mean.raw - end.raw)
        distance = FixedPoint26.from_raw(distance_raw, self.fractional_bits)
        lhs_raw = distance.multiply(distance).raw >> 1
        rhs = sigma_tt.multiply(FixedPoint26.from_float(log(20.0), self.fractional_bits))
        return ProgramResult(
            lhs_raw <= rhs.raw + 1,
            {
                "sigma_tt": sigma_tt.to_float(),
                "sigma_tt_raw": sigma_tt.raw,
                "window_distance": distance.to_float(),
                "window_distance_raw": distance.raw,
                "lhs": lhs_raw / float(1 << self.fractional_bits),
                "lhs_raw": lhs_raw,
                "rhs": rhs.to_float(),
                "rhs_raw": rhs.raw,
            },
        )

    def _interval(self, program: ProgramId, fields: Mapping[str, object], window: tuple[float, float]) -> ProgramResult:
        if program is ProgramId.KEYFRAME_RANGE and bool(fields.get("is_static", False)):
            return ProgramResult(True, {"is_static": True})
        if "start" not in fields or "end" not in fields:
            raise ValueError(f"{program.value} requires start and end")
        start, end = float(fields["start"]), float(fields["end"])
        if end < start:
            raise ValueError("temporal support end precedes start")
        start_fixed = FixedPoint26.from_float(start, self.fractional_bits)
        end_fixed = FixedPoint26.from_float(end, self.fractional_bits)
        window_start, window_end = (FixedPoint26.from_float(value, self.fractional_bits) for value in window)
        # Support endpoints are closed while rendering windows are [start, end).
        # One-LSB retention applies to numerical PPIM thresholds, not to this
        # temporal-set intersection.
        left = start_fixed.raw < window_end.raw
        right = end_fixed.raw >= window_start.raw
        return ProgramResult(left and right, {"start": start_fixed.to_float(), "end": end_fixed.to_float(), "left": left, "right": right})
