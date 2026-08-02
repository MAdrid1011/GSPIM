"""Typed Ex4DGS keyframe-support and interpolation layout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ..model import DecisionRecord, FrameRequest, Geometry, Matrix3, RecordLocation, Selection, require_unique_physical_locations
from ..ppim import PPIMInterpreter, ProgramId
from .base import require_window


@dataclass(frozen=True)
class ExKeyframe:
    timestamp: float
    mean: tuple[float, float, float]
    covariance: Matrix3
    opacity: float
    color: tuple[float, float, float]


@dataclass(frozen=True)
class ExRecord:
    primitive_id: int
    is_static: bool
    keyframes: tuple[ExKeyframe, ...]
    attribute_bytes: int
    location: RecordLocation


class Ex4DGSLayout:
    """S1 checks keyframe support and S2 linearly evaluates its typed keys."""

    name = "ex4dgs"

    def __init__(self, records: Sequence[ExRecord]) -> None:
        self._records = {record.primitive_id: record for record in records}
        if not self._records or len(self._records) != len(records):
            raise ValueError("Ex4DGS primitive IDs must be nonempty and unique")
        require_unique_physical_locations(
            tuple(record.location for record in records),
            "Ex4DGS layout",
        )

    @classmethod
    def from_mappings(cls, records: Sequence[Mapping[str, object]]) -> "Ex4DGSLayout":
        required = ("primitive_id", "is_static", "keyframes", "attribute_bytes", "die", "bank", "block", "slot")
        converted: list[ExRecord] = []
        for value in records:
            legacy = {"covariance", "opacity", "color"} & set(value)
            if legacy:
                raise ValueError(f"Ex4DGS rendering fields belong to keyframes: {', '.join(sorted(legacy))}")
            missing = [key for key in required if key not in value]
            if missing:
                raise ValueError(f"Ex4DGS record is missing fields: {', '.join(missing)}")
            frames: list[ExKeyframe] = []
            for frame in value["keyframes"]:  # type: ignore[union-attr]
                if not isinstance(frame, Mapping) or not {"timestamp", "mean", "covariance", "opacity", "color"}.issubset(frame):
                    raise ValueError("each Ex4DGS keyframe needs timestamp, mean, covariance, opacity, and color")
                mean = tuple(float(item) for item in frame["mean"])  # type: ignore[arg-type]
                covariance = tuple(tuple(float(item) for item in row) for row in frame["covariance"])  # type: ignore[arg-type]
                color = tuple(float(item) for item in frame["color"])  # type: ignore[arg-type]
                if len(mean) != 3 or len(covariance) != 3 or any(len(row) != 3 for row in covariance) or len(color) != 3:
                    raise ValueError("Ex4DGS keyframe geometry or appearance is invalid")
                frames.append(ExKeyframe(float(frame["timestamp"]), mean, covariance, float(frame["opacity"]), color))
            if not frames or any(right.timestamp <= left.timestamp for left, right in zip(frames, frames[1:])):
                raise ValueError("Ex4DGS keyframes or covariance are invalid")
            primitive_id = int(value["primitive_id"])
            converted.append(ExRecord(
                primitive_id,
                bool(value["is_static"]),
                tuple(frames),
                int(value["attribute_bytes"]),
                RecordLocation(primitive_id, int(value["die"]), int(value["bank"]), int(value["block"]), int(value["slot"])),
            ))
        return cls(converted)

    def decision_records(self) -> tuple[DecisionRecord, ...]:
        records: list[DecisionRecord] = []
        for record in sorted(self._records.values(), key=lambda item: item.primitive_id):
            fields: Mapping[str, object] = {"is_static": record.is_static, "start": record.keyframes[0].timestamp, "end": record.keyframes[-1].timestamp}
            records.append(DecisionRecord(record.primitive_id, fields, self.record_location(record.primitive_id)))
        return tuple(records)

    def select_window(self, window_start: float, window_end: float) -> Selection:
        start, end = require_window(window_start, window_end)
        interpreter = PPIMInterpreter()
        active = {record.record_id for record in self.decision_records() if interpreter.run(ProgramId.KEYFRAME_RANGE, record.fields, start, end).mask}
        return Selection(active, ProgramId.KEYFRAME_RANGE.value)

    def validate_window_frames(self, frames: Sequence[FrameRequest], selection: Selection) -> None:
        """Reject an active dynamic record that cannot supply every S2 timestamp.

        KEYFRAME_RANGE intentionally selects on support/window intersection, as
        PPIM only sees local temporal fields.  S2 subsequently generates every
        selected primitive for every frame in that window, so this model-load
        precondition must be checked before any S1 task or HCF allocation.
        """

        unknown = selection.active_ids - set(self._records)
        if unknown:
            raise ValueError(f"Ex4DGS selection contains unknown records: {sorted(unknown)}")
        for primitive_id in sorted(selection.active_ids):
            record = self._records[primitive_id]
            if record.is_static:
                continue
            start, end = record.keyframes[0].timestamp, record.keyframes[-1].timestamp
            unsupported = tuple(frame.timestamp for frame in frames if frame.timestamp < start or frame.timestamp > end)
            if unsupported:
                raise ValueError(
                    f"dynamic Ex4DGS record {primitive_id} does not support every requested S2 frame: {unsupported}"
                )

    def binding_records(self, record_id: int) -> tuple[int, ...]:
        if record_id not in self._records:
            raise ValueError("unknown Ex4DGS record")
        return (record_id,)

    def record_location(self, primitive_id: int) -> RecordLocation:
        return self._records[primitive_id].location

    def payload(self, primitive_id: int) -> Mapping[str, object]:
        record = self._records[primitive_id]
        return {"primitive_id": primitive_id, "keyframes": record.keyframes, "attribute_bytes": record.attribute_bytes}

    def _frame(self, record: ExRecord, timestamp: float) -> ExKeyframe:
        if record.is_static:
            return record.keyframes[0]
        if timestamp < record.keyframes[0].timestamp or timestamp > record.keyframes[-1].timestamp:
            raise ValueError("dynamic Ex4DGS geometry is outside its declared keyframe support")
        if timestamp == record.keyframes[0].timestamp:
            return record.keyframes[0]
        if timestamp == record.keyframes[-1].timestamp:
            return record.keyframes[-1]
        for left, right in zip(record.keyframes, record.keyframes[1:]):
            if left.timestamp <= timestamp <= right.timestamp:
                ratio = (timestamp - left.timestamp) / (right.timestamp - left.timestamp)
                mean = tuple((1.0 - ratio) * a + ratio * b for a, b in zip(left.mean, right.mean))
                covariance = tuple(tuple((1.0 - ratio) * left.covariance[row][column] + ratio * right.covariance[row][column] for column in range(3)) for row in range(3))
                color = tuple((1.0 - ratio) * a + ratio * b for a, b in zip(left.color, right.color))
                return ExKeyframe(float(timestamp), mean, covariance, (1.0 - ratio) * left.opacity + ratio * right.opacity, color)  # type: ignore[arg-type]
        raise AssertionError("timestamp must be bracketed")

    def generate_geometry_from_payloads(
        self,
        payloads: Mapping[int, Mapping[str, object]],
        timestamp: float,
    ) -> Mapping[int, Geometry]:
        """Generate S2 geometry from the HCF-packed keyframe payloads."""

        output: dict[int, Geometry] = {}
        for primitive_id, payload in sorted(payloads.items()):
            if "keyframes" not in payload or "attribute_bytes" not in payload:
                raise ValueError(f"Active Buffer payload {primitive_id} is missing Ex4DGS fields")
            keys = payload["keyframes"]
            if not isinstance(keys, tuple) or not keys or not all(isinstance(item, ExKeyframe) for item in keys):
                raise ValueError(f"Active Buffer payload {primitive_id} has invalid Ex4DGS keyframes")
            record = ExRecord(
                primitive_id,
                self._records[primitive_id].is_static,
                keys,
                int(payload["attribute_bytes"]),
                self._records[primitive_id].location,
            )
            frame = self._frame(record, timestamp)
            output[primitive_id] = Geometry(primitive_id, frame.mean, frame.covariance, frame.opacity, frame.color, record.attribute_bytes)
        return output
