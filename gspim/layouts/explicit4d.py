"""Explicit 4D Gaussian layout with S1 filtering and S2 conditional slicing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ..model import DecisionRecord, FrameRequest, Geometry, Matrix3, Matrix4, RecordLocation, Selection, require_unique_physical_locations
from ..ppim import PPIMInterpreter, ProgramId
from .base import require_window


@dataclass(frozen=True)
class Explicit4DRecord:
    primitive_id: int
    temporal_mean: float
    temporal_rotation: Matrix4
    temporal_scale: tuple[float, float, float, float]
    mean: tuple[float, float, float]
    opacity: float
    color: tuple[float, float, float]
    attribute_bytes: int
    location: RecordLocation


class Explicit4DLayout:
    """S1 uses TEMP_ACTIVITY, while S2 emits full 3D Gaussian fields."""

    name = "explicit-4d"

    def __init__(self, records: Sequence[Explicit4DRecord]) -> None:
        self._records = {record.primitive_id: record for record in records}
        if not self._records or len(self._records) != len(records):
            raise ValueError("explicit-4D primitive IDs must be nonempty and unique")
        require_unique_physical_locations(
            tuple(record.location for record in records),
            "explicit-4D layout",
        )

    @classmethod
    def from_mappings(cls, records: Sequence[Mapping[str, object]]) -> "Explicit4DLayout":
        required = ("primitive_id", "temporal_mean", "temporal_rotation", "temporal_scale", "mean", "opacity", "color", "attribute_bytes", "die", "bank", "block", "slot")
        converted: list[Explicit4DRecord] = []
        for value in records:
            legacy = {"velocity", "covariance"} & set(value)
            if legacy:
                raise ValueError(f"explicit-4D uses conditional 4D slicing, not legacy fields: {', '.join(sorted(legacy))}")
            missing = [key for key in required if key not in value]
            if missing:
                raise ValueError(f"explicit-4D record is missing fields: {', '.join(missing)}")
            rotation = tuple(tuple(float(item) for item in row) for row in value["temporal_rotation"])  # type: ignore[arg-type]
            scale = tuple(float(item) for item in value["temporal_scale"])  # type: ignore[arg-type]
            mean = tuple(float(item) for item in value["mean"])  # type: ignore[arg-type]
            color = tuple(float(item) for item in value["color"])  # type: ignore[arg-type]
            if len(rotation) != 4 or any(len(row) != 4 for row in rotation) or len(scale) != 4 or len(mean) != 3 or len(color) != 3 or any(component <= 0.0 for component in scale):
                raise ValueError("explicit-4D matrix dimensions are invalid")
            primitive_id = int(value["primitive_id"])
            converted.append(Explicit4DRecord(
                primitive_id, float(value["temporal_mean"]), rotation, scale,  # type: ignore[arg-type]
                mean, float(value["opacity"]), color,
                int(value["attribute_bytes"]), RecordLocation(primitive_id, int(value["die"]), int(value["bank"]), int(value["block"]), int(value["slot"])),
            ))
        return cls(converted)

    def decision_records(self) -> tuple[DecisionRecord, ...]:
        return tuple(DecisionRecord(record.primitive_id, {"temporal_mean": record.temporal_mean, "temporal_rotation": record.temporal_rotation, "temporal_scale": record.temporal_scale}, record.location) for record in sorted(self._records.values(), key=lambda item: item.primitive_id))

    def select_window(self, window_start: float, window_end: float) -> Selection:
        start, end = require_window(window_start, window_end)
        interpreter = PPIMInterpreter()
        active = {record.record_id for record in self.decision_records() if interpreter.run(ProgramId.TEMP_ACTIVITY, record.fields, start, end).mask}
        return Selection(active, ProgramId.TEMP_ACTIVITY.value)

    def validate_window_frames(self, frames: Sequence[FrameRequest], selection: Selection) -> None:
        """Explicit conditional slicing defines geometry at every requested time."""

        unknown = selection.active_ids - set(self._records)
        if unknown:
            raise ValueError(f"explicit-4D selection contains unknown records: {sorted(unknown)}")

    def binding_records(self, record_id: int) -> tuple[int, ...]:
        if record_id not in self._records:
            raise ValueError("unknown explicit-4D record")
        return (record_id,)

    def record_location(self, primitive_id: int) -> RecordLocation:
        return self._records[primitive_id].location

    def payload(self, primitive_id: int) -> Mapping[str, object]:
        record = self._records[primitive_id]
        return {
            "primitive_id": primitive_id,
            "mean": record.mean,
            "temporal_mean": record.temporal_mean,
            "temporal_rotation": record.temporal_rotation,
            "temporal_scale": record.temporal_scale,
            "opacity": record.opacity,
            "color": record.color,
            "attribute_bytes": record.attribute_bytes,
        }

    @staticmethod
    def _covariance(record: Explicit4DRecord) -> tuple[tuple[float, float, float, float], ...]:
        """Construct R diag(scale^2) R^T from the native explicit 4D fields."""

        return tuple(
            tuple(
                sum(record.temporal_rotation[row][axis] * record.temporal_scale[axis] ** 2 * record.temporal_rotation[column][axis] for axis in range(4))
                for column in range(4)
            )
            for row in range(4)
        )

    @classmethod
    def _slice(cls, record: Explicit4DRecord, timestamp: float) -> tuple[tuple[float, float, float], Matrix3]:
        """Condition the native 4D Gaussian on one timestamp for S2."""

        covariance = cls._covariance(record)
        sigma_tt = covariance[3][3]
        if sigma_tt <= 0.0:
            raise ValueError("explicit-4D temporal variance must be positive")
        delta = float(timestamp) - record.temporal_mean
        spatial_temporal = tuple(covariance[axis][3] for axis in range(3))
        mean = tuple(record.mean[axis] + spatial_temporal[axis] * delta / sigma_tt for axis in range(3))
        conditional = tuple(
            tuple(covariance[row][column] - spatial_temporal[row] * spatial_temporal[column] / sigma_tt for column in range(3))
            for row in range(3)
        )
        return mean, conditional  # type: ignore[return-value]

    def generate_geometry_from_payloads(
        self,
        payloads: Mapping[int, Mapping[str, object]],
        timestamp: float,
    ) -> Mapping[int, Geometry]:
        """Generate S2 geometry from HCF's Active Buffer, never the source layout."""

        geometry: dict[int, Geometry] = {}
        required = ("mean", "temporal_mean", "temporal_rotation", "temporal_scale", "opacity", "color", "attribute_bytes")
        for primitive_id, payload in sorted(payloads.items()):
            missing = [key for key in required if key not in payload]
            if missing:
                raise ValueError(f"Active Buffer payload {primitive_id} is missing fields: {', '.join(missing)}")
            rotation = tuple(tuple(float(item) for item in row) for row in payload["temporal_rotation"])  # type: ignore[arg-type]
            scale = tuple(float(item) for item in payload["temporal_scale"])  # type: ignore[arg-type]
            mean = tuple(float(item) for item in payload["mean"])  # type: ignore[arg-type]
            color = tuple(float(item) for item in payload["color"])  # type: ignore[arg-type]
            if len(rotation) != 4 or any(len(row) != 4 for row in rotation) or len(scale) != 4 or len(mean) != 3 or len(color) != 3:
                raise ValueError(f"Active Buffer payload {primitive_id} has invalid explicit-4D fields")
            record = Explicit4DRecord(
                primitive_id,
                float(payload["temporal_mean"]),
                rotation,  # type: ignore[arg-type]
                scale,  # type: ignore[arg-type]
                mean,  # type: ignore[arg-type]
                float(payload["opacity"]),
                color,  # type: ignore[arg-type]
                int(payload["attribute_bytes"]),
                self._records[primitive_id].location,
            )
            sliced_mean, covariance = self._slice(record, timestamp)
            geometry[primitive_id] = Geometry(primitive_id, sliced_mean, covariance, record.opacity, record.color, record.attribute_bytes)
        return geometry
