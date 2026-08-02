"""Anchored 4DGS interval selection and validated binding expansion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ..model import DecisionRecord, FrameRequest, Geometry, Matrix3, RecordLocation, Selection, require_unique_physical_locations
from ..ppim import PPIMInterpreter, ProgramId
from .base import require_window


@dataclass(frozen=True)
class AnchorRecord:
    anchor_id: int
    start: float
    end: float
    binding_start: int
    binding_count: int
    location: RecordLocation


@dataclass(frozen=True)
class AnchoredFrame:
    """One representation-generated 3D Gaussian at an explicit timestamp."""

    timestamp: float
    mean: tuple[float, float, float]
    covariance: Matrix3
    opacity: float
    color: tuple[float, float, float]


@dataclass(frozen=True)
class BoundPayload:
    primitive_id: int
    frames: tuple[AnchoredFrame, ...]
    attribute_bytes: int
    location: RecordLocation


class Anchored4DGSLayout:
    """S1 selects anchor intervals and S2 consumes explicit generated frames."""

    name = "anchored-4dgs"

    def __init__(self, anchors: Sequence[AnchorRecord], payloads: Sequence[BoundPayload]) -> None:
        self._anchors = {anchor.anchor_id: anchor for anchor in anchors}
        self._payloads = {payload.primitive_id: payload for payload in payloads}
        if not self._anchors or len(self._anchors) != len(anchors) or len(self._payloads) != len(payloads):
            raise ValueError("anchor and payload IDs must be nonempty and unique")
        require_unique_physical_locations(
            tuple(anchor.location for anchor in anchors),
            "Anchored 4DGS anchor layout",
        )
        require_unique_physical_locations(
            tuple(payload.location for payload in payloads),
            "Anchored 4DGS payload layout",
        )
        anchor_locations = {(anchor.location.die, anchor.location.bank, anchor.location.block, anchor.location.slot) for anchor in anchors}
        payload_locations = {(payload.location.die, payload.location.bank, payload.location.block, payload.location.slot) for payload in payloads}
        if anchor_locations & payload_locations:
            raise ValueError("Anchored 4DGS anchor and payload rows cannot share a physical mask location")
        missing = {
            primitive_id
            for anchor in anchors
            for primitive_id in range(anchor.binding_start, anchor.binding_start + anchor.binding_count)
        } - set(self._payloads)
        if missing:
            raise ValueError(f"anchor bindings have no payloads: {sorted(missing)}")
        for anchor in anchors:
            if anchor.binding_count < 1:
                raise ValueError("anchor binding_count must be positive")
            binding_ids = range(anchor.binding_start, anchor.binding_start + anchor.binding_count)
            if any(self._payloads[primitive_id].location.die != anchor.location.die for primitive_id in binding_ids):
                raise ValueError("anchor binding range must remain in the anchor's die")

    @classmethod
    def from_mappings(cls, anchors: Sequence[Mapping[str, object]], payloads: Mapping[int, Mapping[str, object]]) -> "Anchored4DGSLayout":
        converted_anchors: list[AnchorRecord] = []
        for value in anchors:
            required_anchor = {"anchor_id", "start", "end", "binding_start", "binding_count", "die", "bank", "block", "slot"}
            if not required_anchor.issubset(value):
                raise ValueError("anchor is missing id, interval, binding range, or location")
            start, end = float(value["start"]), float(value["end"])
            if end < start:
                raise ValueError("anchor interval is invalid")
            anchor_id = int(value["anchor_id"])
            converted_anchors.append(AnchorRecord(
                anchor_id,
                start,
                end,
                int(value["binding_start"]),
                int(value["binding_count"]),
                RecordLocation(anchor_id, int(value["die"]), int(value["bank"]), int(value["block"]), int(value["slot"])),
            ))
        converted_payloads: list[BoundPayload] = []
        required = ("frame_geometry", "attribute_bytes", "die", "bank", "block", "slot")
        for primitive_id, value in payloads.items():
            missing = [key for key in required if key not in value]
            if missing:
                raise ValueError(f"anchor payload {primitive_id} is missing fields: {', '.join(missing)}")
            frames: list[AnchoredFrame] = []
            for frame in value["frame_geometry"]:  # type: ignore[union-attr]
                if not isinstance(frame, Mapping) or not {"timestamp", "mean", "covariance", "opacity", "color"}.issubset(frame):
                    raise ValueError("each anchor frame needs timestamp, mean, covariance, opacity, and color")
                mean = tuple(float(item) for item in frame["mean"])  # type: ignore[arg-type]
                covariance = tuple(tuple(float(item) for item in row) for row in frame["covariance"])  # type: ignore[arg-type]
                color = tuple(float(item) for item in frame["color"])  # type: ignore[arg-type]
                if len(mean) != 3 or len(covariance) != 3 or any(len(row) != 3 for row in covariance) or len(color) != 3:
                    raise ValueError("anchor frame geometry or appearance is invalid")
                frames.append(AnchoredFrame(float(frame["timestamp"]), mean, covariance, float(frame["opacity"]), color))
            if not frames or any(right.timestamp <= left.timestamp for left, right in zip(frames, frames[1:])):
                raise ValueError("anchor frame timestamps must be nonempty and strictly increasing")
            payload_id = int(primitive_id)
            converted_payloads.append(BoundPayload(
                payload_id,
                tuple(frames),
                int(value["attribute_bytes"]),
                RecordLocation(payload_id, int(value["die"]), int(value["bank"]), int(value["block"]), int(value["slot"])),
            ))
        return cls(converted_anchors, converted_payloads)

    def decision_records(self) -> tuple[DecisionRecord, ...]:
        return tuple(DecisionRecord(anchor.anchor_id, {"start": anchor.start, "end": anchor.end}, anchor.location) for anchor in sorted(self._anchors.values(), key=lambda item: item.anchor_id))

    def select_window(self, window_start: float, window_end: float) -> Selection:
        start, end = require_window(window_start, window_end)
        interpreter = PPIMInterpreter()
        active = {record.record_id for record in self.decision_records() if interpreter.run(ProgramId.ANCHOR_OVERLAP, record.fields, start, end).mask}
        return Selection(active, ProgramId.ANCHOR_OVERLAP.value)

    def validate_window_frames(self, frames: Sequence[FrameRequest], selection: Selection) -> None:
        """Require every selected binding to supply every S2 frame before S1."""

        unknown = selection.active_ids - set(self._anchors)
        if unknown:
            raise ValueError(f"Anchored 4DGS selection contains unknown anchors: {sorted(unknown)}")
        requested = tuple(frame.timestamp for frame in frames)
        for anchor_id in sorted(selection.active_ids):
            for primitive_id in self.binding_records(anchor_id):
                available = {frame.timestamp for frame in self._payloads[primitive_id].frames}
                missing = tuple(timestamp for timestamp in requested if timestamp not in available)
                if missing:
                    raise ValueError(
                        f"anchor payload {primitive_id} does not support every requested S2 frame: {missing}"
                    )

    def binding_records(self, record_id: int) -> tuple[int, ...]:
        if record_id not in self._anchors:
            raise ValueError("unknown anchor record")
        anchor = self._anchors[record_id]
        return tuple(range(anchor.binding_start, anchor.binding_start + anchor.binding_count))

    def record_location(self, primitive_id: int) -> RecordLocation:
        return self._payloads[primitive_id].location

    def payload(self, primitive_id: int) -> Mapping[str, object]:
        payload = self._payloads[primitive_id]
        return {"primitive_id": primitive_id, "frame_geometry": payload.frames, "attribute_bytes": payload.attribute_bytes}

    @staticmethod
    def _frame_at(frames: tuple[AnchoredFrame, ...], timestamp: float) -> AnchoredFrame:
        for frame in frames:
            if frame.timestamp == float(timestamp):
                return frame
        raise ValueError("anchor payload has no representation-generated geometry for the requested timestamp")

    def generate_geometry_from_payloads(
        self,
        payloads: Mapping[int, Mapping[str, object]],
        timestamp: float,
    ) -> Mapping[int, Geometry]:
        """Generate S2 geometry from payloads expanded into the Active Buffer."""

        generated: dict[int, Geometry] = {}
        required = ("frame_geometry", "attribute_bytes")
        for primitive_id, payload in sorted(payloads.items()):
            missing = [key for key in required if key not in payload]
            if missing:
                raise ValueError(f"Active Buffer payload {primitive_id} is missing fields: {', '.join(missing)}")
            frames = payload["frame_geometry"]
            if not isinstance(frames, tuple) or not frames or not all(isinstance(frame, AnchoredFrame) for frame in frames):
                raise ValueError(f"Active Buffer payload {primitive_id} has invalid anchored frame geometry")
            frame = self._frame_at(frames, timestamp)
            generated[primitive_id] = Geometry(
                primitive_id,
                frame.mean,
                frame.covariance,
                frame.opacity,
                frame.color,
                int(payload["attribute_bytes"]),
            )
        return generated
