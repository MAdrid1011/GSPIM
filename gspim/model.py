"""Canonical record, camera, block, and geometry contracts for GSPIM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence


Vector3 = tuple[float, float, float]
Vector4 = tuple[float, float, float, float]
Matrix3 = tuple[Vector3, Vector3, Vector3]
Matrix4 = tuple[Vector4, Vector4, Vector4, Vector4]
Color = Vector3


@dataclass(frozen=True)
class Camera:
    """A fixed camera used to produce camera-space and raster data."""

    world_to_camera: Matrix4
    focal_x: float
    focal_y: float
    center_x: float
    center_y: float
    width: int
    height: int
    tile_size: int
    near: float
    far: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "Camera":
        required = ("world_to_camera", "focal_x", "focal_y", "center_x", "center_y", "width", "height", "tile_size", "near", "far")
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"camera is missing fields: {', '.join(missing)}")
        matrix = tuple(tuple(float(item) for item in row) for row in value["world_to_camera"])  # type: ignore[arg-type]
        if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
            raise ValueError("world_to_camera must be 4x4")
        camera = cls(
            matrix,  # type: ignore[arg-type]
            float(value["focal_x"]),
            float(value["focal_y"]),
            float(value["center_x"]),
            float(value["center_y"]),
            int(value["width"]),
            int(value["height"]),
            int(value["tile_size"]),
            float(value["near"]),
            float(value["far"]),
        )
        if camera.width < 1 or camera.height < 1 or camera.tile_size < 1 or camera.far <= camera.near:
            raise ValueError("camera dimensions and depth range must be positive")
        return camera


@dataclass(frozen=True)
class FrameRequest:
    """One render timestamp and its camera."""

    timestamp: float
    camera: Camera


@dataclass(frozen=True)
class RecordLocation:
    """The aligned physical position used by PPIM and HCF."""

    record_id: int
    die: int
    bank: int
    block: int
    slot: int

    def __post_init__(self) -> None:
        if any(value < 0 for value in (self.record_id, self.die, self.bank, self.block, self.slot)):
            raise ValueError("physical record locations must be nonnegative")


def physical_location_key(location: RecordLocation) -> tuple[int, int, int, int]:
    """Return the mask-addressable location, excluding its logical record ID."""

    return (location.die, location.bank, location.block, location.slot)


def require_unique_physical_locations(locations: Sequence[RecordLocation], owner: str) -> None:
    """Reject two logical records that would share one PPIM mask bit."""

    seen: dict[tuple[int, int, int, int], int] = {}
    for location in locations:
        key = physical_location_key(location)
        previous = seen.get(key)
        if previous is not None:
            raise ValueError(
                f"{owner} has duplicate physical location {key} for records {previous} and {location.record_id}"
            )
        seen[key] = location.record_id


@dataclass(frozen=True)
class Geometry:
    """A generated 3D Gaussian before a camera-dependent rendering pass."""

    primitive_id: int
    mean_world: Vector3
    covariance_world: Matrix3
    opacity: float
    color: Color
    attribute_bytes: int

    def __post_init__(self) -> None:
        if len(self.covariance_world) != 3 or any(len(row) != 3 for row in self.covariance_world):
            raise ValueError("covariance_world must be 3x3")
        if self.attribute_bytes < 1:
            raise ValueError("attribute_bytes must be positive")


@dataclass(frozen=True)
class FrameGeometry:
    """A 3D Gaussian with camera-space and screen-space fields."""

    geometry: Geometry
    mean_camera: Vector3
    depth: float
    projected_center: tuple[float, float]
    covariance_2d: tuple[float, float, float]
    tiles: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class DecisionRecord:
    """Temporal fields and an aligned location for one PPIM decision."""

    record_id: int
    fields: Mapping[str, object]
    location: RecordLocation


@dataclass(frozen=True)
class Selection:
    """The selected decision IDs and PPIM program used to obtain them."""

    active_ids: set[int]
    program_name: str


class ModelLayout(Protocol):
    """Common typed interface implemented by all workload representations."""

    name: str

    def decision_records(self) -> tuple[DecisionRecord, ...]: ...

    def select_window(self, window_start: float, window_end: float) -> Selection: ...

    def validate_window_frames(self, frames: Sequence[FrameRequest], selection: Selection) -> None: ...

    def binding_records(self, record_id: int) -> tuple[int, ...]: ...

    def record_location(self, primitive_id: int) -> RecordLocation: ...

    def payload(self, primitive_id: int) -> Mapping[str, object]: ...

    def generate_geometry_from_payloads(
        self,
        payloads: Mapping[int, Mapping[str, object]],
        timestamp: float,
    ) -> Mapping[int, Geometry]: ...
