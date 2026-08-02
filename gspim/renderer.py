"""Deterministic S5 3DGS projection, tile assignment, and alpha compositing."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import ceil, exp, floor, sqrt
from typing import Mapping, Sequence

from .dataflow import Batch
from .hcf import HCFResult
from .model import Camera, FrameGeometry, Geometry


@dataclass(frozen=True)
class RenderedFrame:
    width: int
    height: int
    pixels: tuple[tuple[float, float, float], ...]
    transmittance: tuple[float, ...]
    order: tuple[int, ...]
    tiles_by_id: Mapping[int, tuple[tuple[int, int], ...]]
    checksum: str


def _camera_point(camera: Camera, point: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = point
    matrix = camera.world_to_camera
    return tuple(sum(row[index] * (x, y, z, 1.0)[index] for index in range(4)) for row in matrix[:3])  # type: ignore[return-value]


def camera_depth(geometry: Geometry, camera: Camera) -> float | None:
    """Return only the S3-required camera-space depth.

    Projection, 2D covariance construction, and tile assignment remain S5
    work.  Keeping this helper separate makes the five-stage boundary visible
    in the executable dataflow.
    """

    depth = _camera_point(camera, geometry.mean_world)[2]
    return depth if depth > 0.0 else None


def frame_geometry(geometry: Geometry, camera: Camera) -> FrameGeometry | None:
    """Produce projection and conservative tile coverage for a 3D Gaussian."""

    x, y, depth = _camera_point(camera, geometry.mean_world)
    if depth <= 0.0:
        return None
    center = (camera.focal_x * x / depth + camera.center_x, camera.focal_y * y / depth + camera.center_y)
    rotation = tuple(tuple(camera.world_to_camera[row][column] for column in range(3)) for row in range(3))
    covariance = geometry.covariance_world
    camera_covariance = tuple(
        tuple(sum(rotation[row][left] * covariance[left][right] * rotation[column][right] for left in range(3) for right in range(3)) for column in range(3))
        for row in range(3)
    )
    jacobian = ((camera.focal_x / depth, 0.0, -camera.focal_x * x / (depth * depth)), (0.0, camera.focal_y / depth, -camera.focal_y * y / (depth * depth)))
    screen_covariance = tuple(
        tuple(sum(jacobian[row][left] * camera_covariance[left][right] * jacobian[column][right] for left in range(3) for right in range(3)) for column in range(2))
        for row in range(2)
    )
    var_x = max(1e-6, screen_covariance[0][0])
    var_y = max(1e-6, screen_covariance[1][1])
    cov_xy = screen_covariance[0][1]
    radius = 3.0 * sqrt(max(var_x, var_y))
    tile_x_start = max(0, floor((center[0] - radius) / camera.tile_size))
    tile_x_end = min(ceil(camera.width / camera.tile_size) - 1, floor((center[0] + radius) / camera.tile_size))
    tile_y_start = max(0, floor((center[1] - radius) / camera.tile_size))
    tile_y_end = min(ceil(camera.height / camera.tile_size) - 1, floor((center[1] + radius) / camera.tile_size))
    tiles = tuple((tile_x, tile_y) for tile_y in range(tile_y_start, tile_y_end + 1) for tile_x in range(tile_x_start, tile_x_end + 1))
    return FrameGeometry(geometry, (x, y, depth), depth, center, (var_x, cov_xy, var_y), tiles)


class BatchRenderSession:
    """Mutable per-frame accumulation state for one S5 ping-pong stream."""

    def __init__(self, renderer: "ReferenceRenderer", cameras: Sequence[Camera]) -> None:
        self._renderer = renderer
        self._cameras = tuple(cameras)
        self._colors = [[(0.0, 0.0, 0.0) for _ in range(camera.width * camera.height)] for camera in cameras]
        self._transmittance = [[1.0 for _ in range(camera.width * camera.height)] for camera in cameras]
        self._tiles: list[dict[int, tuple[tuple[int, int], ...]]] = [dict() for _ in cameras]
        self._orders: list[list[int]] = [[] for _ in cameras]

    def consume(self, batch: Batch, packed: HCFResult, fallback_records: Sequence[object]) -> None:
        geometry_by_frame = self._renderer._packed_geometry(batch, packed, fallback_records, self._cameras)
        for frame_index, camera in enumerate(self._cameras):
            order = batch.frame_rols[frame_index]
            missing = set(order) - set(geometry_by_frame[frame_index])
            if missing:
                raise ValueError(f"Batch Buffer is missing geometry for frame {frame_index}: {sorted(missing)}")
            self._renderer._composite(order, geometry_by_frame[frame_index], camera, self._colors[frame_index], self._transmittance[frame_index], self._tiles[frame_index])
            self._orders[frame_index].extend(order)

    def finish(self) -> tuple[RenderedFrame, ...]:
        output: list[RenderedFrame] = []
        for camera, colors, transmittance, tiles, order in zip(self._cameras, self._colors, self._transmittance, self._tiles, self._orders):
            checksum_input = "|".join(f"{value:.9f}" for pixel in colors for value in pixel).encode("ascii")
            output.append(RenderedFrame(camera.width, camera.height, tuple(colors), tuple(transmittance), tuple(order), tiles, sha256(checksum_input).hexdigest()))
        return tuple(output)


class ReferenceRenderer:
    """A semantic renderer retaining pixel state across contiguous batches."""

    @staticmethod
    def _composite(
        order: Sequence[int],
        geometry: Mapping[int, Geometry],
        camera: Camera,
        colors: list[tuple[float, float, float]],
        transmittance: list[float],
        tiles: dict[int, tuple[tuple[int, int], ...]],
    ) -> None:
        for primitive_id in order:
            projected = frame_geometry(geometry[primitive_id], camera)
            if projected is None:
                tiles[primitive_id] = ()
                continue
            tiles[primitive_id] = projected.tiles
            var_x, cov_xy, var_y = projected.covariance_2d
            determinant = var_x * var_y - cov_xy * cov_xy
            if determinant <= 1e-12:
                continue
            inv_xx, inv_xy, inv_yy = var_y / determinant, -cov_xy / determinant, var_x / determinant
            radius = 3.0 * sqrt(max(var_x, var_y))
            min_x, max_x = max(0, floor(projected.projected_center[0] - radius)), min(camera.width - 1, ceil(projected.projected_center[0] + radius))
            min_y, max_y = max(0, floor(projected.projected_center[1] - radius)), min(camera.height - 1, ceil(projected.projected_center[1] + radius))
            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    dx, dy = x + 0.5 - projected.projected_center[0], y + 0.5 - projected.projected_center[1]
                    weight = exp(-0.5 * (inv_xx * dx * dx + 2.0 * inv_xy * dx * dy + inv_yy * dy * dy))
                    alpha = min(0.999, max(0.0, projected.geometry.opacity * weight))
                    index = y * camera.width + x
                    previous = colors[index]
                    colors[index] = tuple(previous[channel] + transmittance[index] * alpha * projected.geometry.color[channel] for channel in range(3))  # type: ignore[assignment]
                    transmittance[index] *= 1.0 - alpha

    @staticmethod
    def _packed_geometry(
        batch: Batch,
        packed: HCFResult,
        fallback_records: Sequence[object],
        cameras: Sequence[Camera],
    ) -> tuple[dict[int, Geometry], ...]:
        records = tuple(packed.packed_records) + tuple(fallback_records)
        if packed.purpose.value != "batch" or {record.primitive_id for record in records} != set(batch.attribute_ids):
            raise ValueError("renderer received data not packed for its batch")
        packed_geometry_by_frame: list[dict[int, Geometry]] = [dict() for _ in cameras]
        for record in records:
            payload = record.payload
            frame_geometry = getattr(payload, "frame_geometry", None)
            if not isinstance(frame_geometry, tuple) or len(frame_geometry) != len(cameras):
                raise ValueError("Batch Buffer payload is missing frame-aligned geometry")
            for frame_index, geometry in enumerate(frame_geometry):
                if geometry is None:
                    continue
                if not isinstance(geometry, Geometry) or geometry.primitive_id != record.primitive_id:
                    raise ValueError("Batch Buffer payload has invalid geometry")
                packed_geometry_by_frame[frame_index][record.primitive_id] = geometry
        return tuple(packed_geometry_by_frame)

    def begin_batches(self, cameras: Sequence[Camera]) -> BatchRenderSession:
        return BatchRenderSession(self, cameras)

    def render_order(self, order: Sequence[int], geometry: Mapping[int, Geometry], camera: Camera) -> RenderedFrame:
        colors = [(0.0, 0.0, 0.0) for _ in range(camera.width * camera.height)]
        transmittance = [1.0 for _ in range(camera.width * camera.height)]
        tiles: dict[int, tuple[tuple[int, int], ...]] = {}
        self._composite(order, geometry, camera, colors, transmittance, tiles)
        checksum_input = "|".join(f"{value:.9f}" for pixel in colors for value in pixel).encode("ascii")
        return RenderedFrame(camera.width, camera.height, tuple(colors), tuple(transmittance), tuple(order), tiles, sha256(checksum_input).hexdigest())

    def render_batches(
        self,
        batches: Sequence[Batch],
        cameras: Sequence[Camera],
        packed_batches: Sequence[HCFResult],
        gpu_fallback_records: Sequence[Sequence[object]],
    ) -> tuple[RenderedFrame, ...]:
        if len(packed_batches) != len(batches) or len(gpu_fallback_records) != len(batches):
            raise ValueError("every render batch requires packed and fallback data")

        session = self.begin_batches(cameras)
        for batch, packed, fallback_records in zip(batches, packed_batches, gpu_fallback_records):
            session.consume(batch, packed, fallback_records)
        return session.finish()
