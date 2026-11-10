"""Deterministic S5 3DGS projection, tile assignment, and alpha compositing."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import ceil, exp, floor, sqrt
from typing import Mapping, Sequence

from .dataflow import Batch
from .hcf import HCFResult
from .model import Camera, FrameGeometry, Geometry


def load_kernel(name, cache):
    """Compile shared CUDA kernels into an external, content-keyed cache."""
    from torch.utils.cpp_extension import load
    from pathlib import Path
    import hashlib
    import torch

    folder = Path(__file__).with_name('pim_kernels')
    source = folder / f'{name}.cu'
    inputs = [source]
    includes = [folder]
    if name == 'ex4dgs_generation_publication':
        inputs.append(folder / 'ex4dgs_generation.cu')
    if name == 'mean':
        inputs.append(folder / 'Reduce.cuh')
    if name == 'projection':
        import gsplat
        upstream = Path(gsplat.__file__).parent / 'cuda'
        includes.extend((upstream / 'include', upstream / 'csrc', upstream / 'csrc/third_party/glm'))
        inputs.extend((upstream / 'include/Common.h', upstream / 'include/Utils.cuh', upstream / 'csrc/Projection.h'))
    flags = ['-O3', *(['-use_fast_math'] if name == 'projection' else [])]
    digest = hashlib.sha256(repr((torch.__version__, torch.version.cuda,
        torch.cuda.get_device_capability(), flags)).encode())
    for path in inputs:
        digest.update(path.read_bytes())
    module = f'gspim_record_{name}_{digest.hexdigest()[:16]}'
    directory = Path(cache) / 'cuda' / module
    directory.mkdir(parents=True, exist_ok=True)
    binary = load(module, [str(source)], build_directory=str(directory),
                  extra_cuda_cflags=flags, extra_include_paths=list(map(str, includes)), verbose=False)
    return binary


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


class CUDARenderer:
    """Native GPU operators shared by frame-by-frame and multi-frame execution."""

    def __init__(self, cache, model=None):
        from pathlib import Path
        import os

        import torch
        from torch.utils.cpp_extension import load

        cache = Path(cache).resolve()
        source_root = Path(__file__).resolve().parents[1]
        if cache == source_root or source_root in cache.parents:
            raise ValueError("CUDA cache must be outside the source checkout")
        cache.mkdir(parents=True, exist_ok=True)
        # gsplat also respects this location when its installed wheel needs JIT.
        os.environ["TORCH_EXTENSIONS_DIR"] = str(cache)
        os.environ.setdefault("MAX_JOBS", "2")
        self.ops = load("gspim_renderer", [str(Path(__file__).with_suffix(".cu"))], extra_cuda_cflags=["-O3"], verbose=False)
        self.torch = torch
        self.generation_model = model if hasattr(model, "motion_rotation") else None
        self.generation_parameters = (self.ops.prepare_generation(
            model.motion, model.motion_rotation, model.opacity_variance, model.interval,
            float(model.config["var_pad"]),
        ) if self.generation_model is not None else None)
        self.window_statistics = None
        if self.generation_parameters is not None:
            from .pim_record_ops import DepthStatistics
            self.window_statistics = DepthStatistics(load_kernel('depth', cache), self.ops)
            self.window_statistics.generation_binary = load_kernel('ex4dgs_generation', cache)
            self.window_statistics.generation_trace_binary = load_kernel('ex4dgs_generation_publication', cache)
            self.window_statistics.generation_view = model.view
            self.window_statistics.generation_span = (
                float(model.config.get('stability_far', model.config['far'])) -
                float(model.config.get('stability_near', model.config['near'])))
            # Finish immutable checkpoint preparation before another stream
            # can submit inference through this renderer.
            torch.cuda.current_stream(model.xyz.device).synchronize()

    def generate(self, model, times):
        if hasattr(model, 'selected_dynamic'):
            source = self.generation_model
            statistics = self.window_statistics
            if statistics.mode is None:
                return tuple(value.index_select(1, model.source_ids)
                             for value in self.generate(source, times))
            statistics.validate = False
            motion, rotation, variance = self.generation_parameters
            binary = (statistics.generation_trace_binary if statistics.trace_binary is not None
                      else statistics.generation_binary)
            result = binary.generate(source.xyz, source.displacement, motion, source.rotation,
                rotation, source.scales, source.opacity, source.opacity_center, variance,
                times, source.duration, source.interval, source.time_shift,
                float(source.config['var_pad']), model.view, statistics.generation_span,
                statistics.mode, model.selected_dynamic)
            statistics.generated[result[0].data_ptr()] = result[3:]
            statistics.last_generation_publication = result[6]
            statistics.last_generation_loads = result[7]
            return result[:3]
        if hasattr(model, "anchor_positions"):
            torch = self.torch
            features = model.anchor_features
            decoded = model.covariance_mlp(features).reshape(-1, 10, 12)
            opacity = model.opacity_mlp(features)
            direction = torch.nn.functional.normalize(
                model.anchor_positions[:, :3]-model.camera_position, dim=-1)
            model.decoded_colors = model.color_mlp(
                torch.cat((features, direction), -1)).reshape(-1, 3)
            return self.ops.generate_anchored(
                model.anchor_positions, model.anchor_offsets, model.anchor_log_scales,
                decoded, opacity, model.anchor_static, times, model.tau_t)
        if hasattr(model, "temporal_means"):
            return self.ops.generate_4d(
                model.xyz, model.scales4d, model.rotation_l, model.rotation_r,
                model.temporal_means, model.opacity, model.masks, model.source_ids,
                model.source_count, times)
        motion, rotation, variance = (self.generation_parameters if model is self.generation_model
                                     else (model.motion, model.motion_rotation, model.opacity_variance))
        return self.ops.generate(model.xyz, model.displacement, motion, model.rotation,
                                 rotation, model.scales, model.opacity,
                                 model.opacity_center, variance, times,
                                 model.duration, model.interval, model.time_shift, float(model.config["var_pad"]))

    def project(self, model, geometry, defer_colors=False):
        from gsplat.cuda._wrapper import fully_fused_projection

        torch = self.torch
        means, covariance, opacity = geometry
        width = len(means)
        if means.shape[1] == 0:
            # gsplat's projection launcher requires at least one primitive.
            # Keep empty windows on the normal ordering/compositing path.
            xy = means.new_empty((width, 0, 2))
            return (xy.to(torch.int32), xy, opacity.new_empty((width, 0)),
                    means.new_empty((width, 0, 3)),
                    means.new_empty(0) if defer_colors else means.new_empty((width, 0, 3)),
                    opacity)
        views = model.view[None].expand(width, -1, -1, -1).contiguous()
        intrinsics = model.intrinsics[None].expand(width, -1, -1, -1).contiguous()
        temporal = hasattr(model, "temporal_means")
        classic = temporal or hasattr(model, "anchor_positions")
        radii, xy, depth, conics, compensation = fully_fused_projection(
            means, covariance, None, None, views, intrinsics, model.width, model.height,
            eps2d=float(model.config["kernel_size"]), near_plane=float(model.config["near"]),
            far_plane=float(model.config["far"]), packed=False, opacities=opacity,
            calc_compensations=not classic,
        )
        radii, xy, depth, conics = [value.squeeze(1) for value in (radii, xy, depth, conics)]
        # Native Ex4DGS rejects centers outside this expanded camera frustum.
        if not classic:
            opacity = self.ops.projection_finish(xy, radii, opacity, compensation.squeeze(1), model.width, model.height)
        if defer_colors:
            return radii, xy, depth, conics, means.new_empty(0), opacity
        colors = self.colors(model, means, radii)
        return radii, xy, depth, conics, colors, opacity

    def colors(self, model, means, radii):
        if hasattr(model, "anchor_positions"):
            return model.decoded_colors[None].expand(len(means), -1, -1).contiguous()
        if hasattr(model, "temporal_means"):
            return self.ops.temporal_colors(
                model.xyz, radii, model.sh, model.camera_position,
                model.frame_times, model.temporal_means)
        return self.ops.batch_colors(means, radii, model.sh, model.camera_position)

    def frame(self, model, projected):
        from gsplat.cuda._wrapper import isect_tiles, isect_offset_encode, rasterize_to_pixels

        radii, xy, depth, conics, colors, opacity = projected
        tile_width, tile_height = (model.width + 15) // 16, (model.height + 15) // 16
        _, keys, ids = isect_tiles(xy, radii, depth, 16, tile_width, tile_height,
                                  segmented=False, sort=True)
        offsets = isect_offset_encode(keys, 1, tile_width, tile_height)
        image, _ = rasterize_to_pixels(
            xy, conics, colors, opacity, model.width, model.height, 16, offsets, ids,
            backgrounds=xy.new_full((1, 3), float(model.config.get("background", 1))),
        )
        return image[0]

    def _tile_records(self, model, projected, orders, assignment, batches, shared):
        radii, xy, _, conics, _, opacity = projected
        count = orders.shape[1]
        tile_width, tile_height = (model.width + 15) // 16, (model.height + 15) // 16
        id_bits = count.bit_length() if shared else 0
        keys, ids = self.ops.ordered_tiles(xy, radii, orders if shared else orders.contiguous(), assignment, batches, tile_width, tile_height, 16, shared)
        return self.ops.cull_tiles(keys, ids, xy, conics, opacity, tile_width, tile_height, 16, 0, radii, id_bits)

    def window(self, model, projected, orders, assignment, batches, shared=False, means=None, sh=None, ordered_attributes=False, source_ids=None, layout=None, output=None, wait_for=None, tile_records=None):
        torch = self.torch
        radii, xy, _, conics, colors, opacity = projected
        width, count = orders.shape
        tile_width, tile_height = (model.width + 15) // 16, (model.height + 15) // 16
        id_bits = count.bit_length() if shared else 0
        # Input is enumerated in ROL order; a stable radix grouping on tile and
        # batch bits retains that order without sorting frame depths again.
        keys, ids = (self._tile_records(model, projected, orders, assignment, batches, shared)
                     if tile_records is None else tile_records)
        image_groups = batches if shared else batches * width
        keys, ids = self.ops.radix(
            keys, ids, 0,
            (tile_width * tile_height).bit_length() + image_groups.bit_length(),
        )
        # The masked compositor accepts packed batch/tile keys and replaces
        # them with coordinates before launching CTAs. Other paths use linear jobs.
        packed_jobs = shared and id_bits > 0 and id_bits + width + 8 <= 32
        jobs, offsets, counts = self.ops.tile_ranges(
            keys, 1 if shared else width, batches, tile_width * tile_height, 0, packed_jobs)
        boundaries = (torch.searchsorted(assignment[0], torch.arange(batches + 1, device=orders.device))
                      .cpu().tolist()) if means is not None or (shared and ordered_attributes) else []
        # Metadata may overlap the preceding span; GPU batch preparation and
        # composition start only after that span's render completes.
        if wait_for is not None:
            torch.cuda.current_stream(xy.device).wait_event(wait_for)
        temporal = hasattr(model, "temporal_means")
        color_means = model.xyz[None] if temporal else means
        return self.ops.batches(xy, conics, colors, opacity, offsets, ids,
                                model.width, model.height, batches, jobs, counts,
                                ids.numel() > 64 * len(jobs), id_bits,
                                color_means if means is not None else xy.new_empty(0),
                                model.sh if sh is None else sh, model.camera_position, orders, boundaries,
                                ordered_attributes, orders.new_empty(0) if source_ids is None else source_ids,
                                orders.new_empty(0) if layout is None else layout,
                                xy.new_empty(0) if output is None else output,
                                float(model.config.get("background", 1)),
                                model.frame_times if temporal else None,
                                model.temporal_means if temporal else None)


def cuda_image_quality(image, target):
    """PSNR and RGB SSIM with the upstream 11x11 Gaussian window."""
    import torch
    import torch.nn.functional as functional

    image = image.clamp(0, 1).permute(2, 0, 1)[None]
    target = target.permute(2, 0, 1)[None]
    psnr = -10 * (image - target).square().mean().log10()
    coordinates = torch.arange(11, device=image.device, dtype=image.dtype) - 5
    gaussian = torch.exp(-coordinates.square() / (2 * 1.5 ** 2))
    gaussian /= gaussian.sum()
    kernel = (gaussian[:, None] * gaussian[None, :])[None, None].expand(3, 1, 11, 11).contiguous()
    moments = functional.conv2d(torch.cat((image, target, image.square(), target.square(), image * target)), kernel, padding=5, groups=3)
    mean_a, mean_b, square_a, square_b, product = moments.unbind(0)
    variance_a = square_a - mean_a.square()
    variance_b = square_b - mean_b.square()
    covariance = product - mean_a * mean_b
    numerator = (2 * mean_a * mean_b + 0.01 ** 2) * (2 * covariance + 0.03 ** 2)
    denominator = (mean_a.square() + mean_b.square() + 0.01 ** 2) * (variance_a + variance_b + 0.03 ** 2)
    return float(psnr.item()), float((numerator / denominator).mean().item())
