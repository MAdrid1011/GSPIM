"""Run deterministic mechanism fixtures and emit images plus dependency traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .layouts.anchored import Anchored4DGSLayout
from .layouts.ex4dgs import Ex4DGSLayout
from .layouts.explicit4d import Explicit4DLayout
from .model import Camera, FrameRequest
from .pipeline import GSPIMPipeline


def _camera() -> Camera:
    return Camera.from_mapping({"world_to_camera": ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)), "focal_x": 8.0, "focal_y": 8.0, "center_x": 4.0, "center_y": 4.0, "width": 8, "height": 8, "tile_size": 4, "near": 0.1, "far": 10.0})


def _covariance() -> tuple[tuple[float, float, float], ...]:
    return ((0.04, 0.0, 0.0), (0.0, 0.04, 0.0), (0.0, 0.0, 0.04))


def _explicit() -> Explicit4DLayout:
    rotation = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    return Explicit4DLayout.from_mappings((
        {"primitive_id": 0, "temporal_mean": 1.0, "temporal_rotation": rotation, "temporal_scale": (0.2, 0.2, 0.2, 0.5), "mean": (-0.2, 0.0, 2.0), "opacity": 0.6, "color": (0.9, 0.1, 0.1), "attribute_bytes": 48, "die": 0, "bank": 0, "block": 0, "slot": 0},
        {"primitive_id": 1, "temporal_mean": 4.4, "temporal_rotation": rotation, "temporal_scale": (0.2, 0.2, 0.2, 1.0), "mean": (0.1, 0.0, 3.0), "opacity": 0.5, "color": (0.1, 0.2, 0.9), "attribute_bytes": 56, "die": 1, "bank": 1, "block": 0, "slot": 0},
    ))


def _ex4dgs() -> Ex4DGSLayout:
    return Ex4DGSLayout.from_mappings((
        {"primitive_id": 10, "is_static": True, "keyframes": ({"timestamp": 0.0, "mean": (0.0, 0.0, 2.2), "covariance": _covariance(), "opacity": 0.5, "color": (0.7, 0.7, 0.2)},), "attribute_bytes": 44, "die": 0, "bank": 0, "block": 0, "slot": 0},
        {"primitive_id": 11, "is_static": False, "keyframes": ({"timestamp": 0.0, "mean": (0.1, 0.0, 2.8), "covariance": _covariance(), "opacity": 0.7, "color": (0.2, 0.8, 0.8)}, {"timestamp": 5.0, "mean": (0.1, 0.0, 3.5), "covariance": _covariance(), "opacity": 0.4, "color": (0.1, 0.4, 0.9)}), "attribute_bytes": 52, "die": 0, "bank": 0, "block": 0, "slot": 1},
    ))


def _anchored() -> Anchored4DGSLayout:
    return Anchored4DGSLayout.from_mappings(({"anchor_id": 20, "start": 0.0, "end": 3.0, "binding_start": 100, "binding_count": 2, "die": 0, "bank": 1, "block": 0, "slot": 0},), {
        100: {"frame_geometry": tuple({"timestamp": float(t), "mean": (-0.1, 0.0, 2.1), "covariance": _covariance(), "opacity": 0.5, "color": (0.6, 0.2, 0.6)} for t in range(6)), "attribute_bytes": 40, "die": 0, "bank": 3, "block": 0, "slot": 0},
        101: {"frame_geometry": tuple({"timestamp": float(t), "mean": (0.1, 0.0, 2.6 + 0.02 * t), "covariance": _covariance(), "opacity": 0.5, "color": (0.2, 0.6, 0.6)} for t in range(6)), "attribute_bytes": 40, "die": 0, "bank": 3, "block": 0, "slot": 1},
    })


def _write_ppm(path: Path, frame) -> None:
    payload = bytearray()
    for color in frame.pixels:
        payload.extend(max(0, min(255, round(component * 255.0))) for component in color)
    path.write_bytes(f"P6\n{frame.width} {frame.height}\n255\n".encode("ascii") + bytes(payload))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic GSPIM mechanism fixtures")
    parser.add_argument("--workload", choices=("explicit4d", "ex4dgs", "anchored", "all"), default="all")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    layouts = {"explicit4d": _explicit, "ex4dgs": _ex4dgs, "anchored": _anchored}
    names = tuple(layouts) if args.workload == "all" else (args.workload,)
    args.out.mkdir(parents=True, exist_ok=True)
    camera = _camera()
    frames = tuple(FrameRequest(float(index), camera) for index in range(6))
    summary: dict[str, object] = {}
    for name in names:
        result = GSPIMPipeline(l2_capacity=512).run_sequence(
            layouts[name](), frames, initial_window_size=4, frame_period=1.0,
        )
        checksums: list[str] = []
        for window_index, window in enumerate(result.windows):
            for frame_index, frame in enumerate(window.rendered_frames):
                _write_ppm(args.out / f"{name}-w{window_index}-f{frame_index}.ppm", frame)
                checksums.append(frame.checksum)
        (args.out / f"{name}-trace.json").write_text(json.dumps(result.trace.to_dict(), indent=2), encoding="ascii")
        summary[name] = {"checksums": checksums, "windows": len(result.windows)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="ascii")
    print(f"wrote mechanism outputs for {', '.join(names)} to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
