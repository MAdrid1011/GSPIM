"""Run the registered CUDA workloads or prepare their inputs."""

from __future__ import annotations

import argparse
import http.client
import subprocess
from pathlib import Path
import sys
import zipfile

from integrations.manifest import CACHE, FRAMES, ROOT, SCENES, WORKLOADS, combinations
from .prepare import input_problems, prepare


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare inputs or run GSPIM CUDA workloads")
    parser.add_argument("--prepare", action="store_true", help="prepare public data and check supplied models")
    parser.add_argument("--workload", choices=(*WORKLOADS, "all"), default="all")
    parser.add_argument("--scene", choices=(*SCENES, "all"), default="all")
    parser.add_argument("--out", type=Path, help="report directory (default: outputs/<workload>/<scene>)")
    parser.add_argument("--checkpoint", type=Path, help="single-combination model override")
    parser.add_argument("--images", type=Path, help="single-combination held-out image override")
    parser.add_argument("--cache", type=Path, default=CACHE, help="external compilation/download cache")
    parser.add_argument("--frames", type=int, default=FRAMES)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=20, help="number of warmup windows")
    args = parser.parse_args(argv)
    if not 2 <= args.frames <= FRAMES or args.repeats < 1 or args.warmup < 1:
        parser.error("frames must be in 2..300; repeats and warmup must be positive")
    cache = args.cache.resolve()
    if cache == ROOT or ROOT in cache.parents:
        parser.error("--cache must be outside the checkout")
    if args.out is not None:
        output = args.out.resolve()
        allowed = ROOT / "outputs"
        if (output == ROOT or ROOT in output.parents) and output != allowed and allowed not in output.parents:
            parser.error("outputs inside the checkout must be under outputs/")
    pairs = tuple(combinations(args.workload, args.scene))
    if (args.checkpoint or args.images) and len(pairs) != 1:
        parser.error("--checkpoint and --images require a single --workload and --scene")
    if args.prepare and (args.checkpoint or args.images or args.out):
        parser.error("--prepare uses the registered input paths; path overrides apply to rendering")
    if args.prepare:
        try:
            problems = prepare(pairs, cache=args.cache)
        except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError, http.client.HTTPException, zipfile.BadZipFile) as error:
            print(f"Preparation failed: {error}", file=sys.stderr)
            return 1
    else:
        problems = input_problems(pairs, frames=args.frames, checkpoint=args.checkpoint, images=args.images)
    if problems:
        print("Missing inputs:\n" + "\n".join(f"- {item}" for item in problems), file=sys.stderr)
        return 1
    if args.prepare:
        print("Inputs are ready.")
        return 0

    import gc
    import torch
    from .pipeline import CUDAPipeline

    if not torch.cuda.is_available():
        parser.error("a CUDA device is required; see docs/setup.md")
    for workload, scene in pairs:
        checkpoint = args.checkpoint or workload.checkpoint(scene)
        images = args.images or workload.image_directory(scene)
        output = args.out or ROOT / "outputs"
        if len(pairs) != 1 or args.out is None:
            output = output / workload.name / scene
        print(f"Running {workload.name}/{scene}", flush=True)
        if workload.name == "explicit4d":
            from integrations.explicit4d import Explicit4DCheckpoint
            model = Explicit4DCheckpoint(checkpoint, images)
        elif workload.name == "anchored":
            from integrations.anchored import AnchoredCheckpoint
            model = AnchoredCheckpoint(checkpoint, images)
        else:
            from integrations.ex4dgs import Ex4DGSCheckpoint
            model = Ex4DGSCheckpoint(checkpoint)
        pipeline = CUDAPipeline(model, args.cache)
        pipeline.evaluate(images, output, args.frames, args.repeats, args.warmup)
        print(f"Report: {output / 'summary.md'}", flush=True)
        del pipeline, model
        gc.collect()
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
