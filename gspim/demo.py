"""Run the registered CUDA workloads or prepare their inputs."""

from __future__ import annotations

import argparse
import csv
import http.client
import subprocess
import tempfile
from pathlib import Path
import sys
import zipfile

from integrations.manifest import CACHE, FRAMES, ROOT, SCENE_FRAMES, SCENES, WORKLOADS, combinations
from .prepare import input_problems, prepare
from .pim import load_or_build, load_trace, recording_identity, simulate_all

DISPLAY_NAMES = {
    "explicit4d": "4DGS-1K",
    "ex4dgs": "Ex4DGS",
    "anchored": "Anchored 4DGS",
}
WORKLOAD_NAMES = {"4dgs1k": "explicit4d", "ex4dgs": "ex4dgs", "anchored": "anchored"}
SCENE_DISPLAY_NAMES = {
    "coffee_martini": "Coffee Martini",
    "cook_spinach": "Cook Spinach",
    "sear_steak": "Sear Steak",
    "flame_steak": "Flame Steak",
    "dance3": "Dance3",
    "actor1_4": "Actor1_4",
}
PLOT_MODEL_NAMES = {
    "explicit4d": "1K",
    "ex4dgs": "Ex",
    "anchored": "Anch.",
}


def _load_registered_trace(workload, scene, frames, identity, *, independent=False, **options):
    """Use a matching prepared trace before recording one into the cache."""
    recording_key = recording_identity(workload, scene, frames, identity,
                                       independent=independent)
    suffix = ".frame.trace.npz" if independent else ".trace.npz"
    path = ROOT / "traces" / workload / f"{scene}{suffix}"
    candidates = [(path, True)] if path.is_file() else []
    cache = options.get("cache")
    if cache is not None:
        folder = Path(cache) / "pim" / "traces"
        cached = folder.glob(f"{workload}-{scene}-*.trace.npz")
        candidates.extend((candidate, False) for candidate in sorted(
            (candidate for candidate in cached
             if candidate.name.endswith(suffix) and
             (independent or not candidate.name.endswith(".frame.trace.npz"))),
            key=lambda candidate: candidate.stat().st_mtime,
            reverse=True,
        ))
    for candidate, registered in candidates:
        try:
            trace = load_trace(candidate)
        except (EOFError, OSError, ValueError):
            if registered:
                raise
            continue
        if (trace.frames == frames and trace.metadata.get("input_sha256") == identity and
                (trace.metadata.get("recording_sha256") == recording_key or
                 _audited_trace_match(trace, cache, workload, scene, recording_key,
                                      independent=independent))):
            if independent and any(window["width"] != 1 for window in trace.metadata["windows"]):
                raise ValueError(f"independent-frame PIM requires single-frame windows: {candidate}")
            return trace
    return load_or_build(workload, scene, ROOT, frames, independent=independent, **options)


def _audited_trace_match(trace, cache, workload, scene, recording_key, *, independent):
    from .trace_cache import audited_compatible

    return audited_compatible(trace, cache, workload, scene, recording_key,
                              independent=independent)


def _validate_trace_images(trace, pipeline) -> None:
    """Render captured windows and require exact RGB and window decisions."""
    import hashlib
    import torch

    checks = trace.metadata.get("image_checks", {})
    if (checks.get("shared_gpu_rgb_error") != 0 or
            checks.get("scheduling_rgb_error") != 0):
        raise ValueError(f"PIM trace has no exact image validation: {trace.path}")
    times = torch.arange(trace.frames, device=pipeline.model.xyz.device, dtype=torch.float32)
    hashes, decisions = [], []
    covered = 0
    with torch.inference_mode():
        for window in trace.metadata["windows"]:
            first, width = window["first_frame"], window["width"]
            if first != covered or not 1 <= width <= 5 or first + width > trace.frames:
                raise ValueError(f"PIM trace windows do not cover the frames: {trace.path}")
            rgb, decision, _, _ = pipeline.dataflow(times[first:first + width], first)
            decisions.append(decision)
            hashes.extend(hashlib.sha256(image.cpu().contiguous().numpy().tobytes()).hexdigest()
                          for image in rgb)
            covered += width
    if covered != trace.frames or hashes != trace.metadata.get("candidate_rgb_sha256"):
        raise ValueError(f"PIM trace RGB differs from GPU Dataflow: {trace.path}")
    if decisions != checks.get("window_decisions"):
        raise ValueError(f"PIM trace window decisions differ from GPU Dataflow: {trace.path}")


def _write_verification(rows: list[dict], output: Path) -> None:
    """Write the result tables and plots for the registered runs."""
    _write_report(rows, output / "verification.md", plots=True)


def _write_report(rows: list[dict], destination: Path, *, plots: bool = False) -> None:
    """Write measured Performance, Ablation and Quality tables."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Results", "", "## Speedup", "",
             "| Model | Dataset | Dataflow | PIM-only | Serial | Full |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(
            f"| {DISPLAY_NAMES[row['workload']]} | {SCENE_DISPLAY_NAMES[row['scene']]} | "
            f"{row['gpu_speedup']:.4f} | {row['pim_speedup']:.4f} | "
            f"{row['pim_dataflow_speedup']:.4f} | {row['full_speedup']:.4f} |")
    if plots:
        lines.extend(["", "![Performance](performance.png)"])
        lines.extend(["", "![Ablation](ablation.png)"])
    lines.extend(["", "## Quality", "",
                  "| Model | Dataset | Frames | PSNR Base | PSNR Dataflow | PSNR Full | SSIM Base | SSIM Dataflow | SSIM Full |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in rows:
        lines.append(
            f"| {DISPLAY_NAMES[row['workload']]} | {SCENE_DISPLAY_NAMES[row['scene']]} | {row['frames']} | "
            f"{row['psnr_baseline']:.3f} | {row['psnr_dataflow']:.3f} | {row['psnr_full']:.3f} | "
            f"{row['ssim_baseline']:.4f} | {row['ssim_dataflow']:.4f} | {row['ssim_full']:.4f} |"
        )
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if plots:
        _write_plots(rows, destination.parent)


def _write_plots(rows: list[dict], output: Path) -> None:
    """Render the two result charts through the pinned Golden-Ratio tool."""
    output = output.resolve()
    plotter = Path(__file__).resolve().parents[1] / "third_party" / "Golden-Ratio" / "main.py"
    if not plotter.is_file():
        raise FileNotFoundError(f"plotter is missing: {plotter}")
    workload_order = tuple(WORKLOADS)
    keys = [(row["workload"], row["scene"]) for row in rows]
    if len(rows) == len(WORKLOADS) * len(SCENES):
        expected = {(workload, scene) for workload in WORKLOADS for scene in SCENES}
        if len(set(keys)) != len(keys) or set(keys) != expected:
            raise ValueError("default result plots require every model and dataset combination once")
    ordered = sorted(rows, key=lambda row: (SCENES.index(row["scene"]), workload_order.index(row["workload"])))
    with tempfile.TemporaryDirectory(prefix="gspim-results-") as temporary:
        temporary = Path(temporary)
        performance_csv = temporary / "performance.csv"
        ablation_csv = temporary / "ablation.csv"
        for path, labels in ((performance_csv, ("GPU", "GPU + GSPIM Dataflow", "GSPIM")),
                             (ablation_csv, ("GPU", "+PIM", "+GSPIM Dataflow", "+PIM + GSPIM Dataflow", "Full GSPIM"))):
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(("group", "label", "major_group", "minor_group", "Normalized Speedup"))
                for row in ordered:
                    group = f"{SCENE_DISPLAY_NAMES[row['scene']]} / {DISPLAY_NAMES[row['workload']]}"
                    major = SCENE_DISPLAY_NAMES[row["scene"]]
                    minor = PLOT_MODEL_NAMES[row["workload"]]
                    values = {
                        "GPU": 1.0,
                        "GPU + GSPIM Dataflow": row["gpu_speedup"],
                        "GSPIM": row["full_speedup"],
                        "+PIM": row["pim_speedup"],
                        "+GSPIM Dataflow": row["gpu_speedup"],
                        "+PIM + GSPIM Dataflow": row["pim_dataflow_speedup"],
                        "Full GSPIM": row["full_speedup"],
                    }
                    for label in labels:
                        writer.writerow((group, label, major, minor, f"{values[label]:.6f}"))
        common = [sys.executable, str(plotter), "--two_level_xaxis", "--width_pt", "540",
                  "--font_size_pt", "7", "--label_font_size_pt", "8", "--y_min", "0", "--y_ticks", "6",
                  "--tight_pad_pt", "4", "--formats", "pdf", "png"]
        subprocess.run(common + ["--palette", "#d4cec6", "#9aa6b2", "#2f6b9a",
                                 "--height_pt", "132",
                                 "--input", str(performance_csv), "--output", str(output / "performance")],
                       cwd=plotter.parent, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        subprocess.run(common + ["--palette", "#b5bdc5", "#d98e3d", "#5b8e7d", "#6fa3c8", "#2f6b9a",
                                 "--height_pt", "132", "--input", str(ablation_csv),
                                 "--output", str(output / "ablation")],
                       cwd=plotter.parent, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _run_pair(workload, scene, *, frames, repeats, warmup, checkpoint, images,
              cache, output, library, config):
    import gc
    import torch
    from .pipeline import CUDAPipeline
    from .pim_trace import input_identity
    from .progress import Progress

    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA device is required; see docs/setup.md")
    run_frames = min(frames, SCENE_FRAMES.get(scene, frames))
    checkpoint = checkpoint or workload.checkpoint(scene)
    images = images or workload.image_directory(scene)
    if workload.name == "explicit4d":
        from integrations.explicit4d import Explicit4DCheckpoint
        model = Explicit4DCheckpoint(checkpoint, images)
    elif workload.name == "anchored":
        from integrations.anchored import AnchoredCheckpoint
        model = AnchoredCheckpoint(checkpoint, images)
    else:
        from integrations.ex4dgs import Ex4DGSCheckpoint
        model = Ex4DGSCheckpoint(checkpoint)
    trace_options = dict(model=model, checkpoint=checkpoint, images=images, cache=cache)
    trace_frames = min(run_frames, 10)
    identity = input_identity(checkpoint, images, model, trace_frames)
    with Progress("Traces", total=2) as progress:
        trace = _load_registered_trace(workload.name, scene, trace_frames, identity,
                                       **trace_options)
        progress.set("Traces", 1, 2)
        independent = _load_registered_trace(workload.name, scene, trace_frames, identity,
                                             independent=True, **trace_options)
    for captured in (trace, independent):
        if captured.metadata.get("input_sha256") != identity:
            raise ValueError(
                f"PIM trace does not match the checkpoint, camera or images: {captured.path}")
        if len(captured.metadata.get("quality", [])) != captured.frames:
            raise ValueError(f"PIM trace has no per-frame quality: {captured.path}")
    with Progress("CUDA"):
        pipeline = CUDAPipeline(model, cache)
    with Progress("Trace images", total=2) as progress:
        for index, captured in enumerate((trace, independent)):
            _validate_trace_images(captured, pipeline)
            progress.set("Trace images", index + 1, 2)
    with Progress("GPU") as progress:
        measured = pipeline.evaluate(images, output, run_frames, repeats, warmup,
                                     progress=progress.set)
    baseline_ms = measured["baseline_ms"] / run_frames
    dataflow_ms = measured["dataflow_ms"] / run_frames
    trace_frames = trace.frames
    # GPU evaluation is complete; release model storage before CPU simulation.
    del pipeline, model
    gc.collect()
    torch.cuda.empty_cache()
    from .pim import _schedule_destination
    destinations = [
        _schedule_destination(trace, library, config, overlap, cache)
        for overlap in (True, False)
    ] + [_schedule_destination(independent, library, config, True, cache, frame_order=True)]
    with Progress("PIM schedules", total=3,
                  completed=lambda: sum(path.is_file() for path in destinations)):
        pim, serial, pim_only = simulate_all(
            trace, independent, library=library, config=config, cache=cache)
    if serial.physical_counts != pim.physical_counts:
        raise ValueError("overlap changed the physical transaction counts")
    row = {
        "workload": measured["workload"],
        "scene": measured["scene"],
        "frames": measured["frames"],
        "repeats": measured["repeats"],
        "warmup": measured["warmup"],
        "baseline_ms": measured["baseline_ms"],
        "baseline_implementation": measured["baseline_implementation"],
        "gpu_speedup": baseline_ms / dataflow_ms,
        "gpu_dataflow_ms": dataflow_ms,
        "full_speedup": baseline_ms / (pim.milliseconds / trace_frames),
        "full_ms": pim.milliseconds / trace_frames,
        "pim_speedup": baseline_ms / (pim_only.milliseconds / independent.frames),
        "pim_dataflow_speedup": baseline_ms / (serial.milliseconds / trace_frames),
        "psnr_baseline": measured["psnr_baseline"],
        "ssim_baseline": measured["ssim_baseline"],
        "psnr_dataflow": measured["psnr_dataflow"],
        "ssim_dataflow": measured["ssim_dataflow"],
        "psnr_full": measured["psnr_dataflow"],
        "ssim_full": measured["ssim_dataflow"],
        "trace_frames": trace_frames,
    }
    import json
    measurements = dict(measured, result=row,
        traces=[dict(path=str(captured.path), sha256=captured.sha256,
                     recording_sha256=captured.metadata['recording_sha256'])
                for captured in (trace, independent)])
    import hashlib
    report_key = hashlib.sha256(str(output.resolve()).encode()).hexdigest()[:20]
    samples = Path(cache) / 'measurements' / report_key
    samples.mkdir(parents=True, exist_ok=True)
    (samples / 'measurements.json').write_text(json.dumps(measurements, indent=2), encoding='utf-8')
    _write_report([row], output / "summary.md")
    print(f"Dataflow {row['gpu_speedup']:.4f}x  PIM-only {row['pim_speedup']:.4f}x  "
          f"Serial {row['pim_dataflow_speedup']:.4f}x  Full {row['full_speedup']:.4f}x", flush=True)
    del trace, independent, captured, pim, serial, pim_only, measured
    gc.collect()
    torch.cuda.empty_cache()
    return row


def _run_pair_child(connection, args):
    try:
        connection.send((None, _run_pair(*args[0], **args[1])))
    except BaseException:
        import traceback
        connection.send((traceback.format_exc(), None))
    finally:
        connection.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare inputs or run GSPIM CUDA workloads")
    parser.add_argument("--prepare", action="store_true",
                        help="prepare datasets and check supplied GSPIM checkpoints")
    parser.add_argument("--workload", choices=(*WORKLOAD_NAMES, "all"), default="all")
    parser.add_argument("--scene", choices=(*SCENES, "all"), default="all")
    parser.add_argument("--out", type=Path, help="report directory (default: outputs/<workload>/<scene>)")
    parser.add_argument("--checkpoint", type=Path, help="single-combination model override")
    parser.add_argument("--images", type=Path, help="single-combination held-out image override")
    parser.add_argument("--cache", type=Path, default=CACHE, help="external compilation/download cache")
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=100, help="number of warmup windows")
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
    pairs = tuple(combinations(WORKLOAD_NAMES.get(args.workload, args.workload), args.scene))
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

    from .pim_memory import prepare_backend

    verification: list[dict] = []
    library, config = prepare_backend(ROOT / "third_party/ramulator2", args.cache)
    from .progress import bar
    for pair_index, (workload, scene) in enumerate(pairs):
        print(f"\n[{pair_index + 1}/{len(pairs)}] {DISPLAY_NAMES[workload.name]} / "
              f"{SCENE_DISPLAY_NAMES[scene]}", flush=True)
        output = args.out or ROOT / "outputs"
        if len(pairs) != 1 or args.out is None:
            name = next(key for key, value in WORKLOAD_NAMES.items() if value == workload.name)
            output = output / name / scene
        keyword = dict(frames=args.frames, repeats=args.repeats, warmup=args.warmup,
                       checkpoint=args.checkpoint, images=args.images, cache=args.cache,
                       output=output, library=library, config=config)
        if len(pairs) == 1:
            verification.append(_run_pair(workload, scene, **keyword))
            continue
        import multiprocessing
        context = multiprocessing.get_context("fork")
        receive, send = context.Pipe(duplex=False)
        process = context.Process(target=_run_pair_child, args=(send, ((workload, scene), keyword)))
        process.start()
        send.close()
        try:
            error, row = receive.recv()
        except EOFError as failure:
            process.join()
            raise RuntimeError(f"{workload.name}/{scene} worker exited without a result") from failure
        finally:
            receive.close()
        process.join()
        if error is not None or process.exitcode:
            raise RuntimeError(error or f"{workload.name}/{scene} worker exited with {process.exitcode}")
        verification.append(row)
    _write_verification(verification, args.out or ROOT / "outputs")
    print(f"\nDone {bar(len(pairs), len(pairs))}\n"
          f"{(args.out or ROOT / 'outputs') / 'verification.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
