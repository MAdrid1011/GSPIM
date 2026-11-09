"""Record native GPU services and block publications for PIM scheduling."""

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from math import isfinite
from pathlib import Path
from statistics import median
from types import SimpleNamespace

import torch

from .pipeline import CUDAPipeline
from .renderer import load_kernel
from .pim_record_ops import (ColorFieldBlocks, CompactFields, CrossingShortcut,
    DepthStatistics, HCFNativeSupply, SegmentedAoS, SourceOps, SpatialExpandedTiles,
    SweptGPU, SweptMasks, render_window)
from .pim_record_models import AnchoredCapture, ExplicitCapture, Ex4DGSCapture
from .pim_trace import input_identity




@dataclass
class ReadyWindow:
    result: object
    calls: list
    ready: object
    done: object = None


def _windows(frames, width):
    start = 0
    while start < frames:
        size = min(width, frames - start)
        if size > 1 and frames - start - size == 1:
            size -= 1
        yield start, size
        start += size


class DeferredTiles:
    """Keep source IDs while submitting complete tile spans to consumers."""

    def __init__(self, source):
        self.source = source
        self.calls = []
        self.submit = None

    def __getattr__(self, name):
        if name == 'packed_layout':
            return lambda *args: None
        return getattr(self.source, name)

    def batches(self, *args):
        args = list(args)
        assert args[2].numel()
        args[12] = args[0].shape[1].bit_length()
        args[17] = []
        args[20] = args[20].new_empty(0)
        if not args[21].numel():
            args[21] = args[0].new_empty((len(args[0]), args[7], args[6], 4))
        self.calls.append(args)
        if self.submit is not None:
            self.submit(args)
        return args[21][..., :3]


class GPURecorder:
    """Publish GPU fields and bind their consumers to the shared Dataflow path."""

    def __init__(self, model, workload, cache):
        self.model, self.workload = model, workload
        self.pipeline = CUDAPipeline(model, cache)
        self.native = self.pipeline.renderer.ops
        kernel = lambda name: load_kernel(name, cache)
        self.stamps = kernel('depth_publication')
        self.mean = kernel('mean')
        self.projection = kernel('projection')
        self.finish = kernel('projection_finish') if workload == 'ex4dgs' else None
        self.depth = (self.pipeline.renderer.window_statistics if workload == 'ex4dgs' else
                      DepthStatistics(kernel('depth'), self.native))
        self.masks = SweptMasks(kernel('masks'), SimpleNamespace(
            window_delta=lambda score: self.mean.delta(score)[0]), False)
        source = SourceOps(self.native, kernel('source_tiles'))
        self.spatial = SpatialExpandedTiles(source, kernel('rectangles'), kernel('spatial_routes'))
        self.supply = HCFNativeSupply(self.spatial, self.masks)
        self.supply.cull_stream = torch.cuda.Stream(priority=-1)
        self.consumer = SegmentedAoS(CompactFields(kernel('segments'), kernel('compact_fields')))
        self.consumer.spatial_source = self.spatial
        self.tiles = DeferredTiles(self.supply)
        self.crossing = CrossingShortcut(kernel('crossings'), self.native)
        self.fields = ColorFieldBlocks(self.pipeline.renderer, self.consumer, kernel(f'colors_{workload}'))
        self.consumer.begin_fields = self.fields.fields
        self.adapter = None
        self.original_generate = self.native.generate
        if workload == 'explicit4d':
            self.depth.generation_binary = kernel(f'{workload}_generation')
            self.depth.generation_trace_binary = self.depth.generation_binary
            self.depth.generation_view = model.view
            self.depth.generation_span = (float(model.config.get('stability_far', model.config['far'])) -
                                          float(model.config.get('stability_near', model.config['near'])))
            self.depth.generation_enabled = True
            self.native.generate = self.depth.generate
        if workload == 'ex4dgs':
            self.adapter = Ex4DGSCapture(self.pipeline.renderer, self.depth)
        elif workload == 'explicit4d':
            self.adapter = ExplicitCapture(self.pipeline.renderer, self.depth)
        elif workload == 'anchored':
            self.adapter = AnchoredCapture(self.pipeline.renderer, self.depth, model,
                                           self.stamps.stamp, kernel('anchored_generation'))
        self.pipeline.renderer.ops = self.tiles
        self.preparation = torch.cuda.Stream(priority=-1)
        self.pipeline.projection_stream = torch.cuda.Stream(priority=-1)
        self.render = torch.cuda.Stream()
        self.lanes = [torch.cuda.Stream() for _ in range(2)]
        self.recording = False
        self.starts, self.phases = [], []
        self.phase_ctas = {'projection': [], 'window_delta': []}
        if self.finish is not None:
            self.phase_ctas['projection_finish'] = []

    def mark(self, kind, function, *args, **kwargs):
        if not self.recording:
            return function(*args, **kwargs)
        before = self.stamps.stamp(self.times)
        value = function(*args, **kwargs)
        after = self.stamps.stamp(self.times)
        self.phases.append((kind, len(self.starts) - 1, before, after))
        return value

    def remember(self, means, projected):
        self.fields.remember(means, projected)

    def colors(self, *args):
        return self.mark('colors', self.fields.colors, *args)

    def publish(self, projected):
        self.consumer.begin_window_fields(projected)

    def visible(self, *args):
        return self.mark('crossing', self.crossing.visible, *args)

    def span(self, *args):
        return self.mark('sort', self.crossing.span, *args)

    def orders(self, means, view, near, far, operators):
        if not isfinite(near) or not isfinite(far) or far <= near:
            raise ValueError('depth stability requires finite, increasing camera bounds')
        depth, score, minimum = self.depth.run(means, view, far - near)
        width, count = depth.shape
        stable, unstable, adjustment = self.mark('window_delta', self.classify, score)
        if self.crossing.defer(stable, count, minimum):
            self.supply.sorting_inputs(stable, minimum)
            return stable.long()[None].expand(width, -1), adjustment, stable, depth
        def sort(values, keys):
            return operators.radix(keys.contiguous(), values.contiguous(), 0, 32)[1]
        def mixed():
            ordered = sort(*self.supply.sorting_inputs(stable, minimum))
            per_frame = torch.stack([sort(*self.supply.sorting_inputs(unstable, depth[frame]))
                                     for frame in range(width)]).contiguous()
            return operators.merge(ordered, per_frame, minimum.contiguous(), depth), ordered
        orders, stable = self.mark('mixed_sort', mixed)
        return orders, adjustment, stable, depth

    def classify(self, score):
        if not self.recording:
            return self.supply.classify(score)
        stable, unstable, expected = self.supply.classes[self.supply.class_cursor]
        delta, ctas = self.mean.delta(score)
        self.phase_ctas['window_delta'].append(ctas)
        assert delta == expected
        self.supply.class_cursor += 1
        return stable, unstable, delta

    def render_window(self, means, projected, orders, stable, shared_tiles,
                      output=None, visible_ids=None, wait_for=None, model=None):
        if not projected[4].numel():
            colors = self.colors(model or self.model, means, projected[0])
            projected = (*projected[:4], colors, projected[5])
        self.consumer.begin_fields(projected)
        if not shared_tiles:
            self.consumer.begin_window_fields(projected)
        return render_window(self.pipeline, means, projected, orders, stable, shared_tiles,
                             output, visible_ids, wait_for, model, self.consumer.begin_fields)

    def sequence(self, frames, width, *, prepare=False, lanes=2, slots=4):
        self.times = torch.arange(frames, device=self.model.xyz.device, dtype=torch.float32)
        if self.adapter is not None:
            self.adapter.reset()
        self.consumer.reset()
        self.fields.reset()
        self.supply.reset(prepare)
        origin = torch.cuda.Event()
        origin.record()
        self.preparation.wait_event(origin)
        self.render.wait_event(origin)
        for lane in self.lanes:
            lane.wait_event(origin)
        active = [None] * slots
        outputs, decisions, submitted = [], [], []
        def submit(args):
            self.consumer.producer_completion = self.supply.cull_completions[-1]
            self.consumer.prepare_gpu(args)
            ready = torch.cuda.Event()
            ready.record()
            lane = self.lanes[(index + len(submitted)) % lanes]
            with torch.cuda.stream(lane):
                lane.wait_event(ready)
                for value in args:
                    if isinstance(value, torch.Tensor) and value.is_cuda:
                        value.record_stream(lane)
                self.consumer.batches(*args)
                done = torch.cuda.Event()
                done.record()
                submitted.append(done)
        for index, (start, size) in enumerate(_windows(frames, width)):
            slot = index % slots
            with torch.cuda.stream(self.preparation):
                if active[slot] is not None:
                    self.preparation.wait_event(active[slot].done)
                if self.recording:
                    self.starts.append(self.stamps.stamp(self.times))
                self.tiles.calls = []
                self.tiles.submit = None if prepare else submit
                result = self.pipeline.dataflow(self.times[start:start + size], start, execution=self)
                self.tiles.submit = None
                if prepare:
                    for args in self.tiles.calls:
                        self.consumer.prepare(args)
                ready = torch.cuda.Event()
                ready.record()
                current = ReadyWindow(result, self.tiles.calls, ready)
            with torch.cuda.stream(self.render):
                self.render.wait_event(ready)
                for done in submitted:
                    self.render.wait_event(done)
                submitted.clear()
                current.done = torch.cuda.Event()
                current.done.record()
            active[slot] = current
            outputs.append(result[0])
            decisions.append(result[1])
        torch.cuda.current_stream().wait_event(current.done)
        for event in self.supply.cull_completions:
            torch.cuda.current_stream().wait_event(event)
        return outputs, decisions

    def observed_projection(self, model, geometry, defer_colors=False):
        assert defer_colors
        means, covariance, opacity = geometry
        width = len(means)
        views = model.view[None].expand(width, -1, -1, -1).contiguous()
        intrinsics = model.intrinsics[None].expand(width, -1, -1, -1).contiguous()
        radii, xy, depth, conics, compensation, ctas = self.projection.run(
            means, covariance, opacity, views, intrinsics, model.width, model.height,
            float(model.config['kernel_size']), float(model.config['near']), float(model.config['far']),
            self.workload == 'ex4dgs')
        self.phase_ctas['projection'].append(ctas)
        if self.finish is not None:
            before = self.stamps.stamp(self.times)
            opacity, ctas = self.finish.run(xy, radii, opacity, compensation, model.width, model.height)
            after = self.stamps.stamp(self.times)
            self.phase_ctas['projection_finish'].append(ctas)
            self.phases.append(('projection_finish', len(self.starts) - 1, before, after))
        return radii, xy, depth, conics, means.new_empty(0), opacity

    @contextmanager
    def instrument(self):
        renderer = self.pipeline.renderer
        methods = [(renderer, name, getattr(renderer, name)) for name in ('generate', 'project')]
        methods.append((self.native, 'generate', self.native.generate))
        renderer.project = self.observed_projection
        for target, name, method in methods[:2]:
            setattr(target, name, lambda *args, _kind=name, _fn=getattr(target, name), **kwargs:
                    self.mark(_kind, _fn, *args, **kwargs))
        self.recording = True
        self.consumer.trace = True
        self.depth.trace_binary = self.stamps
        self.depth.trace_rows = []
        try:
            yield
        finally:
            torch.cuda.synchronize()
            self.recording = False
            self.consumer.trace = False
            self.depth.trace_binary = None
            for target, name, method in reversed(methods):
                setattr(target, name, method)

    def clear_observations(self):
        self.starts.clear()
        self.phases.clear()
        self.depth.trace_rows.clear()
        for rows in self.phase_ctas.values():
            rows.clear()

    def close(self):
        renderer = self.pipeline.renderer
        renderer.ops = self.native
        self.native.generate = self.original_generate
        if self.adapter is not None:
            renderer.generate = self.adapter.original_generate
            if self.workload != 'ex4dgs':
                name = 'generate_4d' if self.workload == 'explicit4d' else 'generate_anchored'
                setattr(self.native, name, self.adapter.original)


def _measure(function, repeats):
    images, milliseconds = None, []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        images = function()
        end.record()
        end.synchronize()
        milliseconds.append(start.elapsed_time(end))
    return images, median(milliseconds)


def record(model, workload, checkpoint, images, cache, frames=10, *, independent=False,
           repeats=4, warmup=20):
    """Record services, publications and exact scheduling-image checks."""
    from PIL import Image
    import numpy as np
    from .renderer import cuda_image_quality

    width = 1 if independent else min(5, frames)
    windows = list(_windows(frames, width))
    recorder = GPURecorder(model, workload, cache)
    pipeline = recorder.pipeline
    times = torch.arange(frames, device=model.xyz.device, dtype=torch.float32)
    try:
        with torch.inference_mode():
            # Establish the exact depth instruction variant and physical layouts.
            recorder.sequence(frames, width, prepare=True)
            recorder.depth.validate = False
            for _ in range(max(1, (warmup + len(windows) - 1) // len(windows))):
                recorder.sequence(frames, width)
            with recorder.instrument():
                for _ in range(2):
                    recorder.sequence(frames, width)
                    torch.cuda.synchronize()
                    recorder.clear_observations()
                (captured_rgb, decisions), capture_ms = _measure(lambda: recorder.sequence(frames, width), 1)
                captured_rgb = [image.clone() for image in captured_rgb]
            segment_evidence = recorder.consumer.evidence()
            recorder.supply.validate_sources()
            rectangle_evidence = recorder.spatial.evidence()
            spatial_evidence = recorder.spatial.spatial_evidence(recorder.supply.sources)
            color_publications = [value.cpu() for value in recorder.fields.publications]
            color_loads = [value.cpu() for value in recorder.fields.load_frontiers]
            # Scheduling resources may overlap without changing any image or decision.
            serial_rgb, serial_decisions = recorder.sequence(frames, width, lanes=1, slots=2)
            torch.cuda.synchronize()
            if decisions != serial_decisions or any(not torch.equal(a, b) for a, b in zip(captured_rgb, serial_rgb)):
                raise ValueError('GPU scheduling changed images or window decisions')
            recorder.close()
            reference = [pipeline.dataflow(times[first:first + size], first)
                         for first, size in windows]
            reference_rgb = [value[0].clone() for value in reference]
            assert decisions == [value[1] for value in reference]
            if any(not torch.equal(actual, expected) for actual, expected in zip(captured_rgb, reference_rgb)):
                raise ValueError('PIM recording changed GPU Dataflow images')
            pipeline.renderer.ops = SweptGPU(recorder.masks, recorder.native)
            shared = [pipeline.dataflow(times[first:first + size], first)
                      for first, size in windows]
            shared_error = max(float((a[0] - b).abs().max()) for a, b in zip(shared, reference_rgb))
            if shared_error != 0:
                raise ValueError('shared GPU culling changed reference images')
            pipeline.renderer.ops = recorder.native
            baseline = lambda: [pipeline.baseline(times[frame:frame + 1], frame) for frame in range(frames)]
            dataflow = lambda: [pipeline.dataflow(times[first:first + size], first)
                                for first, size in windows]
            baseline_rgb, baseline_ms = _measure(baseline, repeats)
            _, dataflow_ms = _measure(dataflow, repeats)
            quality = []
            for window, (actual, reference) in enumerate(zip(captured_rgb, reference_rgb)):
                for local in range(len(actual)):
                    frame = windows[window][0] + local
                    target = Image.open(Path(images) / f'{getattr(model, "image_prefix", "")}{frame:04d}.png').convert('RGB')
                    if target.size != (model.width, model.height):
                        target = target.resize((model.width, model.height), Image.Resampling.BILINEAR)
                    target = torch.from_numpy(np.asarray(target).copy()).to(model.xyz.device, dtype=torch.float32) / 255
                    before = cuda_image_quality(reference[local], target)
                    after = cuda_image_quality(actual[local], target)
                    quality.append(dict(frame=frame, reference=cuda_image_quality(baseline_rgb[frame], target),
                                        dataflow=before, candidate=after))
            rows = [dict(depth=row[0].cpu(), score=row[1].cpu(), minimum=row[2].cpu(),
                         publication=row[3][:, :2].cpu(), window_start=int(start.cpu()))
                    for row, start in zip(recorder.depth.trace_rows, recorder.starts)]
            phases = [dict(kind=kind, window=window, start_ns=int(a.cpu()), end_ns=int(b.cpu()))
                      for kind, window, a, b in recorder.phases]
            for mixed in [value for value in phases if value['kind'] == 'mixed_sort']:
                if any(value['window'] == mixed['window'] and value['kind'] == 'sort'
                       for value in phases):
                    continue
                phases.remove(mixed)
                phases.append(dict(mixed, kind='crossing', end_ns=mixed['start_ns']))
                for frame in range(len(rows[mixed['window']]['depth'])):
                    phases.append(dict(mixed, kind='sort', shared_sort=frame > 0))
            representation = recorder.adapter.source()
            attributes = representation.pop('attributes', {})
            generation, loads = recorder.adapter.records, recorder.adapter.loads
            phase_ctas = {kind: [value.cpu() for value in values] for kind, values in recorder.phase_ctas.items()}
            phase_ctas['colors'] = color_publications
            resources = dict(projection=recorder.projection.resources(), window_delta=recorder.mean.resources(),
                             colors=recorder.fields.binary.resources())
            if recorder.finish is not None:
                resources['projection_finish'] = recorder.finish.resources()
            supply = dict(windows=rows, phases=phases, generation_ctas=[value.cpu() for value in generation],
                generation_resources=(recorder.adapter.binary.resources() if workload == 'anchored' else
                                      recorder.depth.generation_trace_binary.launch_resources()),
                attributes=attributes, generation_layout='records', fused_generation_depth=workload != 'anchored',
                **representation, checkpoint_sha256=input_identity(checkpoint, images, model, frames)['checkpoint'],
                block_records=256, trace_rgb_equal=True, phase_ctas=phase_ctas, phase_resources=resources,
                phase_threads={kind: recorder.mean.threads() if kind == 'window_delta' else 192 if kind == 'colors' else 256
                               for kind in phase_ctas},
                color_load_frontiers=color_loads)
            if workload == 'anchored':
                supply.update(depth_ctas=[row[3].cpu() for row in recorder.depth.trace_rows],
                              depth_resources=recorder.stamps.resources(), decoded_color_fields=True)
            else:
                supply['generation_load_frontiers'] = [value.cpu() for value in loads]
            folder = Path(__file__).parent
            hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in (folder / 'pim_record.py', folder / 'pim_record_ops.py',
                                   folder / 'pim_record_models.py', folder / 'pipeline.py', folder / 'renderer.py', folder / 'renderer.cu',
                                   folder.parent / 'integrations' / f'{workload}.py',
                                   *sorted((folder / 'pim_kernels').iterdir())) if path.is_file()}
            return dict(scene=Path(images).parent.name, frames=frames, window_slots=4, baseline_ms=baseline_ms,
                durations_ms={'native_shared': dataflow_ms},
                configuration=dict(first=0, frames=frames, window_width=width, representation=workload,
                    consumer_reserve=0, compact_consumer_fields=True, segment_cta_warps=4),
                hashes=hashes, shared_gpu_rgb_error=shared_error, scheduling_rgb_error=0.,
                quality=quality, decisions=decisions, segmented_supply=segment_evidence,
                rectangle_supply=rectangle_evidence, spatial_expansion=spatial_evidence,
                pim_sources=recorder.supply.sources, decision_supply=supply,
                gpu_resources=dict(consumer=recorder.consumer.binary.launch_resources(),
                    cull=recorder.masks.binary.launch_resources(), consumer_threads=128, cull_threads=256),
                trace_capture_ms=capture_ms, input_sha256=input_identity(checkpoint, images, model, frames),
                candidate_rgb_sha256=[hashlib.sha256(image.cpu().contiguous().numpy().tobytes()).hexdigest()
                                      for window in captured_rgb for image in window])
    finally:
        recorder.close()
