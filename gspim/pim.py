"""Run captured GPU publications and PPIM/HCF requests on a shared timeline."""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import ast
import hashlib
import os
import time

import numpy as np

from .pim_memory import FeedbackGraph, MemoryBackend
from .pim_schedule import (Resources, block_dependencies, field_queues, frame_pipeline,
                           prepare, release_fields, serial_boundaries, validate_staging)

SCHEDULE_ABI = b'gspim-schedule-2'


@dataclass
class CapturedTrace:
    path: Path
    sha256: str
    metadata: dict
    _bundle: dict | None = None

    @property
    def bundle(self):
        if self._bundle is None:
            from .pim_trace import read_trace
            self._bundle = read_trace(self.path)
        return self._bundle

    @property
    def frames(self):
        return len(self.metadata['frame_events'])


@dataclass
class ScheduleResult:
    milliseconds: float
    latency_ms: np.ndarray
    submitted_ms: np.ndarray
    completed_ms: np.ndarray
    physical_counts: dict
    staging: list
    resources: dict
    capture_sha256: str
    trace_sha256: str
    wall_seconds: float
    task_times: np.ndarray

    @property
    def p95_ms(self):
        return float(np.percentile(self.latency_ms, 95))

    @property
    def p99_ms(self):
        return float(np.percentile(self.latency_ms, 99))


@contextmanager
def _controller(lanes, spans):
    windows = [span['window'] for span in spans]
    if not windows or 10 * len(windows) >= 256:
        raise ValueError('trace spans exceed the controller lane namespace')
    options = {
        'PIM_STAGING_BYTES': '16384', 'PIM_WINDOW_PRIORITY': str(len(windows)),
        'PIM_SPAN_WINDOWS': ','.join(map(str, windows)),
        'PIM_AUXILIARY_LANE_START': str(10 * len(windows)),
        'PIM_PARALLEL_ALU': '1', 'PIM_GPU_SOURCE_ORDER': '1',
        'PIM_ROW_HIT_FIRST': '1', 'PIM_READY_DEPENDENCIES': '1',
        'PIM_ACTUAL_GPU_SUBMISSIONS': '1', 'PIM_MEMORY_WINDOW': '128',
        'PIM_BANK_SUBMISSIONS': '1', 'PIM_FIELD_BANK_LANES': ','.join(map(str, lanes)),
    }
    previous = {key: os.environ.get(key) for key in options}
    os.environ.update(options)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def load_trace(path):
    """Load a capture with explicit block, span and per-frame event mappings."""
    from .pim_trace import read_metadata

    path = Path(path)
    bundle = read_metadata(path)
    digest = bundle.get('capture_sha256')
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f'trace has no GPU capture digest: {path}')
    required = ('windows', 'spans', 'field_producers', 'block_dependencies', 'frame_events')
    missing = [key for key in required if not bundle.get(key)]
    if missing:
        raise ValueError(f'trace is missing captured event mappings: {", ".join(missing)}')
    if not bundle['ctas']:
        raise ValueError('captured requests and GPU resource descriptions are required')
    if len(bundle['frame_events']) != bundle['measurements']['frames']:
        raise ValueError('frame events and GPU measurements disagree')
    with path.open('rb') as stream:
        trace_digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return CapturedTrace(path, trace_digest, bundle)


def load_or_build(workload, scene, root, frames=10, *, independent=False,
                  model=None, checkpoint=None, images=None, cache=None):
    """Reuse matching traces or record the selected GPU path into the cache."""
    suffix = '.frame.trace.npz' if independent else '.trace.npz'
    path = Path(root) / 'traces' / workload / f'{scene}{suffix}'
    identity = None
    if model is not None:
        from .pim_trace import input_identity
        identity = input_identity(checkpoint, images, model, frames)
    if identity is None:
        raise FileNotFoundError(f'matching PIM inputs are required to build a trace: {path}')

    import torch
    source = Path(__file__).parent
    key = hashlib.sha256(repr((workload, scene, frames, independent, identity,
        torch.__version__, torch.version.cuda, torch.cuda.get_device_name(),
        torch.cuda.get_device_capability())).encode())
    for name in ('pim_capture.py', 'pim_record.py', 'pim_record_ops.py',
                 'pim_record_models.py', 'pim_requests.py', 'pim_trace.py',
                 'pipeline.py', 'dataflow.py', 'renderer.py', 'renderer.cu'):
        key.update((source / name).read_bytes())
    key.update((source.parent / 'integrations' / f'{workload}.py').read_bytes())
    for kernel in sorted((source / 'pim_kernels').iterdir()):
        key.update(kernel.read_bytes())
    recording_key = key.hexdigest()
    if path.is_file():
        trace = load_trace(path)
        if (trace.frames == frames and trace.metadata.get('input_sha256') == identity and
                trace.metadata.get('recording_sha256') == recording_key):
            if independent and any(window['width'] != 1 for window in trace.metadata['windows']):
                raise ValueError(f'independent-frame PIM requires single-frame windows: {path}')
            return trace
    if cache is None:
        raise FileNotFoundError(f'matching PIM inputs are required to build a trace: {path}')

    import gc
    import tempfile
    from .pim_capture import build_trace
    from .pim_record import record
    from .pim_trace import write_trace

    folder = Path(cache) / 'pim' / 'traces'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{workload}-{scene}-{recording_key}{suffix}'
    if not path.is_file():
        print(f'Recording {workload}/{scene} ({frames} frames, '
              f'{"independent" if independent else "Dataflow"})', flush=True)
        capture = record(model, workload, checkpoint, images, cache, frames,
                         independent=independent)
        gc.collect()
        torch.cuda.empty_cache()
        with tempfile.TemporaryDirectory(prefix='capture-', dir=folder) as temporary:
            raw = Path(temporary) / 'capture.pth'
            torch.save(capture, raw)
            with raw.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            bundle = build_trace(capture, digest)
            bundle['recording_sha256'] = recording_key
            write_trace(bundle, path)
        del bundle, capture
        gc.collect()
    trace = load_trace(path)
    if (trace.frames != frames or trace.metadata.get('input_sha256') != identity or
            trace.metadata.get('recording_sha256') != recording_key):
        raise ValueError(f'cached PIM trace does not match the requested inputs: {path}')
    if independent and any(window['width'] != 1 for window in trace.metadata['windows']):
        raise ValueError(f'independent-frame PIM requires single-frame windows: {path}')
    return trace


def _schedule_destination(trace, library, config, overlap, cache, *, frame_order=False):
    if cache is None:
        return None
    source = Path(__file__)
    mode = str((overlap, frame_order)) if frame_order else str(overlap)
    digest = hashlib.sha256(SCHEDULE_ABI + (trace.sha256 + mode).encode())
    for path in (source.with_name('pim_memory.py'), source.with_name('pim_schedule.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        digest.update(ast.dump(tree, include_attributes=False).encode())
    for path in (Path(library), Path(config)):
        digest.update(path.read_bytes())
    return Path(cache) / 'pim' / 'schedules' / f'{digest.hexdigest()}.npz'


def _clone_bundle(bundle):
    clone = dict(bundle)
    clone['nodes'] = [dict(node, parents=list(node['parents']), memory=list(node['memory']),
                                  publications=list(node['publications']))
                      for node in bundle['nodes']]
    return clone


def _prepare_bundle(bundle):
    release_fields(bundle)
    queues = field_queues(bundle)
    work = _prepare_bundle_after_queues(bundle)
    return queues, work


def _prepare_bundle_after_queues(bundle):
    block_dependencies(bundle)
    if calibration := bundle.get('mean_submission'):
        resource = bundle['resources']['window_delta']
        assert resource == calibration['resources']
        assert bundle['resources']['window_delta_threads'] == calibration['threads']
        gap = (int(calibration['submit_gap_ns']) * 4 + 4) // 5
        for cta in bundle['ctas']:
            if cta['kind'] == 'window_delta':
                parents = bundle['nodes'][cta['first']]['parents']
                match = [i for i, (parent, _) in enumerate(parents)
                         if bundle['nodes'][parent]['kind'] == 'window_delta_start']
                assert len(match) == 1
                i = match[0]
                parents[i] = parents[i][0], gap
    work = prepare(bundle)
    bundle['_prepared'] = True
    return work


def _frame_times(bundle, times):
    """Return ready-window submission and per-frame completion cycles."""
    completed = []
    for frame in bundle['frame_events']:
        assert frame['complete']
        completed.append(max(int(times[node, 1]) for node in frame['complete']))
    completed = np.asarray(completed, dtype=np.int64)
    submitted = np.empty(len(completed), dtype=np.int64)
    previous_completion = 0
    for window in bundle['windows']:
        first = int(window['first_frame'])
        last = first + int(window['width'])
        ready = max(int(times[window['start'], 0]), previous_completion)
        submitted[first:last] = ready
        completed[first:last] = np.maximum(completed[first:last], ready)
        previous_completion = int(completed[first:last].max())
    return submitted, completed


def _advance_schedule(trace, bundle, work, backend, started, workers):
    resources = Resources(bundle, work)
    graph = FeedbackGraph(backend)
    graph.nodes = bundle['nodes']
    outputs, feedback = graph.run(bundle['requests'], quantum=800,
                                  exact_gpu_events=True, resources=resources, workers=workers)
    staging = [validate_staging(requests, done) for requests, (done, _) in
               zip(bundle['requests'], outputs)]
    counts = {key: sum(stats[key] for _, stats in outputs[::2])
              for key in ('gpu_reads', 'gpu_writes', 'local_reads', 'local_writes')}
    expected = {key: sum(int(np.count_nonzero(stream['kind'] == kind))
                        for stream in bundle['requests'])
                for kind, key in enumerate(('gpu_reads', 'gpu_writes', 'local_reads', 'local_writes'))}
    assert counts == expected
    times = feedback['task_times']
    submitted, completed = _frame_times(bundle, times)
    submitted = submitted * 1.25e-6
    completed = completed * 1.25e-6
    assert np.all(completed >= submitted)
    milliseconds = feedback['total_cycles'] * 1.25e-6
    assert milliseconds >= completed.max()
    result = ScheduleResult(milliseconds, completed - submitted, submitted, completed,
                            counts, staging, feedback['resource_dispatch'], bundle['capture_sha256'],
                            trace.sha256, time.perf_counter() - started, times)
    return result


def simulate(trace, *, library, config, workers=8, overlap=True, frame_order=False,
             cache=None):
    """Advance finite GPU and memory resources through final drain."""
    destination = _schedule_destination(trace, library, config, overlap, cache,
                                        frame_order=frame_order)
    if destination is not None and destination.is_file():
        return _read_schedule(destination, trace)
    bundle = trace.bundle
    if bundle.get('_prepared'):
        raise ValueError('load a fresh trace before running a second schedule')
    started = time.perf_counter()
    if frame_order:
        if not overlap:
            raise ValueError('frame ordering retains overlap within each frame')
        frame_pipeline(bundle)
    elif not overlap:
        serial_boundaries(bundle)
    queues, work = _prepare_bundle(bundle)
    backend = MemoryBackend(library, config)
    with _controller(queues['lanes'], bundle['spans']):
        result = _advance_schedule(trace, bundle, work, backend, started, workers)
    if destination is not None:
        _write_schedule(destination, result)
    return result


def simulate_dataflow_pair(trace, *, library, config, workers=8, cache=None):
    """Run overlapping and serial schedules from one expanded Dataflow trace."""
    destinations = tuple(_schedule_destination(trace, library, config, overlap, cache)
                         for overlap in (True, False))
    cached = tuple(_read_schedule(path, trace) if path is not None and path.is_file() else None
                   for path in destinations)
    if all(result is not None for result in cached):
        return cached
    if any(result is not None for result in cached):
        missing = cached.index(None)
        result = simulate(trace, library=library, config=config, workers=workers,
                          overlap=missing == 0, cache=cache)
        values = list(cached)
        values[missing] = result
        return tuple(values)

    from concurrent.futures import ThreadPoolExecutor

    overlap_bundle = trace.bundle
    if overlap_bundle.get('_prepared'):
        raise ValueError('load a fresh trace before running a second schedule')
    serial_bundle = _clone_bundle(overlap_bundle)
    serial_boundaries(serial_bundle)
    release_fields(overlap_bundle)
    release_fields(serial_bundle)
    queues = field_queues(overlap_bundle)
    overlap_work = _prepare_bundle_after_queues(overlap_bundle)
    serial_work = _prepare_bundle_after_queues(serial_bundle)
    backend = MemoryBackend(library, config)
    started = time.perf_counter()
    with _controller(queues['lanes'], overlap_bundle['spans']):
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = (
                executor.submit(_advance_schedule, trace, overlap_bundle,
                                overlap_work, backend, started, workers),
                executor.submit(_advance_schedule, trace, serial_bundle,
                                serial_work, backend, started, workers),
            )
            results = tuple(future.result() for future in futures)
    for destination, result in zip(destinations, results):
        if destination is not None:
            _write_schedule(destination, result)
    return results


def _simulate_independent(path, library, config, cache, workers):
    trace = load_trace(path)
    simulate(trace, library=library, config=config, workers=workers,
             overlap=True, frame_order=True, cache=cache)
    return str(_schedule_destination(trace, library, config, True, cache,
                                     frame_order=True))


def simulate_all(trace, independent, *, library, config, workers=8, cache=None):
    """Run the three ablation schedules concurrently when their caches are cold."""
    destination = _schedule_destination(independent, library, config, True, cache,
                                        frame_order=True)
    if destination is None or destination.is_file():
        pim, serial = simulate_dataflow_pair(
            trace, library=library, config=config, workers=workers, cache=cache)
        pim_only = simulate(independent, library=library, config=config, workers=workers,
                            overlap=True, frame_order=True, cache=cache)
        return pim, serial, pim_only

    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    context = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
        future = executor.submit(_simulate_independent, str(independent.path), str(library),
                                 str(config), str(cache), workers)
        pim, serial = simulate_dataflow_pair(
            trace, library=library, config=config, workers=workers, cache=cache)
        completed = Path(future.result())
    if completed != destination or not completed.is_file():
        raise RuntimeError('independent-frame schedule did not produce its cache')
    return pim, serial, _read_schedule(completed, independent)


def _write_schedule(destination, result):
    import pickle
    import zlib
    from .pim_trace import _plain

    metadata = {key: value for key, value in vars(result).items() if not isinstance(value, np.ndarray)}
    values = {key: value for key, value in vars(result).items() if isinstance(value, np.ndarray)}
    values['metadata'] = np.frombuffer(zlib.compress(pickle.dumps(_plain(metadata), protocol=5)), dtype=np.uint8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.npz.part')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(destination)


def _read_schedule(source, trace):
    import io
    import zlib
    from .pim_trace import _MetadataReader

    with np.load(source, allow_pickle=False) as values:
        metadata = _MetadataReader(io.BytesIO(zlib.decompress(values['metadata'].tobytes()))).load()
        result = ScheduleResult(**metadata, **{key: values[key].copy() for key in
            ('latency_ms', 'submitted_ms', 'completed_ms', 'task_times')})
    submitted, completed = _frame_times(trace.metadata, result.task_times)
    result.submitted_ms = submitted * 1.25e-6
    result.completed_ms = completed * 1.25e-6
    result.latency_ms = result.completed_ms - result.submitted_ms
    if (result.trace_sha256 != trace.sha256 or result.capture_sha256 != trace.metadata['capture_sha256'] or
            len(result.completed_ms) != trace.frames or not np.isfinite(result.completed_ms).all() or
            np.any(result.completed_ms < result.submitted_ms) or
            not np.array_equal(result.latency_ms, result.completed_ms-result.submitted_ms) or
            result.milliseconds < result.completed_ms.max()):
        raise ValueError(f'invalid cached PIM schedule: {source}')
    return result
