"""Runtime tests for task visibility, actual counts, and scheduling edges."""

import unittest


class RuntimeTests(unittest.TestCase):
    """Task dependencies and memory visibility govern GPU/PIM execution."""

    def test_task_completion_carries_real_output_and_visibility_is_enforced(self):
        from gspim.hcf import LiveRange, ReorgPurpose
        from gspim.ppim import ProgramId
        from gspim.runtime import MemoryRegion, PIMTaskDescriptor, PIMTaskKind, Runtime, TaskOutcome, VisibilityError

        runtime = Runtime(dies=2)
        source, destination = MemoryRegion("source"), MemoryRegion("destination")
        runtime.gpu_write(source)
        task = PIMTaskDescriptor("select", PIMTaskKind.SELECT, 0, source, destination, ProgramId.TEMP_ACTIVITY, (0,), ReorgPurpose.ACTIVE, (0,))
        with self.assertRaises(VisibilityError):
            runtime.submit(task)
        runtime.writeback(source)
        runtime.submit(task)
        outcome = TaskOutcome(2, (LiveRange("active", 0, 2),), (), ())
        runtime.complete("select", outcome)
        self.assertEqual(runtime.event("select").output_count, 2)
        with self.assertRaises(VisibilityError):
            runtime.gpu_acquire(destination, "select")
        runtime.invalidate(destination)
        runtime.gpu_acquire(destination, "select")

    def test_pim_task_requires_an_explicit_physical_bank_scope(self):
        from gspim.hcf import ReorgPurpose
        from gspim.ppim import ProgramId
        from gspim.runtime import MemoryRegion, PIMTaskDescriptor, PIMTaskKind, Runtime

        runtime = Runtime(dies=1)
        source = MemoryRegion("source")
        runtime.writeback(source)
        descriptor = PIMTaskDescriptor(
            "ambiguous", PIMTaskKind.SELECT, 0, source, MemoryRegion("mask"),
            ProgramId.TEMP_ACTIVITY, (0,), ReorgPurpose.ACTIVE,
        )
        with self.assertRaisesRegex(ValueError, "bank scope"):
            runtime.submit(descriptor)

    def test_ping_pong_batch_buffers_preserve_producer_and_consumer_ownership(self):
        from gspim.scheduling import PingPongBatchBuffers

        buffers = PingPongBatchBuffers[str]()
        first = buffers.pack("batch-0")
        self.assertEqual(first, 0)
        self.assertEqual(buffers.next_slot, 1)
        second = buffers.pack("batch-1")
        self.assertEqual(second, 1)
        self.assertEqual(buffers.consume(first), "batch-0")
        self.assertEqual(buffers.pack("batch-2"), 0)
        with self.assertRaisesRegex(RuntimeError, "still occupied"):
            buffers.pack("batch-3")

    def test_pim_pack_can_complete_while_a_prior_gpu_render_is_live(self):
        from gspim.hcf import ReorgPurpose
        from gspim.runtime import MemoryRegion, PIMTaskDescriptor, PIMTaskKind, Runtime, TaskOutcome

        runtime = Runtime(dies=1)
        source, destination = MemoryRegion("selection"), MemoryRegion("batch")
        runtime.gpu_write(source)
        runtime.writeback(source)
        runtime.submit_gpu_stage("render-0", "S5", ())
        runtime.submit(PIMTaskDescriptor("pack-1", PIMTaskKind.REORG, 0, source, destination, None, (1,), ReorgPurpose.BATCH, (0,)))
        runtime.complete("pack-1", TaskOutcome(1, (), (), ()))
        self.assertIn("render-0", runtime.live_at_completion["pack-1"])
        runtime.complete_gpu_stage("render-0")

    def test_hcf_block_completion_releases_gpu_work_before_final_reorg(self):
        from gspim.hcf import BlockCompletion, LiveRange, ReorgPurpose
        from gspim.ppim import ProgramId
        from gspim.runtime import MemoryRegion, PIMTaskDescriptor, PIMTaskKind, Runtime, TaskOutcome

        runtime = Runtime(dies=1)
        source, destination = MemoryRegion("mask"), MemoryRegion("active")
        runtime.gpu_write(source)
        runtime.writeback(source)
        runtime.submit(PIMTaskDescriptor("select", PIMTaskKind.SELECT, 0, source, source, ProgramId.TEMP_ACTIVITY, (0,), ReorgPurpose.ACTIVE, (3,)))
        runtime.complete("select", TaskOutcome(1, (), (), ()))
        runtime.submit(PIMTaskDescriptor("reorg", PIMTaskKind.REORG, 0, source, destination, None, (0, 1), ReorgPurpose.ACTIVE, (3,), ("select",)))
        block = BlockCompletion(0, 3, 1, LiveRange("active", 0, 1), (7,), (7,))
        event = runtime.complete_block("reorg", block)
        self.assertFalse(runtime.event("reorg").complete)
        runtime.invalidate(destination)
        runtime.gpu_acquire(destination, event.task_id)
        runtime.complete_gpu_stage("s2-b1", "S2", (event.task_id,))
        runtime.complete("reorg", TaskOutcome(1, (LiveRange("active", 0, 1),), (block,), ()))
        self.assertTrue(runtime.event("reorg").complete)

    def test_independent_frames_overlap_pim_preparation_but_serialize_consumers(self):
        from gspim.pim_schedule import frame_pipeline

        def node(kind):
            return dict(earliest=0, duration=1, kind=kind, parents=[], memory=[],
                        publications=[])

        bundle = {
            "nodes": [node("start"), node("consumer"), node("complete-0"),
                      node("complete-1")],
            "windows": [{"width": 1, "start": 0}, {"width": 1, "start": 0}],
            "frame_events": [{"complete": [2]}, {"complete": [3]}],
            "spans": [{"window": 0}, {"window": 1}],
            "ctas": [{"kind": "decoder", "span": 1, "first": 0},
                     {"kind": "consumer", "span": 1, "first": 1}],
            "requests": [],
        }
        frame_pipeline(bundle)
        self.assertNotIn((4, 0), bundle["nodes"][0]["parents"])
        self.assertIn((4, 0), bundle["nodes"][1]["parents"])

    def test_window_latency_keeps_early_pim_preparation_outside_ready_time(self):
        import numpy as np
        from gspim.pim import _frame_times

        bundle = {
            "windows": [
                {"start": 0, "first_frame": 0, "width": 2},
                {"start": 1, "first_frame": 2, "width": 2},
            ],
            "frame_events": [
                {"complete": [2]}, {"complete": [3]},
                {"complete": [4]}, {"complete": [5]},
            ],
        }
        times = np.asarray([
            [0, 0], [2, 2], [0, 10], [0, 12], [0, 11], [0, 18],
        ], dtype=np.int64)
        submitted, completed = _frame_times(bundle, times)
        np.testing.assert_array_equal(submitted, [0, 0, 12, 12])
        np.testing.assert_array_equal(completed, [10, 12, 12, 18])

    def test_trace_discovery_skips_invalid_external_cache(self):
        import os
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        from gspim.demo import _load_registered_trace

        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            traces = cache / "pim" / "traces"
            traces.mkdir(parents=True)
            invalid = traces / "ex4dgs-coffee_martini-new.trace.npz"
            valid = traces / "ex4dgs-coffee_martini-old.trace.npz"
            invalid.touch()
            valid.touch()
            os.utime(invalid, (2, 2))
            os.utime(valid, (1, 1))
            expected = SimpleNamespace(frames=10, metadata={"input_sha256": "inputs"})

            def load(path):
                if path == invalid:
                    raise ValueError("invalid trace")
                return expected

            with patch("gspim.demo.load_trace", side_effect=load), \
                    patch("gspim.demo.load_or_build") as build:
                actual = _load_registered_trace(
                    "ex4dgs", "coffee_martini", 10, "inputs", cache=cache)

            self.assertIs(actual, expected)
            build.assert_not_called()
