"""Runtime tests for task visibility, actual counts, and scheduling edges."""

import unittest


class RuntimeTests(unittest.TestCase):
    """The runtime is a dependency protocol rather than a latency model."""

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
