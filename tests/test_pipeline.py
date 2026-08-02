"""End-to-end fixed-fixture comparison against the independent reference model."""

import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tests.fixtures import ANCHOR_PAYLOADS, ANCHORS, CAMERA, EX4DGS_RECORDS, EXPLICIT_RECORDS, TIMESTAMPS


class PipelineTests(unittest.TestCase):
    """All five stages, HCF results, trace edges, and images agree with the oracle."""

    def test_explicit_4d_pipeline_matches_reference_on_five_frame_window(self):
        from gspim.layouts.explicit4d import Explicit4DLayout
        from gspim.model import Camera, FrameRequest
        from gspim.pipeline import GSPIMPipeline
        from reference_model.pipeline import run_explicit_window

        layout = Explicit4DLayout.from_mappings(EXPLICIT_RECORDS)
        frames = tuple(FrameRequest(timestamp, Camera.from_mapping(CAMERA)) for timestamp in TIMESTAMPS[:5])
        actual = GSPIMPipeline(l2_capacity=512).run_window(layout, frames, window_start=0.0, window_end=5.0)
        expected = run_explicit_window(EXPLICIT_RECORDS, CAMERA, TIMESTAMPS[:5], 0.0, 5.0, 512)
        golden = json.loads((Path(__file__).parent / "golden" / "explicit4d_window.json").read_text(encoding="ascii"))
        self.assertEqual(golden["window"], {"start": 0.0, "end": 5.0})
        self.assertEqual(actual.active_ids, expected.active_ids)
        self.assertEqual(actual.render_order_plan.rols, expected.rols)
        self.assertEqual(tuple(batch.footprint.total_bytes for batch in actual.batches), expected.batch_bytes)
        self.assertEqual(tuple(frame.checksum for frame in actual.rendered_frames), expected.checksums)
        self.assertEqual(sorted(actual.active_ids), golden["active_ids"])
        self.assertEqual(sorted(actual.stable_ids), golden["stable_ids"])
        self.assertEqual(sorted(actual.unstable_ids), golden["unstable_ids"])
        self.assertEqual([list(rol) for rol in actual.render_order_plan.rols], golden["rols"])
        self.assertEqual([batch.footprint.total_bytes for batch in actual.batches], golden["batch_bytes"])
        self.assertEqual([frame.checksum for frame in actual.rendered_frames], golden["checksums"])
        for parent, child in golden["required_edges"]:
            self.assertTrue(actual.trace.has_dependency(parent, child))

    def test_three_workload_fixtures_run_and_six_frames_exercise_loopback(self):
        from gspim.layouts.anchored import Anchored4DGSLayout
        from gspim.layouts.ex4dgs import Ex4DGSLayout
        from gspim.layouts.explicit4d import Explicit4DLayout
        from gspim.model import Camera, FrameRequest
        from gspim.pipeline import GSPIMPipeline

        camera = Camera.from_mapping(CAMERA)
        four_frames = tuple(FrameRequest(timestamp, camera) for timestamp in TIMESTAMPS[:4])
        two_frames = tuple(FrameRequest(timestamp, camera) for timestamp in TIMESTAMPS[4:])
        pipeline = GSPIMPipeline(l2_capacity=512)
        sequence = pipeline.run_sequence(
            Explicit4DLayout.from_mappings(EXPLICIT_RECORDS), four_frames + two_frames,
            initial_window_size=4, frame_period=1.0,
        )
        golden = json.loads((Path(__file__).parent / "golden" / "explicit4d_six_frame_sequence.json").read_text(encoding="ascii"))
        self.assertEqual([len(window.frames) for window in sequence.windows], golden["window_sizes"])
        self.assertEqual([window.next_window_size for window in sequence.windows], golden["next_window_sizes"])
        self.assertEqual(
            [[frame.checksum for frame in window.rendered_frames] for window in sequence.windows],
            golden["checksums"],
        )
        for parent, child in golden["required_edges"]:
            self.assertTrue(sequence.trace.has_dependency(parent, child))
        self.assertIn("w0:s5-render-0", sequence.windows[1].live_at_completion["w1:s1-select-d0"])
        for layout in (Ex4DGSLayout.from_mappings(EX4DGS_RECORDS), Anchored4DGSLayout.from_mappings(ANCHORS, ANCHOR_PAYLOADS)):
            result = pipeline.run_window(layout, four_frames, window_start=0.0, window_end=4.0)
            self.assertTrue(result.rendered_frames)
            self.assertTrue(all(frame.checksum for frame in result.rendered_frames))

    def test_ex4dgs_and_anchored_match_independent_golden_windows(self):
        from gspim.layouts.anchored import Anchored4DGSLayout
        from gspim.layouts.ex4dgs import Ex4DGSLayout
        from gspim.model import Camera, FrameRequest
        from gspim.pipeline import GSPIMPipeline
        from reference_model.pipeline import run_anchored_window, run_ex4dgs_window

        frames = tuple(FrameRequest(timestamp, Camera.from_mapping(CAMERA)) for timestamp in TIMESTAMPS[:5])
        cases = (
            (
                "ex4dgs",
                Ex4DGSLayout.from_mappings(EX4DGS_RECORDS),
                run_ex4dgs_window(EX4DGS_RECORDS, CAMERA, TIMESTAMPS[:5], 0.0, 5.0, 512),
            ),
            (
                "anchored",
                Anchored4DGSLayout.from_mappings(ANCHORS, ANCHOR_PAYLOADS),
                run_anchored_window(ANCHORS, ANCHOR_PAYLOADS, CAMERA, TIMESTAMPS[:5], 0.0, 5.0, 512),
            ),
        )
        for name, layout, expected in cases:
            actual = GSPIMPipeline(l2_capacity=512).run_window(layout, frames, window_start=0.0, window_end=5.0)
            golden = json.loads((Path(__file__).parent / "golden" / f"{name}_window.json").read_text(encoding="ascii"))
            self.assertEqual(golden["window"], {"start": 0.0, "end": 5.0})
            self.assertEqual(actual.active_ids, expected.active_ids)
            self.assertEqual(actual.stable_ids, expected.stable_ids)
            self.assertEqual(actual.unstable_ids, expected.unstable_ids)
            self.assertEqual(actual.render_order_plan.rols, expected.rols)
            self.assertEqual(tuple(batch.footprint.total_bytes for batch in actual.batches), expected.batch_bytes)
            self.assertEqual(tuple(frame.checksum for frame in actual.rendered_frames), expected.checksums)
            self.assertEqual(sorted(actual.active_ids), golden["active_ids"])
            self.assertEqual(sorted(actual.stable_ids), golden["stable_ids"])
            self.assertEqual(sorted(actual.unstable_ids), golden["unstable_ids"])
            self.assertEqual([list(rol) for rol in actual.render_order_plan.rols], golden["rols"])
            self.assertEqual([batch.footprint.total_bytes for batch in actual.batches], golden["batch_bytes"])
            self.assertEqual([frame.checksum for frame in actual.rendered_frames], golden["checksums"])

    def test_ex4dgs_partial_support_fails_before_any_s1_or_hcf_work(self):
        from gspim.hcf import HierarchicalCompactionFabric
        from gspim.layouts.ex4dgs import Ex4DGSLayout
        from gspim.model import Camera, FrameRequest
        from gspim.pipeline import GSPIMPipeline
        from gspim.runtime import Runtime
        from reference_model.pipeline import run_ex4dgs_window

        partial = dict(EX4DGS_RECORDS[1])
        partial["keyframes"] = (
            dict(EX4DGS_RECORDS[1]["keyframes"][0]),
            {**EX4DGS_RECORDS[1]["keyframes"][1], "timestamp": 2.0},
        )
        frames = tuple(FrameRequest(timestamp, Camera.from_mapping(CAMERA)) for timestamp in TIMESTAMPS[:4])
        runtime = Runtime()
        hcf = Mock(spec=HierarchicalCompactionFabric)
        with self.assertRaisesRegex(ValueError, "does not support every requested S2 frame"):
            GSPIMPipeline(l2_capacity=512).run_window(
                Ex4DGSLayout.from_mappings((partial,)),
                frames,
                window_start=0.0,
                window_end=4.0,
                runtime=runtime,
                hcf=hcf,
            )
        hcf.reorganize.assert_not_called()
        self.assertEqual(runtime.trace.nodes, {})
        with self.assertRaisesRegex(ValueError, "does not support every requested S2 frame"):
            run_ex4dgs_window((partial,), CAMERA, TIMESTAMPS[:4], 0.0, 4.0, 512)

    def test_anchored_partial_support_fails_before_any_s1_or_hcf_work(self):
        from gspim.hcf import HierarchicalCompactionFabric
        from gspim.layouts.anchored import Anchored4DGSLayout
        from gspim.model import Camera, FrameRequest
        from gspim.pipeline import GSPIMPipeline
        from gspim.runtime import Runtime
        from reference_model.pipeline import run_anchored_window

        partial_payloads = {primitive_id: dict(payload) for primitive_id, payload in ANCHOR_PAYLOADS.items()}
        partial_payloads[100]["frame_geometry"] = partial_payloads[100]["frame_geometry"][:2]
        frames = tuple(FrameRequest(timestamp, Camera.from_mapping(CAMERA)) for timestamp in TIMESTAMPS[:4])
        runtime = Runtime()
        hcf = Mock(spec=HierarchicalCompactionFabric)
        with self.assertRaisesRegex(ValueError, "does not support every requested S2 frame"):
            GSPIMPipeline(l2_capacity=512).run_window(
                Anchored4DGSLayout.from_mappings(ANCHORS, partial_payloads),
                frames,
                window_start=0.0,
                window_end=4.0,
                runtime=runtime,
                hcf=hcf,
            )
        hcf.reorganize.assert_not_called()
        self.assertEqual(runtime.trace.nodes, {})
        with self.assertRaisesRegex(ValueError, "does not support every requested S2 frame"):
            run_anchored_window(ANCHORS, partial_payloads, CAMERA, TIMESTAMPS[:4], 0.0, 4.0, 512)

    def test_overflow_completion_is_owned_by_its_physical_die(self):
        from gspim.config import ArchitectureConfig
        from gspim.layouts.explicit4d import Explicit4DLayout
        from gspim.model import Camera, FrameRequest
        from gspim.pipeline import GSPIMPipeline

        overflow_records = [dict(record) for record in EXPLICIT_RECORDS]
        overflow_records[2]["die"] = 0
        overflow_records[1].update({"block": 1, "slot": 0})
        overflow_records[3].update({"block": 1, "slot": 0})
        layout = Explicit4DLayout.from_mappings(overflow_records)
        config = ArchitectureConfig(pim_banks_per_die=15, reserved_banks_per_die=1, block_records=1)
        frames = tuple(FrameRequest(timestamp, Camera.from_mapping(CAMERA)) for timestamp in TIMESTAMPS[:5])
        result = GSPIMPipeline(config=config, l2_capacity=512).run_window(layout, frames, window_start=0.0, window_end=5.0)
        self.assertIn("w0:s1-gpu-fallback-d0-bank1-b0", result.trace.nodes)
        event = result.completion_events["w0:s1-reorg-d0"]
        self.assertEqual(event.overflow_ids, (2,))

    def test_pipeline_rejects_locations_outside_the_configured_ppim_block(self):
        from gspim.config import ArchitectureConfig
        from gspim.layouts.explicit4d import Explicit4DLayout
        from gspim.model import Camera, FrameRequest
        from gspim.pipeline import GSPIMPipeline

        records = [dict(record) for record in EXPLICIT_RECORDS[:1]]
        records[0]["slot"] = 1
        frames = tuple(FrameRequest(timestamp, Camera.from_mapping(CAMERA)) for timestamp in TIMESTAMPS[:2])
        with self.assertRaisesRegex(ValueError, "PPIM block width"):
            GSPIMPipeline(config=ArchitectureConfig(block_records=1)).run_window(
                Explicit4DLayout.from_mappings(records),
                frames,
                window_start=0.0,
                window_end=2.0,
            )

    def test_s3_moves_only_sort_metadata_and_s5_consumes_packed_batches(self):
        from gspim.layouts.explicit4d import Explicit4DLayout
        from gspim.model import Camera, FrameRequest
        from gspim.pipeline import GSPIMPipeline, SortingMetadata

        layout = Explicit4DLayout.from_mappings(EXPLICIT_RECORDS)
        frames = tuple(FrameRequest(timestamp, Camera.from_mapping(CAMERA)) for timestamp in TIMESTAMPS[:5])
        result = GSPIMPipeline(l2_capacity=512).run_window(layout, frames, window_start=0.0, window_end=5.0)
        for packed in result.s3_pack_results:
            self.assertTrue(all(isinstance(record.payload, SortingMetadata) for record in packed.packed_records))
            self.assertTrue(all(not hasattr(record.payload, "opacity") for record in packed.packed_records))
        self.assertTrue(all(set(packed.output_ids) == set(batch.attribute_ids) for batch, packed in zip(result.batches, result.batch_pack_results)))
        if len(result.batches) > 1:
            self.assertIn("w0:s5-render-0", result.live_at_completion["w0:s5-pack-1-d0"])

    def test_s1_geometry_comes_from_active_buffer_payload_not_source_layout(self):
        from gspim.hcf import HCFRecord, HCFResult, HierarchicalCompactionFabric, ReorgPurpose
        from gspim.layouts.explicit4d import Explicit4DLayout
        from gspim.model import Camera, FrameRequest
        from gspim.pipeline import GSPIMPipeline

        layout = Explicit4DLayout.from_mappings(EXPLICIT_RECORDS)
        frames = tuple(FrameRequest(timestamp, Camera.from_mapping(CAMERA)) for timestamp in TIMESTAMPS[:5])
        baseline = GSPIMPipeline(l2_capacity=512).run_window(layout, frames, window_start=0.0, window_end=5.0)
        original = HierarchicalCompactionFabric.reorganize

        def corrupt_active(self, records, selected_ids, purpose):
            result = original(self, records, selected_ids, purpose)
            if purpose is not ReorgPurpose.ACTIVE:
                return result
            packed = []
            for record in result.packed_records:
                payload = dict(record.payload)
                if record.primitive_id == 0:
                    payload["opacity"] = 0.0
                packed.append(HCFRecord(record.primitive_id, record.location, payload))
            return HCFResult(result.purpose, result.output_ids, tuple(packed), result.die_ranges, result.block_completions, result.overflow_ids)

        with patch.object(HierarchicalCompactionFabric, "reorganize", corrupt_active):
            changed = GSPIMPipeline(l2_capacity=512).run_window(layout, frames, window_start=0.0, window_end=5.0)
        self.assertNotEqual(
            tuple(frame.checksum for frame in baseline.rendered_frames),
            tuple(frame.checksum for frame in changed.rendered_frames),
        )

    def test_anchored_s1_leaves_binding_expansion_to_hcf(self):
        from gspim.layouts.anchored import Anchored4DGSLayout
        from gspim.model import Camera, FrameRequest
        from gspim.pipeline import GSPIMPipeline

        layout = Anchored4DGSLayout.from_mappings(ANCHORS, ANCHOR_PAYLOADS)
        frames = tuple(FrameRequest(timestamp, Camera.from_mapping(CAMERA)) for timestamp in TIMESTAMPS[:5])
        self.assertFalse(hasattr(layout, "expand_records"))
        result = GSPIMPipeline(l2_capacity=512).run_window(layout, frames, window_start=0.0, window_end=5.0)
        self.assertEqual(result.active_ids, {100, 101})

    def test_s3_and_s5_reject_invalid_hcf_payloads(self):
        from gspim.hcf import HCFRecord, HCFResult, HierarchicalCompactionFabric, ReorgPurpose
        from gspim.layouts.explicit4d import Explicit4DLayout
        from gspim.model import Camera, FrameRequest
        from gspim.pipeline import GSPIMPipeline

        layout = Explicit4DLayout.from_mappings(EXPLICIT_RECORDS)
        frames = tuple(FrameRequest(timestamp, Camera.from_mapping(CAMERA)) for timestamp in TIMESTAMPS[:5])
        original = HierarchicalCompactionFabric.reorganize

        def corrupt_metadata(self, records, selected_ids, purpose):
            result = original(self, records, selected_ids, purpose)
            if purpose not in (ReorgPurpose.STABLE, ReorgPurpose.UNSTABLE) or not result.packed_records:
                return result
            first, *rest = result.packed_records
            packed = (HCFRecord(first.primitive_id, first.location, object()), *rest)
            return HCFResult(result.purpose, result.output_ids, packed, result.die_ranges, result.block_completions, result.overflow_ids)

        with patch.object(HierarchicalCompactionFabric, "reorganize", corrupt_metadata):
            with self.assertRaisesRegex(RuntimeError, "non-metadata"):
                GSPIMPipeline(l2_capacity=512).run_window(layout, frames, window_start=0.0, window_end=5.0)

        def corrupt_batch(self, records, selected_ids, purpose):
            result = original(self, records, selected_ids, purpose)
            if purpose is not ReorgPurpose.BATCH or not result.packed_records:
                return result
            first, *rest = result.packed_records
            packed = (HCFRecord(first.primitive_id, first.location, object()), *rest)
            return HCFResult(result.purpose, result.output_ids, packed, result.die_ranges, result.block_completions, result.overflow_ids)

        with patch.object(HierarchicalCompactionFabric, "reorganize", corrupt_batch):
            with self.assertRaisesRegex(ValueError, "Batch Buffer"):
                GSPIMPipeline(l2_capacity=512).run_window(layout, frames, window_start=0.0, window_end=5.0)

    def test_s1_s3_release_each_physical_block_without_window_barrier(self):
        from gspim.layouts.explicit4d import Explicit4DLayout
        from gspim.model import Camera, FrameRequest
        from gspim.pipeline import GSPIMPipeline

        records = [dict(record) for record in EXPLICIT_RECORDS]
        records[1]["block"] = 1
        records[1]["temporal_mean"] = 2.0
        layout = Explicit4DLayout.from_mappings(records)
        frames = tuple(FrameRequest(timestamp, Camera.from_mapping(CAMERA)) for timestamp in TIMESTAMPS[:5])
        result = GSPIMPipeline(l2_capacity=512).run_window(layout, frames, window_start=0.0, window_end=5.0)
        self.assertTrue(result.trace.has_dependency("w0:s1-reorg-d0:bank0-b0", "w0:s2-d0-bank0-b0"))
        s3_gpu_zero = "w0:s3-gpu-d0-bank0-b0"
        s3_gpu_one = "w0:s3-gpu-d0-bank0-b1"
        s3_block_zero = "w0:s3-select-d0-bank0-b0"
        s3_block_one = "w0:s3-select-d0-bank0-b1"
        self.assertTrue(result.trace.has_dependency("w0:s2-d0-bank0-b0", s3_gpu_zero))
        self.assertFalse(result.trace.has_dependency("w0:s2-d0-bank0-b1", s3_gpu_zero))
        self.assertTrue(result.trace.has_dependency(s3_gpu_zero, s3_block_zero))
        self.assertTrue(result.trace.has_dependency("w0:s2-d0-bank0-b1", s3_gpu_one))
        self.assertTrue(result.trace.has_dependency(s3_gpu_one, s3_block_one))
        self.assertIn("w0:s1-reorg-d0", result.live_at_completion["w0:s1-reorg-d0:bank0-b0"])
        self.assertEqual(result.trace.nodes[s3_block_zero].dependencies, (s3_gpu_zero,))
        self.assertEqual(
            sorted(metadata.active_index for metadata in result.sorting_metadata.values()),
            list(range(len(result.active_ids))),
        )

    def test_multi_batch_pipeline_preserves_rol_and_overlaps_pack_with_render(self):
        from gspim.layouts.explicit4d import Explicit4DLayout
        from gspim.model import Camera, FrameRequest
        from gspim.pipeline import GSPIMPipeline
        from reference_model.pipeline import run_explicit_window

        records = [dict(record) for record in EXPLICIT_RECORDS]
        records[1]["temporal_mean"] = 2.0
        records[2]["temporal_rotation"] = (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        records[2]["temporal_mean"] = 2.0
        records[2]["temporal_scale"] = (0.2, 0.2, 0.2, 0.5)
        layout = Explicit4DLayout.from_mappings(records)
        frames = tuple(FrameRequest(timestamp, Camera.from_mapping(CAMERA)) for timestamp in TIMESTAMPS[:5])
        result = GSPIMPipeline(l2_capacity=180).run_window(layout, frames, window_start=0.0, window_end=5.0)

        self.assertEqual(len(result.batches), 3)
        self.assertTrue(all(batch.footprint.total_bytes <= 180 for batch in result.batches))
        self.assertTrue(all(frame.order == result.render_order_plan.rols[index] for index, frame in enumerate(result.rendered_frames)))
        self.assertTrue(result.trace.has_dependency("w0:s5-pack-1-d0", "w0:s5-render-1"))
        self.assertIn("w0:s5-render-0", result.live_at_completion["w0:s5-pack-1-d0"])
        reference = run_explicit_window(records, CAMERA, TIMESTAMPS[:5], 0.0, 5.0, 512)
        self.assertEqual(
            tuple(frame.checksum for frame in result.rendered_frames),
            reference.checksums,
        )

    def test_sequence_drain_never_emits_a_one_frame_tail(self):
        from gspim.layouts.explicit4d import Explicit4DLayout
        from gspim.model import Camera, FrameRequest
        from gspim.pipeline import GSPIMPipeline

        frames = tuple(FrameRequest(timestamp, Camera.from_mapping(CAMERA)) for timestamp in TIMESTAMPS)
        result = GSPIMPipeline(l2_capacity=512).run_sequence(
            Explicit4DLayout.from_mappings(EXPLICIT_RECORDS),
            frames,
            initial_window_size=5,
            frame_period=1.0,
        )
        self.assertEqual([len(window.frames) for window in result.windows], [4, 2])
