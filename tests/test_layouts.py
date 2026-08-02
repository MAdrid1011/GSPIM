"""Tests for typed representation adapters and their fail-closed inputs."""

import unittest

from tests.fixtures import ANCHOR_PAYLOADS, ANCHORS, CAMERA, EX4DGS_RECORDS, EXPLICIT_RECORDS, TIMESTAMPS


class LayoutTests(unittest.TestCase):
    """Every workload must select through PPIM and generate complete geometry."""

    def test_explicit_4d_uses_temp_program_not_keyframe_mask_or(self):
        from gspim.layouts.explicit4d import Explicit4DLayout

        layout = Explicit4DLayout.from_mappings(EXPLICIT_RECORDS)
        selection = layout.select_window(0.0, 5.0)
        self.assertEqual(selection.active_ids, {0, 2})
        self.assertEqual(selection.program_name, "TEMP_ACTIVITY")
        geometry = layout.generate_geometry_from_payloads({0: layout.payload(0)}, 2.0)[0]
        self.assertEqual(len(geometry.covariance_world), 3)
        self.assertAlmostEqual(geometry.mean_world[0], -0.9741379310, places=8)
        self.assertEqual(geometry.mean_world[2], 2.0)
        legacy = dict(EXPLICIT_RECORDS[0])
        legacy["velocity"] = (0.0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            Explicit4DLayout.from_mappings((legacy,))
        duplicate = [dict(record) for record in EXPLICIT_RECORDS[:2]]
        duplicate[1]["slot"] = duplicate[0]["slot"]
        with self.assertRaisesRegex(ValueError, "duplicate physical location"):
            Explicit4DLayout.from_mappings(duplicate)

    def test_ex4dgs_selects_support_and_generates_covariance(self):
        from gspim.layouts.ex4dgs import Ex4DGSLayout
        from gspim.model import Camera, FrameRequest

        layout = Ex4DGSLayout.from_mappings(EX4DGS_RECORDS)
        self.assertEqual(layout.select_window(4.0, 6.0).active_ids, {10, 11})
        self.assertEqual(layout.select_window(6.0, 7.0).active_ids, {10})
        geometry = layout.generate_geometry_from_payloads({11: layout.payload(11)}, 2.5)[11]
        self.assertAlmostEqual(geometry.mean_world[2], 3.15)
        self.assertAlmostEqual(geometry.covariance_world[0][0], 0.065)
        self.assertAlmostEqual(geometry.opacity, 0.55)
        self.assertAlmostEqual(geometry.color[2], 0.85)
        with self.assertRaisesRegex(ValueError, "outside its declared keyframe support"):
            layout.generate_geometry_from_payloads({11: layout.payload(11)}, -0.1)
        partial = dict(EX4DGS_RECORDS[1])
        partial["keyframes"] = (
            dict(EX4DGS_RECORDS[1]["keyframes"][0]),
            {**EX4DGS_RECORDS[1]["keyframes"][1], "timestamp": 2.0},
        )
        partial_layout = Ex4DGSLayout.from_mappings((partial,))
        partial_selection = partial_layout.select_window(0.0, 4.0)
        frames = tuple(FrameRequest(timestamp, Camera.from_mapping(CAMERA)) for timestamp in TIMESTAMPS[:4])
        self.assertEqual(partial_selection.active_ids, {11})
        with self.assertRaisesRegex(ValueError, "does not support every requested S2 frame"):
            partial_layout.validate_window_frames(frames, partial_selection)
        legacy = dict(EX4DGS_RECORDS[0])
        legacy["opacity"] = 0.5
        with self.assertRaises(ValueError):
            Ex4DGSLayout.from_mappings((legacy,))

    def test_anchored_layout_expands_only_bound_payloads_and_rejects_missing_ones(self):
        from gspim.layouts.anchored import Anchored4DGSLayout

        layout = Anchored4DGSLayout.from_mappings(ANCHORS, ANCHOR_PAYLOADS)
        selection = layout.select_window(1.0, 3.0)
        self.assertEqual(selection.active_ids, {20})
        self.assertEqual(layout.select_window(2.0, 3.0).active_ids, {20})
        self.assertEqual(layout.select_window(3.0, 4.0).active_ids, set())
        self.assertEqual(layout.binding_records(20), (100, 101))
        self.assertFalse(hasattr(layout, "expand_records"))
        with self.assertRaises(ValueError):
            Anchored4DGSLayout.from_mappings(ANCHORS, {100: ANCHOR_PAYLOADS[100]})
        cross_die = {primitive_id: dict(payload) for primitive_id, payload in ANCHOR_PAYLOADS.items()}
        cross_die[100]["die"] = 1
        with self.assertRaisesRegex(ValueError, "anchor's die"):
            Anchored4DGSLayout.from_mappings(ANCHORS, cross_die)
        colliding = {primitive_id: dict(payload) for primitive_id, payload in ANCHOR_PAYLOADS.items()}
        colliding[100].update({"bank": 1, "block": 0, "slot": 0})
        with self.assertRaisesRegex(ValueError, "share a physical mask location"):
            Anchored4DGSLayout.from_mappings(ANCHORS, colliding)
