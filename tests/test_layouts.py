"""Small design inputs and tests for typed representation adapters."""

import unittest

TIMESTAMPS = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)

CAMERA = {
    "world_to_camera": (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    ),
    "focal_x": 8.0,
    "focal_y": 8.0,
    "center_x": 4.0,
    "center_y": 4.0,
    "width": 8,
    "height": 8,
    "tile_size": 4,
    "near": 0.1,
    "far": 10.0,
}

IDENTITY4 = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)
COUPLED_4D_ROTATION = (
    (0.7071067811865476, 0.0, 0.0, -0.7071067811865476),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.7071067811865476, 0.0, 0.0, 0.7071067811865476),
)
DEPTH_COUPLED_4D_ROTATION = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 0.7071067811865476, -0.7071067811865476),
    (0.0, 0.0, 0.7071067811865476, 0.7071067811865476),
)
DIAGONAL_COVARIANCE = (
    (0.04, 0.0, 0.0),
    (0.0, 0.04, 0.0),
    (0.0, 0.0, 0.04),
)

EXPLICIT_RECORDS = (
    {
        "primitive_id": 0,
        "temporal_mean": 1.0,
        "temporal_rotation": COUPLED_4D_ROTATION,
        "temporal_scale": (0.2, 0.2, 0.2, 0.5),
        "mean": (-0.25, 0.0, 2.0),
        "opacity": 0.65,
        "color": (0.9, 0.1, 0.1),
        "attribute_bytes": 48,
        "die": 0,
        "bank": 0,
        "block": 0,
        "slot": 0,
    },
    {
        "primitive_id": 1,
        "temporal_mean": 6.0,
        "temporal_rotation": IDENTITY4,
        "temporal_scale": (0.2, 0.2, 0.2, 0.1),
        "mean": (0.20, 0.0, 2.5),
        "opacity": 0.4,
        "color": (0.1, 0.9, 0.1),
        "attribute_bytes": 48,
        "die": 0,
        "bank": 0,
        "block": 0,
        "slot": 1,
    },
    {
        "primitive_id": 2,
        "temporal_mean": 4.4,
        "temporal_rotation": DEPTH_COUPLED_4D_ROTATION,
        "temporal_scale": (0.2, 0.2, 0.2, 1.0),
        "mean": (0.0, 0.15, 3.0),
        "opacity": 0.55,
        "color": (0.1, 0.2, 0.9),
        "attribute_bytes": 56,
        "die": 1,
        "bank": 1,
        "block": 0,
        "slot": 0,
    },
    {
        "primitive_id": 3,
        "temporal_mean": -1.0,
        "temporal_rotation": IDENTITY4,
        "temporal_scale": (0.2, 0.2, 0.2, 0.1),
        "mean": (0.0, -0.2, 3.2),
        "opacity": 0.5,
        "color": (0.9, 0.8, 0.1),
        "attribute_bytes": 40,
        "die": 1,
        "bank": 1,
        "block": 0,
        "slot": 1,
    },
)

EX4DGS_RECORDS = (
    {
        "primitive_id": 10,
        "is_static": True,
        "keyframes": ({"timestamp": 0.0, "mean": (-0.1, 0.0, 2.2), "covariance": DIAGONAL_COVARIANCE, "opacity": 0.5, "color": (0.7, 0.7, 0.2)},),
        "attribute_bytes": 44,
        "die": 0,
        "bank": 0,
        "block": 0,
        "slot": 0,
    },
    {
        "primitive_id": 11,
        "is_static": False,
        "keyframes": (
            {"timestamp": 0.0, "mean": (0.0, 0.0, 2.8), "covariance": DIAGONAL_COVARIANCE, "opacity": 0.7, "color": (0.2, 0.8, 0.8)},
            {"timestamp": 5.0, "mean": (0.0, 0.0, 3.5), "covariance": ((0.09, 0.0, 0.0), (0.0, 0.04, 0.0), (0.0, 0.0, 0.04)), "opacity": 0.4, "color": (0.1, 0.4, 0.9)},
        ),
        "attribute_bytes": 52,
        "die": 0,
        "bank": 0,
        "block": 0,
        "slot": 1,
    },
)

ANCHORS = (
    {"anchor_id": 20, "start": 0.0, "end": 2.0, "binding_start": 100, "binding_count": 2, "die": 0, "bank": 1, "block": 0, "slot": 0},
    {"anchor_id": 21, "start": 7.0, "end": 8.0, "binding_start": 102, "binding_count": 1, "die": 1, "bank": 2, "block": 0, "slot": 0},
)
ANCHOR_PAYLOADS = {
    100: {"frame_geometry": tuple({"timestamp": float(t), "mean": (-0.1, 0.0, 2.1), "covariance": DIAGONAL_COVARIANCE, "opacity": 0.5, "color": (0.6, 0.2, 0.6)} for t in range(6)), "attribute_bytes": 40, "die": 0, "bank": 3, "block": 6, "slot": 0},
    101: {"frame_geometry": tuple({"timestamp": float(t), "mean": (0.1, 0.0, 2.6 + 0.02 * t), "covariance": DIAGONAL_COVARIANCE, "opacity": 0.5, "color": (0.2, 0.6, 0.6)} for t in range(6)), "attribute_bytes": 40, "die": 0, "bank": 3, "block": 6, "slot": 1},
    102: {"frame_geometry": tuple({"timestamp": float(t), "mean": (0.0, 0.0, 3.0), "covariance": DIAGONAL_COVARIANCE, "opacity": 0.4, "color": (0.4, 0.4, 0.4)} for t in range(6)), "attribute_bytes": 40, "die": 1, "bank": 4, "block": 6, "slot": 0},
}


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
        unsupported = dict(EXPLICIT_RECORDS[0])
        unsupported["velocity"] = (0.0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            Explicit4DLayout.from_mappings((unsupported,))
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
        misplaced = dict(EX4DGS_RECORDS[0])
        misplaced["opacity"] = 0.5
        with self.assertRaises(ValueError):
            Ex4DGSLayout.from_mappings((misplaced,))

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
