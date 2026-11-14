"""Tests for checkpoint boundaries, workload registration and input preparation."""

import unittest
from pathlib import Path


class IntegrationTests(unittest.TestCase):
    """Input checks validate the complete selection before loading CUDA models."""

    def test_ex4dgs_layout_rejects_missing_fields(self):
        from gspim.layouts.ex4dgs import Ex4DGSLayout

        with self.assertRaises(ValueError):
            Ex4DGSLayout.from_mappings(({"primitive_id": 1},))

    def test_manifest_covers_eighteen_combinations_and_real_coffee_ids(self):
        from integrations.manifest import DATA_SHA256, combinations

        pairs = tuple(combinations())
        self.assertEqual(len(pairs), 18)
        self.assertEqual(len({(workload.name, scene) for workload, scene in pairs}), 18)
        cameras = sorted(name for name in DATA_SHA256["coffee_martini"] if name.endswith(".mp4"))
        self.assertEqual(len(cameras), 18)
        self.assertNotIn("cam03.mp4", cameras)
        self.assertIn("cam20.mp4", cameras)
        self.assertIn("dance3", {scene for _, scene in pairs})
        self.assertIn("actor1_4", {scene for _, scene in pairs})

    def test_missing_inputs_are_listed_without_preparation(self):
        import tempfile
        from unittest.mock import patch
        from integrations.manifest import combinations
        from gspim.prepare import input_problems

        with tempfile.TemporaryDirectory() as temporary, patch("gspim.prepare.download") as download:
            problems = input_problems(combinations("anchored", "coffee_martini"), Path(temporary))
        self.assertEqual(len(problems), 3)
        self.assertTrue(any("checkpoints/anchored4dgs/coffee_martini.pth" in item for item in problems))
        self.assertTrue(any("GSPIM checkpoint bundle" in item for item in problems))
        self.assertTrue(any("300 missing images" in item for item in problems))
        download.assert_not_called()
