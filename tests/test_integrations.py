"""Tests for opt-in external input validation and the public release gate."""

import subprocess
import sys
import unittest
from pathlib import Path


class IntegrationTests(unittest.TestCase):
    """External inputs are explicit and malformed data must not be invented."""

    def test_ex4dgs_normalized_adapter_rejects_missing_fields(self):
        from integrations.ex4dgs import layout_from_normalized_records

        with self.assertRaises(ValueError):
            layout_from_normalized_records(({"primitive_id": 1},))

    def test_public_tree_gate_runs(self):
        root = Path(__file__).resolve().parents[1]
        completed = subprocess.run([sys.executable, "tools/check_public_tree.py"], cwd=root, capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_release_gate_allows_local_checkouts_but_rejects_tracking_them(self):
        from tools.check_public_tree import tracks_root

        self.assertFalse(tracks_root({"README.md"}, "third_party"))
        self.assertTrue(tracks_root({"third_party/Ex4DGS/LICENSE"}, "third_party"))
        self.assertTrue(tracks_root({"baselines"}, "baselines"))
