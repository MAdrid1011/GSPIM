"""S2--S4 tests for depth semantics, ROL preservation, and capacity safety."""

import unittest


class DataflowTests(unittest.TestCase):
    """Sorting and batching retain each frame's exact render order."""

    def test_depth_stability_uses_camera_space_normalization(self):
        from gspim.dataflow import depth_stability_score

        self.assertAlmostEqual(depth_stability_score((1.0, 1.2, 1.1), 0.0, 10.0), 0.9851119396, places=8)
        with self.assertRaises(ValueError):
            depth_stability_score((1.0,), 1.0, 1.0)

    def test_stable_once_unstable_per_frame_merge_preserves_all_rols(self):
        from gspim.dataflow import fixed_width_radix_sort, merge_path, plan_render_orders

        plan = plan_render_orders({0: (1.0, 1.0), 1: (2.0, 0.5), 2: (3.0, 3.0)}, {0, 2})
        self.assertEqual(plan.stable_order, (0, 2))
        self.assertEqual(plan.unstable_order_by_frame, ((1,), (1,)))
        self.assertEqual(plan.rols, ((0, 1, 2), (1, 0, 2)))
        self.assertEqual(fixed_width_radix_sort((3, 1, 2), {1: 2.0, 2: 1.0, 3: 1.0}), (2, 3, 1))
        self.assertEqual(
            merge_path((0, 4, 6), (1, 2, 3, 5), {0: 0.0, 4: 4.0, 6: 6.0}, {1: 1.0, 2: 2.0, 3: 3.0, 5: 5.0}),
            (0, 1, 2, 3, 4, 5, 6),
        )

    def test_batch_footprint_never_exceeds_capacity(self):
        from gspim.dataflow import CapacityError, make_batch_plan, plan_render_orders

        plan = plan_render_orders({1: (1.0, 1.0), 2: (2.0, 2.0)}, {1, 2})
        batches = make_batch_plan(plan, {1: 24, 2: 24}, 128)
        self.assertTrue(all(batch.footprint.total_bytes <= 128 for batch in batches))
        unsafe = plan_render_orders({1: (1.0,), 2: (2.0,)}, set())
        with self.assertRaises(CapacityError):
            make_batch_plan(unsafe, {1: 80, 2: 80}, 100)
