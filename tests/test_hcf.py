"""HCF tests: masks must move payloads into typed live ranges."""

import unittest


class HcfTests(unittest.TestCase):
    """Compaction verifies actual field movement, not just selected counts."""

    def test_hcf_fifo_gathers_payloads_and_reports_block_ranges(self):
        from gspim.hcf import HCFRecord, HierarchicalCompactionFabric, RecordLocation, ReorgPurpose

        records = {
            0: HCFRecord(0, RecordLocation(0, 0, 0, 0, 0), {"attribute": "a"}),
            1: HCFRecord(1, RecordLocation(1, 0, 1, 0, 0), {"attribute": "b"}),
            2: HCFRecord(2, RecordLocation(2, 1, 0, 0, 0), {"attribute": "c"}),
        }
        result = HierarchicalCompactionFabric(reserved_capacity_per_die=2).reorganize(records, {0, 2}, ReorgPurpose.ACTIVE)
        self.assertEqual(result.output_ids, (0, 2))
        self.assertEqual(tuple(record.payload["attribute"] for record in result.packed_records), ("a", "c"))
        self.assertEqual(result.die_ranges[0].count, 1)
        self.assertEqual(result.die_ranges[1].count, 1)
        self.assertEqual(tuple(block.output_ids for block in result.block_completions), ((0,), (2,)))

    def test_hcf_shares_reserved_capacity_by_live_range_and_reports_overflow(self):
        from gspim.hcf import HCFRecord, HierarchicalCompactionFabric, RecordLocation, ReorgPurpose

        records = {index: HCFRecord(index, RecordLocation(index, 0, index, 0, 0), {"id": index}) for index in range(3)}
        fabric = HierarchicalCompactionFabric(reserved_capacity_per_die=1)
        active = fabric.reorganize(records, {0, 1}, ReorgPurpose.ACTIVE)
        self.assertEqual(active.overflow_ids, (1,))
        blocked = fabric.reorganize(records, {2}, ReorgPurpose.STABLE)
        self.assertEqual(blocked.output_ids, ())
        self.assertEqual(blocked.overflow_ids, (2,))
        fabric.release(active)
        stable = fabric.reorganize(records, {2}, ReorgPurpose.STABLE)
        self.assertEqual(active.purpose, ReorgPurpose.ACTIVE)
        self.assertEqual(stable.purpose, ReorgPurpose.STABLE)
        self.assertNotEqual(active.die_ranges[0].region, stable.die_ranges[0].region)
        self.assertEqual(stable.die_ranges[0].start, 0)

    def test_hcf_expands_anchor_bindings_inside_the_fifo_path(self):
        from gspim.hcf import HCFRecord, HierarchicalCompactionFabric, RecordLocation, ReorgPurpose

        bound = (
            HCFRecord(100, RecordLocation(100, 0, 1, 0, 0), {"attribute": "first"}),
            HCFRecord(101, RecordLocation(101, 0, 1, 0, 1), {"attribute": "second"}),
        )
        anchor = HCFRecord(20, RecordLocation(20, 0, 0, 2, 0), {"binding_start": 100, "binding_count": 2}, bound)
        result = HierarchicalCompactionFabric(reserved_capacity_per_die=4).reorganize({20: anchor, **{record.primitive_id: record for record in bound}}, {20}, ReorgPurpose.ACTIVE)
        self.assertEqual(result.output_ids, (100, 101))
        self.assertEqual(result.block_completions[0].input_ids, (100, 101))
        self.assertEqual(tuple(record.payload["attribute"] for record in result.packed_records), ("first", "second"))

    def test_hcf_fallback_retains_the_original_block_and_only_marks_its_tail(self):
        from gspim.hcf import HCFRecord, HierarchicalCompactionFabric, RecordLocation, ReorgPurpose

        records = {
            index: HCFRecord(index, RecordLocation(index, 0, 0, 0, index), {"attribute": index})
            for index in range(3)
        }
        result = HierarchicalCompactionFabric(reserved_capacity_per_die=2).reorganize(
            records, {0, 1, 2}, ReorgPurpose.ACTIVE,
        )
        self.assertEqual(result.output_ids, (0, 1))
        self.assertEqual(result.overflow_ids, (2,))
        self.assertEqual(len(result.gpu_fallbacks), 1)
        fallback = result.gpu_fallbacks[0]
        self.assertEqual((fallback.die, fallback.bank, fallback.block), (0, 0, 0))
        self.assertEqual(tuple(record.primitive_id for record in fallback.source_records), (0, 1, 2))
        self.assertEqual(fallback.overflow_ids, (2,))
