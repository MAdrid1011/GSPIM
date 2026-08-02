"""PPIM program tests: exact programs, fixed-point boundaries, and failures."""

import math
import unittest

from tests.fixtures import EXPLICIT_RECORDS


class PpimTests(unittest.TestCase):
    """The PPIM interpreter must execute the selected program, never a host shortcut."""

    def test_temp_activity_executes_temporal_covariance_program(self):
        from gspim.ppim import PPIMInterpreter, ProgramId

        interpreter = PPIMInterpreter()
        active = interpreter.run(ProgramId.TEMP_ACTIVITY, EXPLICIT_RECORDS[0], 0.0, 4.0)
        inactive = interpreter.run(ProgramId.TEMP_ACTIVITY, EXPLICIT_RECORDS[1], 0.0, 4.0)
        self.assertTrue(active.mask)
        self.assertFalse(inactive.mask)
        self.assertAlmostEqual(active.registers["sigma_tt"], 0.145, places=4)
        self.assertIn("window_distance", active.registers)
        self.assertAlmostEqual(active.registers["rhs"], 0.145 * math.log(20.0), places=4)

    def test_fixed_point_raw_vector_matches_the_rtl_program_vector(self):
        from gspim.ppim import PPIMInterpreter, ProgramId

        # This is the TEMP vector used by hardware/PpimSpec.scala: the temporal
        # row is (1, 0, 0, 0), the scale is (0.5, 0, 0, 0), and all values use
        # Q16.  The expected raw Sigma_tt is 16384 in both implementations.
        record = {
            "temporal_mean": 1.0,
            "temporal_rotation": (
                (0.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0, 0.0),
            ),
            "temporal_scale": (0.5, 0.0, 0.0, 0.0),
        }
        result = PPIMInterpreter().run(ProgramId.TEMP_ACTIVITY, record, 0.0, 4.0)
        self.assertTrue(result.mask)
        self.assertEqual(result.registers["sigma_tt_raw"], 16384)

    def test_boundary_rules_retain_s1_but_make_s3_unstable(self):
        from gspim.ppim import PPIMInterpreter, ProgramId

        interpreter = PPIMInterpreter(fractional_bits=10)
        s1 = interpreter.run(ProgramId.TEMP_ACTIVITY, EXPLICIT_RECORDS[0], 1.5, 1.5)
        stable = interpreter.run(ProgramId.DEPTH_STABLE, {"score": 0.98}, tau=0.98)
        self.assertTrue(s1.mask)
        self.assertFalse(stable.mask)

    def test_keyframe_and_anchor_programs_obey_half_open_window_overlap(self):
        from gspim.ppim import PPIMInterpreter, ProgramId

        interpreter = PPIMInterpreter()
        static = interpreter.run(ProgramId.KEYFRAME_RANGE, {"is_static": True}, 4.0, 5.0)
        dynamic = interpreter.run(ProgramId.KEYFRAME_RANGE, {"is_static": False, "start": 2.0, "end": 5.0}, 4.0, 5.0)
        starts_at_end = interpreter.run(ProgramId.KEYFRAME_RANGE, {"is_static": False, "start": 5.0, "end": 6.0}, 4.0, 5.0)
        ends_at_start = interpreter.run(ProgramId.KEYFRAME_RANGE, {"is_static": False, "start": 2.0, "end": 4.0}, 4.0, 5.0)
        anchor = interpreter.run(ProgramId.ANCHOR_OVERLAP, {"start": 7.0, "end": 8.0}, 0.0, 5.0)
        anchor_starts_at_end = interpreter.run(ProgramId.ANCHOR_OVERLAP, {"start": 5.0, "end": 6.0}, 4.0, 5.0)
        self.assertTrue(static.mask)
        self.assertTrue(dynamic.mask)
        self.assertFalse(starts_at_end.mask)
        self.assertTrue(ends_at_start.mask)
        self.assertFalse(anchor.mask)
        self.assertFalse(anchor_starts_at_end.mask)

    def test_missing_program_fields_are_rejected(self):
        from gspim.ppim import PPIMInterpreter, ProgramId

        with self.assertRaises(ValueError):
            PPIMInterpreter().run(ProgramId.TEMP_ACTIVITY, {"temporal_mean": 0.0}, 0.0, 1.0)
