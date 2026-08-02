"""S5 tests for projection, tile assignment, alpha compositing, and image hashes."""

import unittest

from tests.fixtures import CAMERA, EXPLICIT_RECORDS


class RendererTests(unittest.TestCase):
    """The renderer must expose a real image instead of scalar compositing only."""

    def test_projection_tiles_and_forward_compositing_are_deterministic(self):
        from gspim.layouts.explicit4d import Explicit4DLayout
        from gspim.model import Camera, FrameRequest
        from gspim.renderer import ReferenceRenderer

        layout = Explicit4DLayout.from_mappings(EXPLICIT_RECORDS)
        frame = FrameRequest(1.0, Camera.from_mapping(CAMERA))
        geometry = layout.generate_geometry_from_payloads(
            {primitive_id: layout.payload(primitive_id) for primitive_id in (0, 2)},
            frame.timestamp,
        )
        image = ReferenceRenderer().render_order((0, 2), geometry, frame.camera)
        self.assertEqual(image.width, 8)
        self.assertEqual(image.height, 8)
        self.assertTrue(image.tiles_by_id[0])
        self.assertEqual(len(image.checksum), 64)
        self.assertNotEqual(image.checksum, ReferenceRenderer().render_order((2, 0), geometry, frame.camera).checksum)
