"""Versioned deterministic inputs for the public mechanism regression suite."""

FIXTURE_VERSION = 1
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
