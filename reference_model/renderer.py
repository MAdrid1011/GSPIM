"""Independent deterministic raster checksum helper."""

from hashlib import sha256
from math import ceil, exp, floor, sqrt
from typing import Mapping, Sequence


def render_checksum(order: Sequence[int], geometry: Mapping[int, Mapping[str, object]], camera: Mapping[str, object]) -> str:
    width, height = int(camera["width"]), int(camera["height"])
    colors = [(0.0, 0.0, 0.0) for _ in range(width * height)]
    transmittance = [1.0 for _ in range(width * height)]
    for primitive_id in order:
        record = geometry[primitive_id]
        mean = tuple(float(value) for value in record["mean"])  # type: ignore[arg-type]
        matrix = camera["world_to_camera"]  # type: ignore[assignment]
        x, y, depth = tuple(sum(float(row[index]) * (mean[0], mean[1], mean[2], 1.0)[index] for index in range(4)) for row in matrix[:3])  # type: ignore[index]
        if depth <= 0.0:
            continue
        focal_x, focal_y = float(camera["focal_x"]), float(camera["focal_y"])
        center_x, center_y = focal_x * x / depth + float(camera["center_x"]), focal_y * y / depth + float(camera["center_y"])
        covariance = record["covariance"]  # type: ignore[assignment]
        rotation = tuple(tuple(float(matrix[row][column]) for column in range(3)) for row in range(3))  # type: ignore[index]
        camera_covariance = tuple(tuple(sum(rotation[row][left] * float(covariance[left][right]) * rotation[column][right] for left in range(3) for right in range(3)) for column in range(3)) for row in range(3))  # type: ignore[index]
        jacobian = ((focal_x / depth, 0.0, -focal_x * x / (depth * depth)), (0.0, focal_y / depth, -focal_y * y / (depth * depth)))
        screen_covariance = tuple(tuple(sum(jacobian[row][left] * camera_covariance[left][right] * jacobian[column][right] for left in range(3) for right in range(3)) for column in range(2)) for row in range(2))
        var_x = max(1e-6, screen_covariance[0][0])
        var_y = max(1e-6, screen_covariance[1][1])
        cov_xy = screen_covariance[0][1]
        determinant = var_x * var_y - cov_xy * cov_xy
        radius = 3.0 * sqrt(max(var_x, var_y))
        for py in range(max(0, floor(center_y - radius)), min(height - 1, ceil(center_y + radius)) + 1):
            for px in range(max(0, floor(center_x - radius)), min(width - 1, ceil(center_x + radius)) + 1):
                dx, dy = px + 0.5 - center_x, py + 0.5 - center_y
                inverse_xx, inverse_xy, inverse_yy = var_y / determinant, -cov_xy / determinant, var_x / determinant
                alpha = min(0.999, max(0.0, float(record["opacity"]) * exp(-0.5 * (inverse_xx * dx * dx + 2.0 * inverse_xy * dx * dy + inverse_yy * dy * dy))))
                index = py * width + px
                color = record["color"]  # type: ignore[assignment]
                colors[index] = tuple(colors[index][channel] + transmittance[index] * alpha * float(color[channel]) for channel in range(3))  # type: ignore[assignment,index]
                transmittance[index] *= 1.0 - alpha
    return sha256("|".join(f"{value:.9f}" for pixel in colors for value in pixel).encode("ascii")).hexdigest()
