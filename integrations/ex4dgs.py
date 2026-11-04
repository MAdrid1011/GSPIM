"""Load Ex4DGS checkpoint parameters and held-out camera calibration."""

from __future__ import annotations

from pathlib import Path


class Ex4DGSCheckpoint:
    """Native cubic/slerp Ex4DGS inference parameters from a model archive.

    PLY attributes, SH coefficients, opacity envelopes, and camera poses are
    loaded from PLY files. Camera poses come from cameras.json, and cfg_args
    is parsed as literal keyword arguments.
    """

    def __init__(self, path: Path, device: str = "cuda") -> None:
        import ast
        import io
        import json
        import zipfile

        import numpy as np
        import torch
        from plyfile import PlyData

        path = Path(path)
        archive = zipfile.ZipFile(path) if path.is_file() else None
        try:
            def read(name: str) -> bytes:
                return archive.read(name) if archive else (path / name).read_bytes()

            expression = ast.parse(read("cfg_args").decode("utf-8"), mode="eval").body
            if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Name) or expression.func.id != "Namespace" or expression.args:
                raise ValueError("cfg_args must contain literal Namespace keyword arguments")
            self.config = {item.arg: ast.literal_eval(item.value) for item in expression.keywords}
            if self.config.get("interp_type") != "cube" or self.config.get("rot_interp_type") != "slerp":
                raise ValueError("native inference currently supports cubic position and slerp rotation")
            if self.config.get("loader") != "neural3dvideo" or self.config.get("sh_degree") != 3:
                raise ValueError("native inference requires N3V cameras and degree-three SH")
            self.cameras = json.loads(read("cameras.json"))
            prefix = f"point_cloud/iteration_{int(self.config['iterations'])}/"
            static = PlyData.read(io.BytesIO(read(prefix + "point_cloud.ply")))["vertex"].data
            dynamic = PlyData.read(io.BytesIO(read(prefix + "dynamic_point_cloud.ply")))["vertex"].data
        finally:
            if archive:
                archive.close()

        def fields(records, names):
            return torch.from_numpy(np.stack([records[name] for name in names], axis=-1).astype(np.float32)).to(device)

        def numbered(records, prefix):
            names = sorted((name for name in records.dtype.names if name.startswith(prefix)), key=lambda name: tuple(map(int, name[len(prefix):].split("_"))))
            if not names:
                raise ValueError(f"checkpoint has no {prefix} attributes")
            return fields(records, names)

        self.xyz = fields(static, ("x", "y", "z"))
        self.displacement = numbered(static, "xyz_disp_")
        self.motion = numbered(dynamic, "motion_xyz_").reshape(len(dynamic), -1, 3)
        self.rotation = numbered(static, "rot_")
        self.motion_rotation = numbered(dynamic, "motion_rot_").reshape(len(dynamic), -1, 4)
        self.scales = torch.cat((numbered(static, "scale_"), numbered(dynamic, "motion_scale_"))).exp()
        self.opacity = torch.cat((fields(static, ("opacity",)), fields(dynamic, ("motion_opacity",)))).sigmoid().flatten()
        self.opacity_center = numbered(dynamic, "motion_opacity_c_")
        self.opacity_variance = numbered(dynamic, "motion_opacity_v_")
        coefficients = []
        for records, prefix in ((static, ""), (dynamic, "motion_")):
            dc = numbered(records, prefix + "f_dc_").reshape(-1, 1, 3)
            rest = numbered(records, prefix + "f_rest_").reshape(-1, 3, 15).transpose(1, 2)
            coefficients.append(torch.cat((dc, rest), dim=1))
        self.sh = torch.cat(coefficients).contiguous()
        self.interval = float(self.config["time_interval"])
        self.time_shift = float(self.config["time_pad"]) + self.interval
        self.duration = int(self.config["duration"])
        if self.interval <= 0 or self.duration < 2 or self.motion.shape[1] != self.motion_rotation.shape[1]:
            raise ValueError("invalid native keyframe timing")
        last_key = int((self.duration - 1 + self.time_shift) // self.interval)
        if last_key + 2 >= self.motion.shape[1]:
            raise ValueError("checkpoint keyframes do not cover the evaluation sequence")

        # Upstream Scene serializes held-out cam00 first, followed by training cameras.
        camera = self.cameras[0]
        resolution = int(self.config["resolution"])
        if resolution not in (1, 2, 4, 8):
            raise ValueError("checkpoint resolution must be an integer image divisor")
        self.width = int(camera["width"]) // resolution
        self.height = int(camera["height"]) // resolution
        camera_to_world = torch.eye(4, device=device)
        camera_to_world[:3, :3] = torch.tensor(camera["rotation"], device=device)
        camera_to_world[:3, 3] = torch.tensor(camera["position"], device=device)
        self.view = camera_to_world.inverse()[None].contiguous()
        self.camera_position = camera_to_world[:3, 3].contiguous()
        self.intrinsics = torch.tensor([[[camera["fx"] / resolution, 0, self.width / 2], [0, camera["fy"] / resolution, self.height / 2], [0, 0, 1]]], device=device)
        self.name = path.stem if path.is_file() else path.name
