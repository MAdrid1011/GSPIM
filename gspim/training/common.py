"""Pinned native rasterization, cameras, losses and training storage paths."""

import importlib.util
import os
import runpy

import numpy as np
import torch

from integrations.manifest import ROOT, CACHE, ORIGINAL4DGS, FUSED_SSIM

UPSTREAM = ROOT / "third_party" / ORIGINAL4DGS.name
TRAINING = ROOT / "checkpoints/training"
FUSED_SSIM_COMMIT = FUSED_SSIM.commit
ANCHORED_TRAINING_STEPS = 2000


def initialize_cache():
    cache = CACHE / "training"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_EXTENSIONS_DIR"] = str(cache)


def cameras(scene):
    """Run the pinned n3v2blender camera math without extracting videos/JSON."""
    helpers = runpy.run_path(str(UPSTREAM / "scripts/n3v2blender.py"))
    bounds = np.load(ROOT / "datasets/n3v" / scene / "poses_bounds.npy")
    original = bounds[:, :15].reshape(-1, 3, 5)
    height, width, focal = original[0, :, -1]
    poses = np.concatenate((original[..., 1:2], original[..., 0:1],
                            -original[..., 2:3], original[..., 3:4]), -1)
    poses = np.concatenate((poses, np.tile([[[0, 0, 0, 1]]], (len(poses), 1, 1))), 1)
    poses[:, :3, 1:3] *= -1
    poses = poses[:, [1, 0, 2, 3], :]
    poses[:, 2, :] *= -1
    up = poses[:, :3, 1].sum(0)
    rotation = np.pad(helpers["rotmat"](up / np.linalg.norm(up), [0, 0, 1]), (0, 1))
    rotation[-1, -1] = 1
    poses = rotation @ poses
    total = np.zeros(3)
    weight = 0.0
    for i in range(len(poses)):
        for j in range(i + 1, len(poses)):
            point, w = helpers["closest_point_2_lines"](
                poses[i, :3, 3], poses[i, :3, 2], poses[j, :3, 3], poses[j, :3, 2])
            if w > 0.01:
                total += point * w
                weight += w
    poses[:, :3, 3] -= total / weight
    poses[:, :3, 3] *= 4 / np.linalg.norm(poses[:, :3, 3], axis=-1).mean()
    poses[:, :3, 1:3] *= -1  # Blender to COLMAP, as in dataset_readers.py.
    return poses, int(width) // 2, int(height) // 2, float(focal) / 2


def training_ssim(image, target):
    """11x11 SSIM with zero padding and fused FP32 accumulation."""
    from fused_ssim import fused_ssim

    return fused_ssim(image[None].contiguous(), target[None].contiguous(),
                      padding="same", train=True)



def load_native():
    initialize_cache()
    spec = importlib.util.spec_from_file_location(
        "native_4dgs_rasterizer", UPSTREAM / "gaussian_renderer/diff_gaussian_rasterization.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module



def settings(native, c2w, width, height, focal, frame, degree, degree_t):
    graphics = runpy.run_path(str(UPSTREAM / "utils/graphics_utils.py"))
    view = torch.tensor(np.linalg.inv(c2w), device="cuda", dtype=torch.float32).T.contiguous()
    fovx, fovy = 2 * np.arctan(width / (2 * focal)), 2 * np.arctan(height / (2 * focal))
    projection = graphics["getProjectionMatrixCenterShift"](
        .01, 100, width / 2, height / 2, focal, focal, width, height).cuda().T
    return native.GaussianRasterizationSettings(
        image_height=height, image_width=width,
        tanfovx=float(np.tan(fovx / 2)), tanfovy=float(np.tan(fovy / 2)),
        bg=torch.zeros(3, device="cuda"), scale_modifier=1.0,
        viewmatrix=view, projmatrix=(view @ projection).contiguous(),
        sh_degree=degree, sh_degree_t=degree_t,
        campos=torch.tensor(c2w[:3, 3], device="cuda", dtype=torch.float32),
        timestamp=frame / 30, time_duration=10.0,
        rot_4d=True, gaussian_dim=4, force_sh_3d=False, prefiltered=False, debug=False)
