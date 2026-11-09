"""N3V checkpoint adapter for the pruned and fine-tuned 4DGS-1K model."""

from pathlib import Path
from types import SimpleNamespace


def n3v_cameras(path):
    """Apply the pinned 4DGS n3v2blender normalization to poses_bounds."""
    import numpy as np

    original = np.load(path)[:, :15].reshape(-1, 3, 5)
    height, width, focal = original[0, :, -1]
    poses = np.concatenate((original[..., 1:2], original[..., 0:1],
                            -original[..., 2:3], original[..., 3:4]), -1)
    poses = np.concatenate((poses, np.tile([[[0, 0, 0, 1]]], (len(poses), 1, 1))), 1)
    poses[:, :3, 1:3] *= -1
    poses = poses[:, [1, 0, 2, 3], :]
    poses[:, 2, :] *= -1
    up = poses[:, :3, 1].sum(0)
    up /= np.linalg.norm(up)
    cross = np.cross(up, [0, 0, 1])
    skew = np.array([[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]],
                     [-cross[1], cross[0], 0]])
    rotation = np.eye(4)
    rotation[:3, :3] = np.eye(3) + skew + skew @ skew * (
        (1-up[2]) / (np.linalg.norm(cross)**2 + 1e-10))
    poses = rotation @ poses
    total, weight = np.zeros(3), 0.0
    for i in range(len(poses)):
        for j in range(i+1, len(poses)):
            oa, ob = poses[i, :3, 3], poses[j, :3, 3]
            da, db = poses[i, :3, 2], poses[j, :3, 2]
            da, db = da/np.linalg.norm(da), db/np.linalg.norm(db)
            cross = np.cross(da, db)
            w = np.linalg.norm(cross)**2
            delta = ob-oa
            ta = min(0, np.linalg.det([delta, db, cross])/(w+1e-10))
            tb = min(0, np.linalg.det([delta, da, cross])/(w+1e-10))
            if w > .01:
                total += (oa+ta*da+ob+tb*db)*.5*w
                weight += w
    poses[:, :3, 3] -= total/weight
    poses[:, :3, 3] *= 4/np.linalg.norm(poses[:, :3, 3], axis=-1).mean()
    poses[:, :3, 1:3] *= -1
    return poses, int(width)//2, int(height)//2, float(focal)/2


class Explicit4DCheckpoint:
    """Load a 1K checkpoint and its training-view contribution masks.

    Each call gathers an independent window's active primitives. A single-frame
    call uses a two-keyframe mask; a Dataflow call uses the union of masks
    needed by its requested frames.
    """

    def __init__(self, path, images, device="cuda"):
        import numpy as np
        import torch

        path, images = Path(path), Path(images)
        with torch.serialization.safe_globals([
            np._core.multiarray.scalar, np.dtype,
            type(np.dtype("float64")), type(np.dtype("float32")),
        ]):
            payload = torch.load(path, weights_only=True, mmap=True, map_location="cpu")
        if len(payload) != 3:
            raise ValueError("explicit4d CUDA evaluation requires the pruned 1K model and masks")
        fields, iteration, metadata = payload
        if (len(fields) != 19 or fields[0] != 3 or fields[18] != 2 or
                not fields[16] or fields[17] is not None or
                not isinstance(metadata["source_iteration"], int) or
                metadata["source_iteration"] <= 0):
            raise ValueError("unsupported 1K checkpoint schema or source training state")
        steps = metadata.get("finetune_steps")
        if (not isinstance(steps, int) or steps <= 0 or
                iteration != metadata["source_iteration"] + steps):
            raise ValueError("complete the recorded masked 1K fine-tuning budget before evaluation")
        if not torch.equal(metadata["keyframes"], torch.arange(0, 301, 20, dtype=torch.int32)):
            raise ValueError("1K keyframes must cover 0..300 at 20-frame intervals")
        count = len(fields[1])
        masks = metadata["masks"]
        if masks.shape != (16, count) or masks.dtype != torch.bool:
            raise ValueError("1K contribution masks must match the pruned Gaussian count")
        self.xyz = fields[1].to(device)
        self.sh = torch.cat((fields[2], fields[3]), 1).to(device)
        self.scales4d = torch.cat((fields[4], fields[14]), -1).to(device).exp()
        self.rotation_l = torch.nn.functional.normalize(fields[5].to(device), dim=-1)
        self.rotation_r = torch.nn.functional.normalize(fields[15].to(device), dim=-1)
        self.temporal_means = fields[13].to(device).flatten()
        self.opacity = fields[6].to(device).sigmoid().flatten()
        self.masks = masks.to(device)
        camera = metadata.get("camera")
        if camera is None:
            poses, self.width, self.height, focal = n3v_cameras(images.parent / "poses_bounds.npy")
            pose, fx, fy = poses[0], focal, focal
            cx, cy = self.width / 2, self.height / 2
        else:
            pose = np.asarray(camera["camera_to_world"], dtype=np.float64)
            self.width, self.height = int(camera["width"]), int(camera["height"])
            fx, fy, cx, cy = (float(camera[key]) for key in ("fx", "fy", "cx", "cy"))
            if (pose.shape != (4, 4) or not np.isfinite(pose).all() or
                    self.width < 1 or self.height < 1 or min(fx, fy) <= 0 or
                    not np.isfinite([fx, fy, cx, cy]).all()):
                raise ValueError("invalid checkpoint camera calibration")
        c2w = torch.tensor(pose, dtype=torch.float32, device=device)
        self.view = torch.tensor(np.linalg.inv(pose), dtype=torch.float32,
                                 device=device)[None].contiguous()
        self.camera_position = c2w[:3, 3].contiguous()
        self.intrinsics = torch.tensor([[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]],
                                       device=device, dtype=torch.float32)
        self.name, self.duration = path.stem, int(metadata.get("frame_count", 300))
        if not 2 <= self.duration <= 300:
            raise ValueError("checkpoint frame count must lie in 2..300")
        self.workload, self.image_prefix = "explicit4d", "cam00_"
        self.config = {"kernel_size": .3, "near": .2, "far": float("inf"), "background": 0,
                       "stability_near": .01, "stability_far": 100.0}
        self.source_count = count
        self.iteration, self.source_iteration = iteration, metadata["source_iteration"]

    def select(self, times, first=None):
        if first is None:
            first = int(times[0].item())
        if first < 0 or first+len(times) > self.duration or not 1 <= len(times) <= 5:
            raise ValueError("1K window must contain one to five in-range N3V frames")
        left, right = first//20, (first+len(times)-1)//20
        # Keep mask filtering, prefix/select and attribute gathering inside
        # every measured call. Only the representation's keyframe masks persist.
        ids = self.masks[left:right+2].any(0).nonzero().flatten()
        values = {name: getattr(self, name).index_select(0, ids) for name in (
            "xyz", "sh", "scales4d", "rotation_l", "rotation_r", "temporal_means", "opacity")}
        return SimpleNamespace(**values, source_ids=ids, source_count=self.source_count,
            masks=self.masks, frame_times=times, width=self.width, height=self.height,
            view=self.view, intrinsics=self.intrinsics, camera_position=self.camera_position,
            config=self.config, workload=self.workload)
