"""N3V checkpoint adapter for Anchored 4DGS."""

from pathlib import Path
from types import SimpleNamespace


class AnchoredCheckpoint:
    """Load trained anchor features, binding parameters and conservative intervals.

    Only interval metadata persists across frames. Selection and MLP decoding
    remain inside every measured call; a Dataflow window shares its own decode.
    """

    def __init__(self, path, images, device="cuda"):
        import numpy as np
        import torch

        path, images = Path(path), Path(images)
        payload = torch.load(path, weights_only=True, mmap=True, map_location="cpu")
        if payload.get("method") != "anchored4dgs-paper-reconstruction":
            raise ValueError("unsupported Anchored checkpoint format")
        steps = payload.get("training_steps")
        if (not isinstance(steps, int) or steps <= 0 or
                payload["iteration"] != steps):
            raise ValueError("complete the recorded Anchored training budget before evaluation")
        choices = payload["choices"]
        supported = dict(feature_dim=32, hidden_dim=32, bindings=10,
                         opacity="tanh", covariance_input="feature", color_input="feature_view",
                         covariance_scale="base_times_sigmoid",
                         static_geometry="mean_at_temporal_center_no_temporal_decay")
        if any(choices.get(key) != value for key, value in supported.items()):
            raise ValueError("unsupported Anchored decoder or static geometry convention")
        self.tau_t = float(choices["tau_t"])
        if not 0 < self.tau_t < 1:
            raise ValueError("Anchored temporal threshold must lie between zero and one")
        state = payload["model"]
        self.anchor_positions = state["anchors"].to(device)
        self.anchor_offsets = state["offsets"].to(device)
        self.anchor_features = state["features"].to(device)
        self.anchor_log_scales = state["log_scales"].to(device)
        self.anchor_static = state["static"].to(device)
        self.intervals = payload["intervals"].to(device)
        count = len(self.anchor_positions)
        self.anchor_count = count
        self.static_anchor_count = int(state["static"].sum())
        shapes = ((self.anchor_positions, (count, 4)), (self.anchor_offsets, (count, 10, 4)),
                  (self.anchor_features, (count, 32)), (self.anchor_log_scales, (count, 10, 4)),
                  (self.intervals, (count, 2)))
        if (not count or self.anchor_static.shape != (count,) or self.anchor_static.dtype != torch.bool or
                any(value.shape != shape or value.dtype != torch.float32 or
                    not bool(torch.isfinite(value).all()) for value, shape in shapes) or
                not bool((self.intervals[:, 0] <= self.intervals[:, 1]).all())):
            raise ValueError("invalid Anchored binding tensors or interval metadata")

        def decoder(name, inputs, outputs, activation=None):
            layers = [torch.nn.Linear(inputs, 32), torch.nn.ReLU(), torch.nn.Linear(32, outputs)]
            if activation is not None:
                layers.append(activation)
            module = torch.nn.Sequential(*layers).to(device)
            prefix = name+"."
            module.load_state_dict({key[len(prefix):]: value for key, value in state.items()
                                    if key.startswith(prefix)})
            return module.eval().requires_grad_(False)

        self.opacity_mlp = decoder("opacity_mlp", 32, 10, torch.nn.Tanh())
        self.covariance_mlp = decoder("covariance_mlp", 32, 120)
        self.color_mlp = decoder("color_mlp", 35, 30, torch.nn.Sigmoid())
        original = np.load(images.parent / "poses_bounds.npy")[:, :15].reshape(-1, 3, 5)
        height, width, focal = original[0, :, -1]
        self.width, self.height = int(width)//2, int(height)//2
        focal = float(focal)/2
        c2w = np.eye(4)
        c2w[:3, :3] = original[0, :, :3][:, [1, 0, 2]]*np.array([1, 1, -1])
        c2w[:3, 3] = original[0, :, 3]
        self.view = torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device=device)[None].contiguous()
        self.camera_position = torch.tensor(c2w[:3, 3], dtype=torch.float32, device=device)
        self.intrinsics = torch.tensor([[[focal, 0, self.width/2], [0, focal, self.height/2],
                                         [0, 0, 1]]], dtype=torch.float32, device=device)
        self.xyz = self.anchor_positions[:, :3].contiguous()
        self.sh = self.xyz.new_empty((0, 0, 3))
        self.name, self.duration = path.stem, 300
        self.iteration = payload["iteration"]
        self.workload, self.image_prefix = "anchored", "cam00_"
        self.config = dict(kernel_size=.3, near=.2, far=float("inf"), background=0,
                           stability_near=.01, stability_far=100.0)

    def select(self, times, first=None):
        if first is None:
            first = int(times[0].item())
        if first < 0 or first+len(times) > self.duration or not 1 <= len(times) <= 5:
            raise ValueError("Anchored window must contain one to five in-range N3V frames")
        # Intervals conservatively enclose every bound Gaussian's temporal
        # support. Preserve ascending anchor/binding IDs for equal-depth ties.
        ids = ((self.intervals[:, 0] <= times[-1]/30) &
               (self.intervals[:, 1] >= times[0]/30)).nonzero().flatten()
        values = {name: getattr(self, name).index_select(0, ids) for name in (
            "anchor_positions", "anchor_offsets", "anchor_features", "anchor_log_scales", "anchor_static")}
        return SimpleNamespace(**values, source_anchors=ids,
            opacity_mlp=self.opacity_mlp, covariance_mlp=self.covariance_mlp, color_mlp=self.color_mlp,
            xyz=values["anchor_positions"][:, :3], sh=self.sh, tau_t=self.tau_t,
            width=self.width, height=self.height, view=self.view, intrinsics=self.intrinsics,
            camera_position=self.camera_position, config=self.config, workload=self.workload)
