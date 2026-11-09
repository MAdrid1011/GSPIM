"""Anchor initialization, four-dimensional decoding, growth and training."""

import numpy as np
from PIL import Image
import torch

from .common import (ROOT, TRAINING, FUSED_SSIM_COMMIT, ANCHORED_TRAINING_STEPS,
                     load_native, settings, training_ssim)


class AnchoredModel(torch.nn.Module):
    """Four-dimensional anchors with ten bindings and three feature decoders."""

    choices = dict(feature_dim=32, hidden_dim=32, bindings=10, tau_t=.05,
                   growth_decay=.1, timespan_quantile=.5, spatial_voxel=.001,
                   temporal_voxel=.033, static_after=1500,
                   covariance_scale="base_times_sigmoid",
                   static_geometry="mean_at_temporal_center_no_temporal_decay",
                   opacity="tanh", covariance_input="feature", color_input="feature_view")

    def __init__(self, anchors, spatial_scales):
        super().__init__()
        count, device = len(anchors), anchors.device
        self.register_buffer("anchors", anchors)
        self.offsets = torch.nn.Parameter(torch.zeros(count, 10, 4, device=device))
        self.features = torch.nn.Parameter(torch.zeros(count, 32, device=device))
        scales = torch.cat((spatial_scales[:, None].expand(-1, 3),
                            anchors.new_full((count, 1), 1.414)), -1)
        # sigmoid(0)=1/2; initialize the decoded scales, not just the base.
        self.log_scales = torch.nn.Parameter((2*scales[:, None].expand(-1, 10, -1)).log().clone())
        self.opacity_mlp = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.ReLU(),
                                              torch.nn.Linear(32, 10), torch.nn.Tanh()).to(device)
        self.covariance_mlp = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.ReLU(),
                                                 torch.nn.Linear(32, 120)).to(device)
        self.color_mlp = torch.nn.Sequential(torch.nn.Linear(35, 32), torch.nn.ReLU(),
                                            torch.nn.Linear(32, 30), torch.nn.Sigmoid()).to(device)
        with torch.no_grad():
            self.opacity_mlp[2].weight.zero_()
            self.opacity_mlp[2].bias.fill_(float(np.arctanh(.1)))
            self.covariance_mlp[2].weight.zero_()
            bias = self.covariance_mlp[2].bias.view(10, 12)
            bias.zero_()
            bias[:, 4] = bias[:, 8] = 1
        for name, shape in (("static", (count,)), ("opacity_sum", (count,)),
                            ("visits", (count,)), ("gradient_sum", (count, 10)),
                            ("gradient_count", (count, 10))):
            self.register_buffer(name, torch.zeros(shape, device=device,
                dtype=torch.bool if name == "static" else anchors.dtype))

    @staticmethod
    def rotation(left, right):
        left = torch.nn.functional.normalize(left, dim=-1)
        right = torch.nn.functional.normalize(right, dim=-1)
        a, b, c, d = left.unbind(-1)
        p, q, r, s = right.unbind(-1)
        lhs = torch.stack((a, -b, -c, -d, b, a, -d, c,
                           c, d, a, -b, d, -c, b, a), -1).reshape(*left.shape[:-1], 4, 4)
        rhs = torch.stack((p, -q, -r, -s, q, p, s, -r,
                           r, -s, p, q, s, r, -q, p), -1).reshape(*right.shape[:-1], 4, 4)
        return lhs @ rhs

    def decode(self, camera_position):
        predicted = self.covariance_mlp(self.features).view(-1, 10, 12)
        scales = self.log_scales.exp()*predicted[..., :4].sigmoid()
        rotation = self.rotation(predicted[..., 4:8], predicted[..., 8:12])
        transform = rotation*scales[..., None, :]
        covariance = transform @ transform.transpose(-1, -2)
        direction = torch.nn.functional.normalize(self.anchors[:, :3]-camera_position, dim=-1)
        colors = self.color_mlp(torch.cat((self.features, direction), -1)).view(-1, 10, 3)
        return self.anchors[:, None]+self.offsets, covariance, self.opacity_mlp(self.features), colors, scales

    def slice(self, timestamp, camera_position):
        mean, covariance, neural_opacity, colors, scales = self.decode(camera_position)
        variance = covariance[..., 3, 3].clamp_min(1e-12)
        cross = covariance[..., :3, 3]
        velocity = cross/variance[..., None]
        delta = timestamp-mean[..., 3]
        dynamic = ~self.static[:, None]
        position = mean[..., :3]+velocity*torch.where(dynamic, delta, 0)[..., None]
        spatial = covariance[..., :3, :3]-cross[..., :, None]*velocity[..., None, :]
        marginal = torch.where(dynamic, torch.exp(-.5*delta.square()/variance), 1)
        opacity = neural_opacity*marginal
        active = ((marginal > self.choices["tau_t"]) & (neural_opacity > 0)).reshape(-1)
        ids = active.nonzero().flatten()
        return dict(means=position.reshape(-1, 3)[ids], covars=spatial.reshape(-1, 3, 3)[ids],
                    opacities=opacity.reshape(-1)[ids], colors=colors.reshape(-1, 3)[ids],
                    ids=ids, velocity=velocity, variance=variance,
                    neural_opacity=neural_opacity, all_means=position,
                    volume=scales[..., :3].prod(-1).mean())

    def optimizer(self, extent):
        return torch.optim.Adam([
            dict(params=[self.offsets], lr=.01*extent, name="offsets"),
            dict(params=[self.features], lr=.0075, name="features"),
            dict(params=[self.log_scales], lr=.007, name="log_scales"),
            dict(params=list(self.opacity_mlp.parameters()), lr=.002, name="opacity_mlp"),
            dict(params=list(self.covariance_mlp.parameters()), lr=.004, name="covariance_mlp"),
            dict(params=list(self.color_mlp.parameters()), lr=.008, name="color_mlp")], eps=1e-15)

    @torch.no_grad()
    def intervals(self):
        """Export conservative S1 metadata, without storing decoded attributes."""
        mean, covariance, _, _, _ = self.decode(self.anchors.new_zeros(3))
        radius = torch.sqrt(-2*covariance[..., 3, 3].clamp_min(1e-12)*
                            np.log(self.choices["tau_t"]))
        center = mean[..., 3]
        margin = 32*torch.finfo(center.dtype).eps*(1+center.abs()+radius)
        lower = (center-radius-margin).amin(-1)
        upper = (center+radius+margin).amax(-1)
        lower = torch.where(self.static, 0, lower)
        upper = torch.where(self.static, 10, upper)
        return torch.stack((lower, upper), -1)

    @torch.no_grad()
    def replace_rows(self, optimizer, kept, additions=None):
        """Resize anchor fields and Adam moments together, preserving MLP state."""
        additions = {} if additions is None else additions
        extra = len(additions.get("anchors", ()))
        for name in ("offsets", "features", "log_scales"):
            group = next(g for g in optimizer.param_groups if g["name"] == name)
            original = getattr(self, name)
            tail = additions.get(name, original.new_empty((0, *original.shape[1:])))
            parameter = torch.nn.Parameter(torch.cat((original[kept], tail)))
            state = optimizer.state.pop(original, None)
            if state is not None:
                for key in ("exp_avg", "exp_avg_sq"):
                    state[key] = torch.cat((state[key][kept], torch.zeros_like(tail)))
                optimizer.state[parameter] = state
            group["params"] = [parameter]
            setattr(self, name, parameter)
        for name in ("anchors", "static", "opacity_sum", "visits", "gradient_sum", "gradient_count"):
            original = getattr(self, name)
            tail = additions.get(name, original.new_zeros((extra, *original.shape[1:])))
            setattr(self, name, torch.cat((original[kept], tail)))

    @torch.no_grad()
    def adjust(self, optimizer, iteration, camera_position):
        """4D voxel growth, timespan weighting, velocity selection and pruning."""
        mean, covariance, _, _, _ = self.decode(camera_position)
        variance = covariance[..., 3, 3].clamp_min(1e-12)
        speed = (covariance[..., :3, 3]/variance[..., None]).norm(dim=-1)
        observed = self.visits > 80
        if iteration >= self.choices["static_after"]:
            self.static |= observed & (speed.sum(-1) < .1)
        duration = 2*torch.sqrt(-2*variance*np.log(self.choices["tau_t"]))
        reference = torch.quantile(duration, self.choices["timespan_quantile"]).clamp_min(1e-6)
        threshold = .0002*self.choices["growth_decay"]**((iteration-1500)/13500)
        threshold = threshold*duration/reference
        gradient = self.gradient_sum/self.gradient_count.clamp_min(1)
        eligible = ((self.gradient_count > 40) | (speed > .01)) & ~self.static[:, None]
        clear = self.gradient_count > 40
        self.gradient_sum[clear] = 0
        self.gradient_count[clear] = 0
        positions = mean.reshape(-1, 4)
        original_features = self.features.detach().clone()
        before = len(self.anchors)
        for level, factor in enumerate((16, 4, 1)):
            selected = ((gradient >= threshold*(2**level)) & eligible &
                        (torch.rand_like(gradient) > .5**(level+1))).reshape(-1)
            if not bool(selected.any()):
                continue
            size = .001*factor
            voxel = self.anchors.new_tensor([size, size, size, .033])
            existing = torch.round(self.anchors/voxel).to(torch.int64)
            proposed = torch.round(positions[selected]/voxel).to(torch.int64)
            unique, inverse = torch.unique(proposed, dim=0, return_inverse=True)
            joined, groups = torch.unique(torch.cat((existing, unique)), dim=0, return_inverse=True)
            occupied = torch.zeros(len(joined), dtype=torch.bool, device=joined.device)
            occupied[groups[:len(existing)]] = True
            new = ~occupied[groups[len(existing):]]
            if not bool(new.any()):
                continue
            feature = original_features[:, None].expand(-1, 10, -1).reshape(-1, 32)[selected]
            pooled = feature.new_full((len(unique), 32), -torch.inf)
            pooled.scatter_reduce_(0, inverse[:, None].expand(-1, 32), feature, reduce="amax")
            anchors = unique[new].to(voxel.dtype)*voxel
            count = len(anchors)
            scale = anchors.new_tensor([2*size, 2*size, 2*size, 2*1.414]).log()
            additions = dict(anchors=anchors, features=pooled[new],
                offsets=anchors.new_zeros((count, 10, 4)),
                log_scales=scale[None, None].expand(count, 10, 4).clone())
            self.replace_rows(optimizer, torch.arange(len(self.anchors), device=anchors.device), additions)
        grown = len(self.anchors)-before
        # Scaffold-GS compares the summed bound opacity against the number of
        # anchor visits. Stats for new anchors are zero and cannot prune them.
        ready = self.visits > 80
        prune = ready & (self.opacity_sum < .005*self.visits)
        removed = int(prune.sum())
        self.opacity_sum[ready] = 0
        self.visits[ready] = 0
        if removed:
            self.replace_rows(optimizer, (~prune).nonzero().flatten())
        return grown, removed, int(self.static.sum())



def anchored_inputs(scene, device="cpu"):
    """Initialize four-dimensional anchors in the original COLMAP coordinate frame."""
    import io
    import json
    import zipfile
    from plyfile import PlyData
    from scipy.spatial import cKDTree

    root = ROOT / "datasets/n3v" / scene
    original = np.load(root / "poses_bounds.npy")[:, :15].reshape(-1, 3, 5)
    poses = np.tile(np.eye(4), (len(original), 1, 1))
    poses[:, :3, :3] = original[:, :, :3][:, :, [1, 0, 2]]*np.array([1, 1, -1])
    poses[:, :3, 3] = original[:, :, 3]
    with zipfile.ZipFile(ROOT / "checkpoints/ex4dgs" / f"{scene}.zip") as archive:
        vertex = PlyData.read(io.BytesIO(archive.read(next(
            n for n in archive.namelist() if n.endswith("input.ply")))))["vertex"].data
        camera_list = json.loads(archive.read(next(
            n for n in archive.namelist() if n.endswith("cameras.json"))))
    # Source packages serialize cam00 first and training cameras in reverse
    # order. Match geometry, never use their row number as the N3V camera ID.
    for camera in camera_list:
        distances = np.linalg.norm(poses[:, :3, 3]-camera["position"], axis=-1)
        index = int(distances.argmin())
        if (distances[index] > 1e-10 or not np.allclose(
                poses[index, :3, :3], camera["rotation"], atol=1e-10, rtol=0)):
            raise ValueError("COLMAP points and N3V cameras are in different coordinate systems")
    xyz = np.stack([vertex[name] for name in ("x", "y", "z")], -1).astype(np.float64)
    times = np.random.default_rng(0).uniform(0, 10, (len(xyz), 1))
    voxel = np.array([.001, .001, .001, .033])
    anchors = np.unique(np.round(np.concatenate((xyz, times), -1)/voxel), axis=0)*voxel
    distances = cKDTree(anchors[:, :3]).query(anchors[:, :3], k=4, workers=2)[0][:, 1:]
    scales = np.sqrt(np.maximum((distances**2).mean(-1), 1e-7))
    height, width, focal = original[0, :, -1]
    return (torch.tensor(anchors, dtype=torch.float32, device=device),
            torch.tensor(scales, dtype=torch.float32, device=device),
            poses, int(width)//2, int(height)//2, float(focal)/2)



def anchored_render(native, model, raster_settings, timestamp):
    """Differentiable four-dimensional slicing followed by 3D rasterization."""
    values = model.slice(timestamp, raster_settings.campos)
    covariance = values["covars"]
    packed = torch.stack((covariance[:, 0, 0], covariance[:, 0, 1], covariance[:, 0, 2],
                          covariance[:, 1, 1], covariance[:, 1, 2], covariance[:, 2, 2]), -1)
    screenspace = torch.zeros_like(values["means"], requires_grad=True)
    output = native.GaussianRasterizer(raster_settings._replace(
        rot_4d=False, gaussian_dim=3, force_sh_3d=True, sh_degree=0, sh_degree_t=0))(
        means3D=values["means"], means2D=screenspace,
        opacities=values["opacities"][:, None], colors_precomp=values["colors"],
        cov3D_precomp=packed, flow_2d=torch.zeros_like(values["means"][:, :2]))
    return output, values, screenspace



def train_anchored(scene="sear_steak", until=ANCHORED_TRAINING_STEPS):
    """Train anchor features, offsets and decoders using the registered image convention."""
    from concurrent.futures import ThreadPoolExecutor
    import time

    if not 1 <= until <= ANCHORED_TRAINING_STEPS:
        raise ValueError("Anchored short preparation ends at its recorded 2000-step budget")
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    anchors, scales, poses, width, height, focal = anchored_inputs(scene, "cuda")
    checkpoint = TRAINING / "anchored4dgs" / f"{scene}.pth"
    payload = torch.load(checkpoint, weights_only=True, map_location="cpu") if checkpoint.exists() else None
    if payload is not None:
        if payload["method"] != "anchored4dgs-paper-reconstruction" or payload["choices"] != AnchoredModel.choices:
            raise ValueError("checkpoint format or Anchored model parameters do not match")
        anchors = payload["model"]["anchors"].cuda()
        scales = anchors.new_ones(len(anchors))
    model = AnchoredModel(anchors, scales)
    centers = poses[1:, :3, 3]
    extent = float(np.linalg.norm(centers-centers.mean(0), axis=-1).max()*1.1)
    optimizer = model.optimizer(extent)
    image_root = ROOT / "datasets/n3v" / scene / "images"
    names = sorted(path.stem for path in image_root.parent.glob("cam[0-9][0-9].mp4"))
    if len(names) != len(poses) or names[0] != "cam00":
        raise ValueError("N3V camera names do not match the training poses")
    samples = [(camera, frame) for camera in range(1, len(poses)) for frame in range(300)]
    permutation, cursor, first = rng.permutation(len(samples)), 0, 0
    fused_from = 1
    if payload is not None:
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        first, cursor = payload["iteration"], payload["cursor"]
        fused_from = payload.get("fused_ssim_from_iteration", first + 1)
        permutation = payload["permutation"].numpy()
        rng.bit_generator.state = payload["numpy_rng"]
        torch.set_rng_state(payload["torch_rng"])
        torch.cuda.set_rng_state(payload["cuda_rng"])
        del payload
    if first >= until:
        print("Anchored already trained to", first, checkpoint, flush=True)
        return checkpoint
    ssim = training_ssim
    native = load_native()
    raster_settings = [settings(native, pose, width, height, focal, 0, 0, 0) for pose in poses]

    def read_image(sample):
        camera, frame = samples[int(sample)]
        with Image.open(image_root / f"{names[camera]}_{frame:04d}.png") as image:
            return np.array(image)

    print("Anchored real training", scene, "resume", first, "until", until,
          "anchors", len(model.anchors), "extent", extent, "choices", model.choices, flush=True)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as loader:
        prefetched = None
        for iteration in range(first+1, until+1):
            for group in optimizer.param_groups:
                if group["name"] == "offsets":
                    group["lr"] = .01*extent*(.0001/.01)**(iteration/30000)
                elif group["name"] == "opacity_mlp":
                    group["lr"] = .002*(.00002/.002)**(iteration/30000)
                elif group["name"] == "color_mlp":
                    group["lr"] = .008*(.00005/.008)**(iteration/30000)
            if cursor+2 > len(samples):
                permutation, cursor = rng.permutation(len(samples)), 0
            batch = permutation[cursor:cursor+2]
            cursor += 2
            images = prefetched or [loader.submit(read_image, sample) for sample in batch]
            prefetched = ([loader.submit(read_image, sample) for sample in permutation[cursor:cursor+2]]
                          if iteration < until and cursor+2 <= len(samples) else None)
            batch_loss = batch_psnr = 0.0
            for sample, future in zip(batch, images):
                pixels = future.result()
                camera, frame = samples[int(sample)]
                rs = raster_settings[camera]
                output, values, screen = anchored_render(native, model, rs, frame/30)
                image, radii = output[:2]
                target = torch.tensor(pixels, device="cuda", dtype=torch.float32).permute(2, 0, 1)/255
                loss = (.8*(image-target).abs().mean()+.2*(1-ssim(image, target))+.001*values["volume"])/2
                loss.backward()
                with torch.no_grad():
                    if 500 < iteration < 15000:
                        visible = radii > 0
                        ids = values["ids"][visible]
                        model.gradient_sum.view(-1)[ids] += screen.grad[visible, :2].norm(dim=-1)*2
                        model.gradient_count.view(-1)[ids] += 1
                        anchor_visible = native.GaussianRasterizer(rs).markVisible(model.anchors[:, :3].contiguous())
                        model.visits[anchor_visible] += 1
                        model.opacity_sum[anchor_visible] += values["neural_opacity"][anchor_visible].clamp_min(0).sum(-1)
                    batch_loss += loss.detach()
                    batch_psnr += -10*(image.detach().clamp(0, 1)-target).square().mean().log10()/2
                del output, values, screen, image, radii, target, loss
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if 1500 < iteration < 15000 and iteration % 100 == 0:
                grown, pruned, static = model.adjust(optimizer, iteration, raster_settings[1].campos)
                print("Anchored control", iteration, "grown", grown, "pruned", pruned,
                      "static", static, "anchors", len(model.anchors), flush=True)
            if iteration % 10 == 0:
                print("Anchored step", iteration, "loss", round(float(batch_loss), 6),
                      "training PSNR", round(float(batch_psnr), 4), "anchors", len(model.anchors),
                      "seconds/step", round((time.monotonic()-started)/(iteration-first), 3),
                      "peak MiB", round(torch.cuda.max_memory_allocated()/1024**2), flush=True)
            if iteration % 500 == 0 or iteration == until:
                if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
                    raise ValueError("non-finite Anchored model parameters")
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                payload = dict(method="anchored4dgs-paper-reconstruction", iteration=iteration,
                    training_steps=ANCHORED_TRAINING_STEPS,
                    choices=dict(model.choices), model=model.state_dict(), optimizer=optimizer.state_dict(),
                    intervals=model.intervals(),
                    cursor=cursor, permutation=torch.from_numpy(permutation.copy()),
                    numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(), source_doi="10.1145/3757377.3763898",
                    scaffold_commit="59c833b56bbbf510f3f64d40f81721995caced66",
                    images="N3V native bicubic half resolution; training excludes cam00",
                    growth="Scaffold hierarchy 16/4/1; 4D centers; new temporal scale 1.414",
                    statistics="per-view NDC gradients; neural opacity; visits before temporal gating",
                    training_renderer="pinned original 4DGS rasterizer in 3D covariance/color mode")
                payload.update(fused_ssim_commit=FUSED_SSIM_COMMIT,
                               fused_ssim_from_iteration=fused_from)
                temporary = checkpoint.with_suffix(".pth.part")
                torch.save(payload, temporary)
                temporary.replace(checkpoint)
                del payload
                print("saved", checkpoint, "step", iteration, flush=True)
    return checkpoint


def _source_training_views(scene):
    """Load synchronized training views while keeping the held-out camera separate."""
    import json
    from integrations.explicit4d import n3v_cameras

    dataset = ROOT / "datasets/n3v" / scene
    if (dataset / "transforms_train.json").is_file():
        training = json.loads((dataset / "transforms_train.json").read_text())["frames"]
        heldout = json.loads((dataset / "transforms_test.json").read_text())
        return dataset, training, heldout, "transforms_train.json"

    poses, width, height, focal = n3v_cameras(dataset / "poses_bounds.npy")
    videos = sorted(dataset.glob("cam[0-9][0-9].mp4"))
    if len(videos) != len(poses) or videos[0].stem != "cam00":
        raise ValueError("camera video IDs must match the supplied poses")
    training, test = [], []
    for pose, video in zip(poses, videos):
        matrix = pose.copy()
        matrix[:3, 1:3] *= -1
        for path in sorted((dataset / "images").glob(f"{video.stem}_*.png")):
            index = int(path.stem.rsplit("_", 1)[1])
            frame = dict(file_path=str(path.relative_to(dataset).with_suffix("")),
                         time=index / 30, transform_matrix=matrix.tolist(),
                         fl_x=focal, fl_y=focal, cx=width / 2, cy=height / 2)
            (test if video.stem == "cam00" else training).append(frame)
    if not training or not test:
        raise ValueError("training and held-out images are required")
    return dataset, training, dict(frames=test, w=width, h=height), "images/cam01.."


def _source_calibration(frame):
    pose = np.asarray(frame["transform_matrix"]).copy()
    pose[:3, 1:3] *= -1
    intrinsic = [[[frame["fl_x"], 0, frame["cx"]],
                  [0, frame["fl_y"], frame["cy"]], [0, 0, 1]]]
    return (torch.tensor(pose, dtype=torch.float32, device="cuda"),
            torch.tensor(intrinsic, dtype=torch.float32, device="cuda"))


def _render_shared_views(model, timestamp, poses, intrinsics, width, height):
    from gsplat import rasterization

    values = model.slice(timestamp, poses[0, :3, 3])
    colors = [values["colors"]]
    for pose in poses[1:]:
        direction = torch.nn.functional.normalize(model.anchors[:, :3]-pose[:3, 3], dim=-1)
        decoded = model.color_mlp(torch.cat((model.features, direction), -1)).view(-1, 3)
        colors.append(decoded[values["ids"]])
    rendered, _, _ = rasterization(
        values["means"], None, None, values["opacities"], torch.stack(colors),
        torch.linalg.inv(poses), intrinsics, width, height, near_plane=.2,
        eps2d=.3, packed=False, covars=values["covars"])
    return rendered, values["volume"]


def train_anchored_from_source(scene="sear_steak", source=None, until=None):
    """Fit the supplied Anchored model from a calibrated explicit 4D source."""
    from concurrent.futures import ThreadPoolExecutor
    import hashlib
    from pathlib import Path
    import time
    from fused_ssim import fused_ssim
    from integrations.manifest import WORKLOADS

    parameters = WORKLOADS["anchored"].parameters_for(scene)
    until = parameters["training_steps"] if until is None else until
    if not 1 <= until <= parameters["training_steps"]:
        raise ValueError("training exceeds the supplied model's recorded update budget")
    source = (ROOT / "checkpoints/training/4dgs" / f"{scene}-trained.pth"
              if source is None else Path(source))
    if not source.is_absolute():
        source = ROOT / source
    if not source.is_file():
        raise FileNotFoundError(source)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    fields, source_iteration = torch.load(
        source, map_location="cpu", weights_only=False, mmap=True)[:2]
    if source_iteration != parameters["source_iteration"]:
        raise ValueError("source checkpoint iteration does not match the workload manifest")

    points = fields[1].detach()
    grouping = points
    lower, upper = grouping.amin(0), grouping.amax(0)
    voxel = ((grouping-lower)/(upper-lower).clamp_min(1e-6)*1023).long()
    morton = torch.zeros(len(points), dtype=torch.int64)
    for bit in range(10):
        for axis in range(3):
            morton |= ((voxel[:, axis] >> bit) & 1) << (3*bit+axis)
    count = len(points)//10
    ids = morton.argsort(stable=True)[:count*10].reshape(count, 10)
    means = torch.cat((points, fields[13].detach()), -1)[ids].cuda()
    anchors = means.mean(1).contiguous()
    model = AnchoredModel(anchors, fields[4][ids].detach().exp().mean((1, 2)).cuda())

    with torch.no_grad():
        model.offsets.copy_(means-anchors[:, None])
        model.log_scales.copy_(torch.cat((fields[4][ids], fields[14][ids]), -1).cuda()+np.log(2))
        opacity = fields[6][ids].sigmoid().clamp(1e-5, 1-1e-5).cuda()
        color = (fields[2][ids, 0]*.28209479177387814+.5).clamp(1e-4, 1-1e-4).reshape(count, 30).cuda()
        left = torch.nn.functional.normalize(
            fields[5][ids].cuda()*anchors.new_tensor([1, -1, 1, -1]), dim=-1)
        right = torch.nn.functional.normalize(
            fields[15][ids].cuda()*anchors.new_tensor([1, -1, 1, 1]), dim=-1)
        covariance = torch.cat((torch.zeros_like(left), left, right), -1).reshape(count, 120)
        desired = torch.cat((opacity.atanh().reshape(count, 10), covariance, color.logit()), -1)
        center = desired.mean(0)
        centered = desired-center
        _, _, basis = torch.pca_lowrank(centered, q=32, center=False, niter=3)
        features = centered@basis
        shift = (-features.amin(0)).clamp_min(0)+1
        model.features.copy_(features+shift)
        reconstruction = center-shift@basis.T
        for decoder, lo, hi in ((model.opacity_mlp, 0, 10),
                                (model.covariance_mlp, 10, 130),
                                (model.color_mlp, 130, 160)):
            decoder[0].weight.zero_()
            decoder[0].bias.zero_()
            decoder[0].weight[:, :32].copy_(torch.eye(32, device="cuda"))
            decoder[2].weight.copy_(basis[lo:hi])
            decoder[2].bias.copy_(reconstruction[lo:hi])

    dataset, frames, heldout, view_source = _source_training_views(scene)
    by_time = {}
    for index, frame in enumerate(frames):
        by_time.setdefault(round(frame["time"]*30), []).append(index)
    width, height = heldout["w"], heldout["h"]
    centers = np.stack([np.asarray(frame["transform_matrix"])[:3, 3] for frame in frames])
    extent = float(np.linalg.norm(centers-centers.mean(0), axis=-1).max()*1.1)
    optimizer = torch.optim.Adam(model.optimizer(extent).param_groups, eps=1e-15, fused=True)
    rates = dict(offsets=.00016*extent, features=.0025, log_scales=.005,
                 opacity_mlp=.0001, covariance_mlp=.0001, color_mlp=.0001)
    for group in optimizer.param_groups:
        group["lr"] = rates[group["name"]]

    rng = np.random.default_rng(0)

    def read(index):
        frame = frames[index]
        with Image.open(dataset / (frame["file_path"]+".png")) as image:
            pixels = np.array(image.convert("RGB"))
        return frame, _source_calibration(frame), pixels

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as loader:
        pending = None
        for iteration in range(1, until+1):
            if pending is None:
                first = int(rng.integers(0, len(frames)))
                peers = [value for value in by_time[round(frames[first]["time"]*30)] if value != first]
                batch = [first, int(rng.choice(peers))]
                pending = [loader.submit(read, index) for index in batch]
            samples = [future.result() for future in pending]
            first = int(rng.integers(0, len(frames)))
            peers = [value for value in by_time[round(frames[first]["time"]*30)] if value != first]
            pending = [loader.submit(read, index) for index in (first, int(rng.choice(peers)))]
            poses = torch.stack([camera[0] for _, camera, _ in samples])
            intrinsics = torch.cat([camera[1] for _, camera, _ in samples])
            target = torch.from_numpy(np.stack([pixels for _, _, pixels in samples])).cuda().float()/255
            optimizer.zero_grad(set_to_none=True)
            rendered, volume = _render_shared_views(
                model, samples[0][0]["time"], poses, intrinsics, width, height)
            similarity = fused_ssim(rendered.permute(0, 3, 1, 2).contiguous(),
                                    target.permute(0, 3, 1, 2).contiguous())
            loss = .8*(rendered-target).abs().mean()+.2*(1-similarity)+.001*volume
            loss.backward()
            optimizer.step()
            if iteration % 100 == 0 or iteration == until:
                psnr = -10*(rendered.detach().clamp(0, 1)-target).square().mean().log10()
                print("Anchored source step", iteration, "loss", round(float(loss), 6),
                      "training PSNR", round(float(psnr), 4), flush=True)

    excluded = {"opacity_sum", "visits", "gradient_sum", "gradient_count"}
    state = {name: value.detach().cpu().contiguous()
             for name, value in model.state_dict().items() if name not in excluded}
    first = heldout["frames"][0]
    pose, _ = _source_calibration(first)
    checkpoint = TRAINING / "anchored4dgs" / f"{scene}.pth"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(method="anchored4dgs-paper-reconstruction", iteration=until,
        training_steps=until, choices=dict(model.choices), model=state,
        intervals=model.intervals().cpu(), inference_only=True,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_iteration=source_iteration, training_views=view_source,
        training_batch_size=2, grouping_dimensions=3, shared_geometry=True,
        training_seconds=time.monotonic()-started,
        camera=dict(camera_to_world=pose.cpu().tolist(), width=width, height=height,
                    fx=first["fl_x"], fy=first["fl_y"], cx=first["cx"], cy=first["cy"]),
        frame_count=len(heldout["frames"]))
    temporary = checkpoint.with_suffix(".pth.part")
    torch.save(payload, temporary)
    temporary.replace(checkpoint)
    return checkpoint



def export_checkpoint(source, destination):
    """Export trained anchor fields, decoders and conservative intervals."""
    from pathlib import Path
    from gspim.prepare import sha256

    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("training and inference checkpoint paths must differ")
    payload = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    if payload["iteration"] != payload["training_steps"]:
        raise ValueError("finish the recorded training budget before export")
    training_keys = {"optimizer", "cursor", "permutation", "numpy_rng", "torch_rng", "cuda_rng"}
    payload = {key: value for key, value in payload.items() if key not in training_keys}
    statistic_keys = {"opacity_sum", "visits", "gradient_sum", "gradient_count"}
    payload["model"] = {key: value.detach().cpu().clone() for key, value in payload["model"].items()
                        if key not in statistic_keys}
    payload.update(inference_only=True, training_checkpoint_sha256=sha256(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return destination
