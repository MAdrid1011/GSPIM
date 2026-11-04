"""Original 4DGS continuation, contribution pruning and masked fine-tuning."""

import runpy

import numpy as np
from PIL import Image
import torch

from .common import (ROOT, CACHE, TRAINING, UPSTREAM, FUSED_SSIM_COMMIT,
                     cameras, initialize_cache, load_native, settings, training_ssim)


def read_payload(path):
    with torch.serialization.safe_globals([
        np._core.multiarray.scalar, np.dtype,
        type(np.dtype("float64")), type(np.dtype("float32")),
    ]):
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)



def read_checkpoint(scene, path=None):
    fields, iteration = read_payload(path or TRAINING / "4dgs" / f"{scene}.pth")[:2]
    assert len(fields) == 19 and fields[16]
    environment = fields[17]
    if environment is not None and (
            not torch.is_tensor(environment) or environment.ndim != 3 or
            environment.shape[0] != 3 or not environment.is_floating_point()):
        raise ValueError("unsupported original 4DGS environment map")
    return fields, iteration



def load_checkpoint(scene, path=None):
    fields, iteration = read_checkpoint(scene, path)
    values = {"means3D": fields[1].cuda(),
              "shs": torch.cat((fields[2], fields[3]), 1).cuda(),
              "scales": fields[4].cuda().exp(),
              "rotations": torch.nn.functional.normalize(fields[5].cuda(), dim=-1),
              "opacities": fields[6].cuda().sigmoid(),
              "ts": fields[13].cuda(), "scales_t": fields[14].cuda().exp(),
              "rotations_r": torch.nn.functional.normalize(fields[15].cuda(), dim=-1)}
    values["means2D"] = torch.zeros_like(values["means3D"])
    values["flow_2d"] = torch.zeros_like(values["means3D"][:, :2])
    return values, fields[0], fields[18], iteration



def trainable_model(scene, checkpoint=None, offload=False):
    """Load the pinned GaussianModel definitions for checkpoint continuation.

    Optimizer masks and zero allocations follow the device holding each Adam
    moment, allowing optimizer state to remain on CPU during training.
    """
    import ast
    import math
    from types import SimpleNamespace
    import yaml

    namespace = {"torch": torch, "np": np, "nn": torch.nn}
    namespace.update(runpy.run_path(str(UPSTREAM / "utils/graphics_utils.py")))
    namespace.update(runpy.run_path(str(UPSTREAM / "utils/sh_utils.py")))

    class MomentStorage(ast.NodeTransformer):
        def visit_Subscript(self, node):
            self.generic_visit(node)
            value = node.value
            if (isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name) and
                    value.value.id == "stored_state" and isinstance(value.slice, ast.Constant) and
                    value.slice.value in ("exp_avg", "exp_avg_sq") and
                    isinstance(node.slice, ast.Name) and node.slice.id == "mask"):
                node.slice = ast.Call(func=ast.Attribute(value=node.slice, attr="to", ctx=ast.Load()),
                    args=[ast.Attribute(value=value, attr="device", ctx=ast.Load())], keywords=[])
            return node

        def visit_Call(self, node):
            self.generic_visit(node)
            if (isinstance(node.func, ast.Attribute) and node.func.attr == "zeros_like" and
                    len(node.args) == 1 and isinstance(node.args[0], ast.Name) and
                    node.args[0].id in ("tensor", "extension_tensor")):
                device = ast.parse('stored_state["exp_avg"].device', mode="eval").body
                node.keywords.append(ast.keyword(arg="device", value=device))
            return node

    for filename, kind in (("utils/general_utils.py", ast.FunctionDef),
                           ("scene/gaussian_model.py", ast.ClassDef)):
        source = UPSTREAM / filename
        tree = ast.parse(source.read_text())
        tree.body = [node for node in tree.body if isinstance(node, kind)]
        if kind is ast.ClassDef:
            for model_class in tree.body:
                for method in model_class.body:
                    if isinstance(method, ast.FunctionDef) and method.name in (
                            "_prune_optimizer", "cat_tensors_to_optimizer", "replace_tensor_to_optimizer"):
                        MomentStorage().visit(method)
            ast.fix_missing_locations(tree)
        exec(compile(tree, str(source), "exec"), namespace)
    fields, iteration = read_checkpoint(scene, checkpoint)
    fields = list(fields)
    parameters = {1, 2, 3, 4, 5, 6, 13, 14, 15}
    for index, value in enumerate(fields):
        if torch.is_tensor(value):
            value = value.detach() if offload else value.detach().cuda()
            fields[index] = torch.nn.Parameter(value) if index in parameters else value
    config = yaml.safe_load((UPSTREAM / "configs/dynerf" / f"{scene}.yaml").read_text())
    model = namespace["GaussianModel"](3, gaussian_dim=4, time_duration=[0, 10],
                                        rot_4d=True, force_sh_3d=False, sh_degree_t=2)
    options = SimpleNamespace(**config["OptimizationParams"])
    model.restore(tuple(fields), options)
    if offload:
        # Restore Adam while parameters are on CPU, so load_state_dict never
        # allocates all moments on the GPU. Keep the same Parameter objects.
        for group in model.optimizer.param_groups:
            parameter = group["params"][0]
            parameter.data = parameter.data.cuda()
        for name in ("max_radii2D", "xyz_gradient_accum", "t_gradient_accum", "denom"):
            setattr(model, name, getattr(model, name).cuda())
    loss_source = UPSTREAM / "utils/loss_utils.py"
    tree = ast.parse(loss_source.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    namespace.update(F=torch.nn.functional, Variable=torch.autograd.Variable, exp=math.exp)
    exec(compile(tree, str(loss_source), "exec"), namespace)
    return model, options, namespace["ssim"], iteration



@torch.no_grad()
def offloaded_adam_step(optimizer, chunk_size=4*1024*1024):
    """Run PyTorch Adam with CPU moments in bounded chunks."""
    from torch.optim.adam import adam

    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            state = optimizer.state[parameter]
            if (not parameter.is_contiguous() or not parameter.grad.is_contiguous() or
                    set(state) != {"step", "exp_avg", "exp_avg_sq"} or
                    state["exp_avg"].device.type != "cpu" or group["amsgrad"]):
                raise ValueError("unsupported offloaded Adam layout/state")
            values, gradient = parameter.view(-1), parameter.grad.view(-1)
            average, square = state["exp_avg"].view(-1), state["exp_avg_sq"].view(-1)
            previous_step = state["step"].clone()
            for start in range(0, values.numel(), chunk_size):
                span = slice(start, start+chunk_size)
                mean_gpu = average[span].to(parameter.device)
                square_gpu = square[span].to(parameter.device)
                step = previous_step.clone()
                adam([values[span]], [gradient[span]], [mean_gpu], [square_gpu], [], [step],
                     foreach=group["foreach"], fused=group["fused"],
                     capturable=group["capturable"], differentiable=group["differentiable"],
                     amsgrad=False, beta1=group["betas"][0], beta2=group["betas"][1],
                     lr=group["lr"], weight_decay=group["weight_decay"],
                     eps=group["eps"], maximize=group["maximize"])
                average[span].copy_(mean_gpu)
                square[span].copy_(square_gpu)
                del mean_gpu, square_gpu
            state["step"].copy_(step)



def environment_background(environment, c2w, width, height, focal, device="cpu"):
    """Pinned Camera.get_rays and R=60 environment sampling."""
    view = np.linalg.inv(c2w)
    # Camera goes through getWorld2View2's double-precision inverse round trip.
    view = torch.tensor(np.linalg.inv(np.linalg.inv(view)), dtype=torch.float32).T
    # Camera initializes its center on CPU, then moves it and the view to CUDA.
    origin = view.inverse()[3, :3].to(device)
    camera_to_world = torch.linalg.inv(view.to(device).T)
    j, i = torch.meshgrid(torch.arange(height, dtype=torch.float32),
                          torch.arange(width, dtype=torch.float32), indexing="ij")
    points = torch.stack(((i+.5-width/2)/focal, (j+.5-height/2)/focal,
                          torch.ones_like(i), torch.ones_like(i)), -1).to(device)
    world = points @ camera_to_world.T
    direction = world[..., :3] - origin
    direction = direction / direction.norm(dim=-1, keepdim=True)
    dot = (origin*direction).sum(-1)
    norm = direction.square().sum(-1)
    delta = dot.square()-norm*(origin.square().sum()-60**2)
    if not bool((delta > 0).all()):
        raise ValueError("camera rays miss original 4DGS environment sphere")
    intersection = origin+direction*(-dot+delta.sqrt()/norm)[..., None]
    grid = torch.cat((torch.atan2(intersection[..., 1:2], intersection[..., :1]) /
                      (2*torch.pi)+.5, torch.acos(intersection[..., 2:3]/60)/torch.pi), -1)*2-1
    return torch.nn.functional.grid_sample(environment.detach().to(device)[None], grid[None],
                                           align_corners=False)[0]



def temporal_candidates(model, timestamp):
    """Conservative training allocation; the native 0.05 test still runs."""
    with torch.no_grad():
        a, b, c, d = model.get_rotation.unbind(-1)
        p, q, r, s = model.get_rotation_r.unbind(-1)
        row = torch.stack((a*s-b*r+c*q-d*p, a*r+b*s-c*p-d*q,
                           a*q-b*p-c*s+d*r, a*p+b*q+c*r+d*s), -1)
        variance = (row * model.get_scaling_xyzt).square().sum(-1)
        marginal = torch.exp(-.5 * (timestamp - model.get_t[:, 0]).square() / variance)
        return (marginal > .049).nonzero().flatten()



def train_parameters(model, ids=None):
    if ids is None:
        xyz, dc, rest, scale, rotation, opacity, times, scale_t, rotation_r = (
            model._xyz, model._features_dc, model._features_rest, model._scaling,
            model._rotation, model._opacity, model._t, model._scaling_t, model._rotation_r)
    else:
        xyz, dc, rest, scale, rotation, opacity, times, scale_t, rotation_r = (
            value.index_select(0, ids) for value in (
                model._xyz, model._features_dc, model._features_rest, model._scaling,
                model._rotation, model._opacity, model._t, model._scaling_t, model._rotation_r))
    return {"means3D": xyz, "means2D": torch.zeros_like(xyz, requires_grad=True),
            "shs": torch.cat((dc, rest), 1), "scales": scale.exp(),
            "rotations": torch.nn.functional.normalize(rotation, dim=-1),
            "opacities": opacity.sigmoid(), "ts": times, "scales_t": scale_t.exp(),
            "rotations_r": torch.nn.functional.normalize(rotation_r, dim=-1),
            "flow_2d": torch.zeros_like(xyz[:, :2])}



def native_buffers(native, parameters, raster_settings):
    """Read the pinned renderer's forward buffers without changing its source."""
    p, s = parameters, raster_settings
    empty = torch.empty(0, device=p["means3D"].device)
    result = native._C.rasterize_gaussians(
        s.bg, p["means3D"], empty, p["flow_2d"], p["opacities"], p["ts"],
        p["scales"], p["scales_t"], p["rotations"], p["rotations_r"],
        s.scale_modifier, empty, -1.0, s.viewmatrix, s.projmatrix,
        s.tanfovx, s.tanfovy, s.image_height, s.image_width, p["shs"],
        s.sh_degree, s.sh_degree_t, s.campos, s.timestamp, s.time_duration,
        s.rot_4d, s.gaussian_dim, s.force_sh_3d, s.prefiltered, False)
    count = len(p["means3D"])
    geometry = result[6]
    offset = 0
    fields = []
    for size, dtype in ((count, torch.float32), (count*3, torch.bool),
                        (count, torch.int32), (count*2, torch.float32),
                        (count*6, torch.float32), (count*4, torch.float32),
                        (count*3, torch.float32)):
        offset = (offset + 127) // 128 * 128
        length = size * dtype.itemsize
        fields.append(geometry[offset:offset + length].view(dtype))
        offset += length
    pixels = s.image_height * s.image_width
    ranges_offset = ((pixels*4 + 127) // 128 * 128 + pixels*4 + 127) // 128 * 128
    tiles = ((s.image_width+15)//16) * ((s.image_height+15)//16)
    ranges = result[8][ranges_offset:ranges_offset + tiles*8].view(torch.int32).reshape(-1, 2)
    point_list = result[7][:result[0]*4].view(torch.int32)
    return result, fields[3].reshape(-1, 2), fields[5].reshape(-1, 4), fields[6].reshape(-1, 3), ranges, point_list



def load_scoring():
    """Accumulate alpha-times-transmittance contributions on training views."""
    from torch.utils.cpp_extension import load_inline

    initialize_cache()
    return load_inline(
        name="gspim_4dgs_contribution", cpp_sources="""
void accumulate_scores(torch::Tensor xy, torch::Tensor conic,
    torch::Tensor ranges, torch::Tensor ids, torch::Tensor scores, int width, int height);
""", functions=["accumulate_scores"], cuda_sources=r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

__global__ void contribution_kernel(const float2* xy, const float4* conic,
    const uint2* ranges, const uint32_t* ids, double* scores, int W, int H) {
    const int rank = threadIdx.y*16 + threadIdx.x;
    const int x = blockIdx.x*16 + threadIdx.x;
    const int y = blockIdx.y*16 + threadIdx.y;
    const uint2 range = ranges[blockIdx.y*((W+15)/16) + blockIdx.x];
    __shared__ uint32_t collected_id[256];
    __shared__ float2 collected_xy[256];
    __shared__ float4 collected_conic[256];
    bool done = x >= W || y >= H;
    float T = 1.0f;
    for (uint32_t begin = range.x; begin < range.y; begin += 256) {
        if (__syncthreads_count(done) == 256) break;
        if (begin+rank < range.y) {
            uint32_t id = ids[begin+rank];
            collected_id[rank] = id;
            collected_xy[rank] = xy[id];
            collected_conic[rank] = conic[id];
        }
        __syncthreads();
        int length = min(256u, range.y-begin);
        for (int j = 0; j < length; ++j) {
            float weight = 0.0f;
            if (!done) {
                float2 point = collected_xy[j];
                float dx = point.x-float(x), dy = point.y-float(y);
                float4 c = collected_conic[j];
                float power = -0.5f*(c.x*dx*dx+c.z*dy*dy)-c.y*dx*dy;
                if (power <= 0.0f) {
                    float alpha = min(0.99f, c.w*exp(power));
                    if (alpha >= 1.0f/255.0f) {
                        float next_T = T*(1-alpha);
                        if (next_T < 0.0001f) done = true;
                        else { weight = alpha*T; T = next_T; }
                    }
                }
            }
            double sum = weight;
            for (int distance = 16; distance > 0; distance >>= 1)
                sum += __shfl_down_sync(0xffffffff, sum, distance);
            if ((rank&31) == 0 && sum > 0)
                atomicAdd(scores+collected_id[j], sum);
        }
        __syncthreads();
    }
}

void accumulate_scores(torch::Tensor xy, torch::Tensor conic,
    torch::Tensor ranges, torch::Tensor ids, torch::Tensor scores, int width, int height) {
    TORCH_CHECK(xy.is_cuda() && xy.is_contiguous() && xy.scalar_type() == torch::kFloat32);
    TORCH_CHECK(conic.is_cuda() && conic.is_contiguous() && conic.scalar_type() == torch::kFloat32);
    TORCH_CHECK(ranges.is_cuda() && ranges.is_contiguous() && ranges.scalar_type() == torch::kInt32);
    TORCH_CHECK(ids.is_cuda() && ids.is_contiguous() && ids.scalar_type() == torch::kInt32);
    TORCH_CHECK(scores.is_cuda() && scores.is_contiguous() && scores.scalar_type() == torch::kFloat64);
    c10::cuda::CUDAGuard device_guard(xy.device());
    contribution_kernel<<<dim3((width+15)/16, (height+15)/16), dim3(16, 16), 0,
        at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const float2*>(xy.data_ptr<float>()),
        reinterpret_cast<const float4*>(conic.data_ptr<float>()),
        reinterpret_cast<const uint2*>(ranges.data_ptr<int>()),
        reinterpret_cast<const uint32_t*>(ids.data_ptr<int>()),
        scores.data_ptr<double>(), width, height);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
""", verbose=True)



def prune_fields(fields, kept):
    """Slice both model fields and Adam moments by the same CPU source IDs."""
    import copy

    count = len(fields[1])
    pruned = [value.index_select(0, kept) if torch.is_tensor(value) and
              value.ndim and value.shape[0] == count else value for value in fields]
    pruned[11] = {"param_groups": copy.deepcopy(fields[11]["param_groups"]), "state": {
        key: {name: value.index_select(0, kept) if torch.is_tensor(value) and
              value.ndim and value.shape[0] == count else value
              for name, value in entry.items()}
        for key, entry in fields[11]["state"].items()}}
    return tuple(pruned)



def prepare_1k(scene="sear_steak", finetune_steps=1500):
    """Prune to 20%, accumulate training-view contributions, and build keyframe masks.

    Volume weighting uses descending index floor(.9*N), power .1, and no clamp.
    Temporal and spatial scores are summed over time before multiplication."""
    import hashlib

    if not isinstance(finetune_steps, int) or finetune_steps <= 0:
        raise ValueError("record a positive masked fine-tuning budget")
    source = TRAINING / "4dgs" / f"{scene}-trained.pth"
    if not source.exists():
        source = TRAINING / "4dgs" / f"{scene}.pth"
    fields, iteration = read_checkpoint(scene, source)
    if not isinstance(iteration, int) or iteration <= 0:
        raise ValueError("1K pruning requires a trained original checkpoint")
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    output = TRAINING / "4dgs1k" / f"{scene}.pth"
    if output.exists():
        existing = read_payload(output)
        if len(existing) == 3 and existing[2]["source_sha256"] == digest:
            return output
        raise ValueError("existing 1K checkpoint has a different source")
    native, scoring = load_native(), load_scoring()
    parameters, degree, degree_t, _ = load_checkpoint(scene, source)
    poses, width, height, focal = cameras(scene)
    count = len(parameters["means3D"])
    cache = CACHE / "training" / f"{scene}-1k-score.pth"
    cache.parent.mkdir(parents=True, exist_ok=True)
    first, spatial = 0, torch.zeros(count, device="cuda", dtype=torch.float64)
    if cache.exists():
        previous = read_payload(cache)
        if previous["sha256"] == digest:
            first, spatial = previous["next_frame"], previous["spatial"].cuda()
    with torch.inference_mode():
        for frame in range(first, 300):
            for camera in range(1, len(poses)):
                result, xy, conic, _, ranges, ids = native_buffers(native, parameters,
                    settings(native, poses[camera], width, height, focal, frame, degree, degree_t))
                scoring.accumulate_scores(xy, conic, ranges, ids, spatial, width, height)
                del result, xy, conic, ranges, ids
            if (frame+1) % 20 == 0:
                temporary = cache.with_suffix(".pth.part")
                torch.save({"sha256": digest, "next_frame": frame+1,
                            "spatial": spatial.cpu()}, temporary)
                temporary.replace(cache)
                print("1K training-view scoring", frame+1, "/ 300", flush=True)
        a, b, c, d = parameters["rotations"].double().unbind(-1)
        p, q, r, s = parameters["rotations_r"].double().unbind(-1)
        row = torch.stack((a*s-b*r+c*q-d*p, a*r+b*s-c*p-d*q,
                           a*q-b*p-c*s+d*r, a*p+b*q+c*r+d*s), -1)
        scale4 = torch.cat((parameters["scales"], parameters["scales_t"]), -1).double()
        variance = (row*scale4).square().sum(-1)
        temporal = torch.zeros_like(variance)
        for frame in range(300):
            distance2 = (frame/30-parameters["ts"][:, 0].double()).square()
            marginal = torch.exp(-.5*distance2/variance)
            second = (distance2/variance.square()-1/variance)*marginal
            temporal += 1/(.5*second.abs().tanh()+.5)
        volume = scale4.prod(-1)
        reference = volume.sort(descending=True).values[int(count*.9)]
        score = spatial*temporal*(volume/reference).pow(.1)
        if not bool(torch.isfinite(score).all()):
            raise ValueError("non-finite 1K pruning score")
        keep = score.argsort(descending=True, stable=True)[:count//5].sort().values
        kept = keep.cpu()
        pruned = list(prune_fields(fields, kept))
        # 1K uses a constant background. Contribution scoring depends on alpha*T
        # and is independent of the environment-map RGB values.
        pruned[17] = None
        parameters = {name: value.index_select(0, keep) for name, value in parameters.items()}
        keyframes = torch.arange(0, 301, 20, dtype=torch.int32)
        masks = torch.zeros((len(keyframes), len(kept)), device="cuda", dtype=torch.bool)
        for index, frame in enumerate(keyframes.tolist()):
            contribution = torch.zeros(len(kept), device="cuda", dtype=torch.float64)
            for camera in range(1, len(poses)):
                result, xy, conic, _, ranges, ids = native_buffers(native, parameters,
                    settings(native, poses[camera], width, height, focal, frame, degree, degree_t))
                scoring.accumulate_scores(xy, conic, ranges, ids, contribution, width, height)
                del result, xy, conic, ranges, ids
            masks[index] = contribution > 0
            print("1K keyframe", frame, "active", int(masks[index].sum()), flush=True)
        metadata = {"source_sha256": digest, "source_iteration": iteration,
                    "source_file": source.name, "finetune_steps": finetune_steps,
                    "kept_ids": kept, "keyframes": keyframes, "masks": masks.cpu(),
                    "source_environment_removed": fields[17] is not None,
                    "volume_power": .1, "volume_clamp": False,
                    "temporal_aggregation": "eq6_time_sum_times_eq7_spatial_time_sum"}
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".pth.part")
        torch.save((tuple(pruned), iteration, metadata), temporary)
        temporary.replace(output)
        print("1K pruned", count, "to", len(kept), "saved", output, flush=True)
    return output



def continue_training(scene, until=None, one_k=False):
    """Resume original 4DGS training on training cameras only, with batch four."""
    from concurrent.futures import ThreadPoolExecutor
    import time

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    checkpoint = TRAINING / "4dgs1k" / f"{scene}.pth" if one_k else (
        TRAINING / "4dgs" / f"{scene}-trained.pth")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    metadata = None
    if one_k:
        payload = read_payload(checkpoint)
        metadata = payload[2]
        source_iteration = metadata["source_iteration"]
        final_iteration = source_iteration + metadata["finetune_steps"]
        if (not isinstance(metadata["finetune_steps"], int) or metadata["finetune_steps"] <= 0 or
                not source_iteration <= payload[1] <= final_iteration):
            raise ValueError("1K fine-tuning requires a pruned source and a recorded update budget")
        if until is None:
            until = final_iteration
        if until != final_iteration:
            raise ValueError("1K fine-tuning must finish at the recorded update budget")
        if payload[0][17] is not None:
            raise ValueError("1K fine-tuning requires environment maps to be removed")
        if payload[1] == final_iteration:
            print("1K already fine-tuned", checkpoint, "step", final_iteration, flush=True)
            return checkpoint
        masks = metadata["masks"].cuda()
        del payload
    elif until is None:
        until = 30000
    source_fields, _ = read_checkpoint(scene, checkpoint if checkpoint.exists() else None)
    # Large original checkpoints need their Adam moments off device. Pruned 1K
    # models fit normally; this changes storage only, not the update formula.
    offload = not one_k and len(source_fields[1]) > 3_300_000
    del source_fields
    model, options, ssim, first = trainable_model(
        scene, checkpoint if checkpoint.exists() else None, offload=offload)
    if one_k:
        options.iterations = final_iteration
        options.densify_until_iter = 0
        ssim = training_ssim
        metadata.setdefault("fused_ssim_from_iteration", first + 1)
        metadata["fused_ssim_commit"] = FUSED_SSIM_COMMIT
    native = load_native()
    poses, width, height, focal = cameras(scene)
    centers = poses[1:, :3, 3]
    extent = float(np.linalg.norm(centers - centers.mean(0), axis=-1).max() * 1.1)
    print("resume", first, "until", until, "original training extent", extent,
          "checkpoint extent", model.spatial_lr_scale, flush=True)
    if abs(extent - model.spatial_lr_scale) > .001:
        raise ValueError("checkpoint and reconstructed training-camera extent differ")
    backgrounds = None
    if model.env_map is not None:
        import yaml

        pipeline = yaml.safe_load((UPSTREAM / "configs/dynerf" / f"{scene}.yaml").read_text())["PipelineParams"]
        if first < pipeline["env_optimize_until"]:
            raise ValueError("resuming a still-trainable environment map requires its Adam state")
        # Preserve the frozen environment map throughout checkpoint continuation.
        backgrounds = {camera: environment_background(model.env_map, poses[camera],
            width, height, focal, device=model._xyz.device).cpu()
            for camera in range(1, len(poses))}
    print("CPU Adam moments", offload, "frozen environment", backgrounds is not None, flush=True)
    image_root = ROOT / "datasets/n3v" / scene / "images"
    camera_names = sorted(path.stem for path in image_root.parent.glob("cam[0-9][0-9].mp4"))
    if len(camera_names) != len(poses) or camera_names[0] != "cam00":
        raise ValueError("training images must follow the sorted native N3V camera IDs")
    samples = [(camera, frame) for camera in range(1, len(poses)) for frame in range(300)]
    if any(not (image_root / f"{camera_names[camera]}_{frame:04d}.png").exists()
           for camera, frame in samples):
        raise FileNotFoundError("native training image cache is incomplete")
    camera_settings = [settings(native, pose, width, height, focal, 0,
                                model.active_sh_degree, model.active_sh_degree_t) for pose in poses]

    def read_image(sample):
        camera, frame = samples[int(sample)]
        with Image.open(image_root / f"{camera_names[camera]}_{frame:04d}.png") as image:
            return np.array(image)

    permutation = rng.permutation(len(samples))
    cursor = 0
    if one_k:
        state = metadata.get("training_state")
        if state is not None:
            permutation = state["permutation"].numpy()
            cursor = state["cursor"]
            rng.bit_generator.state = state["numpy_rng"]
        else:
            # Recover absent sampler state from the fixed seed and consumed
            # batch count, including epoch boundaries.
            for _ in range(first - metadata["source_iteration"]):
                if cursor + 4 > len(samples):
                    permutation, cursor = rng.permutation(len(samples)), 0
                cursor += 4
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as loader:
        prefetched = None
        for iteration in range(first + 1, until + 1):
            model.update_learning_rate(iteration)
            if iteration % options.sh_increase_interval == 0:
                model.oneupSHdegree()
            if cursor + 4 > len(samples):
                permutation = rng.permutation(len(samples))
                cursor = 0
            batch = permutation[cursor:cursor + 4]
            cursor += 4
            images = prefetched or [loader.submit(read_image, sample) for sample in batch]
            prefetched = ([loader.submit(read_image, sample) for sample in permutation[cursor:cursor+4]]
                          if iteration < until and cursor+4 <= len(samples) else None)
            count = len(model._xyz)
            if not offload and not one_k and count > 3_300_000:
                for entry in model.optimizer.state.values():
                    for key in ("exp_avg", "exp_avg_sq"):
                        entry[key] = entry[key].cpu()
                torch.cuda.empty_cache()
                offload = True
                print("moved Adam moments to CPU at step", iteration, "Gaussians", count, flush=True)
            if not one_k:
                visibility_count = torch.zeros(count, device="cuda", dtype=torch.float32)
                gradient_sum = torch.zeros_like(visibility_count)
                maximum_radius = torch.zeros_like(model.max_radii2D)
            batch_loss = batch_psnr = 0.0
            for sample, future in zip(batch, images):
                pixels = future.result()
                camera, frame = samples[int(sample)]
                target = torch.tensor(pixels, device="cuda", dtype=torch.float32).permute(2, 0, 1) / 255
                ids = ((masks[frame//20] | masks[frame//20+1]).nonzero().flatten()
                       if one_k else temporal_candidates(model, frame / 30))
                parameters = train_parameters(model, ids)
                raster_settings = camera_settings[camera]._replace(timestamp=frame/30,
                    sh_degree=model.active_sh_degree, sh_degree_t=model.active_sh_degree_t)
                result = native.GaussianRasterizer(raster_settings)(**parameters)
                image, radii = result[:2]
                if backgrounds is not None:
                    image = image+(1-result[3])*backgrounds[camera].to(image.device)
                loss = (.8 * (image - target).abs().mean() + .2 * (1 - ssim(image, target))) / 4
                loss.backward()
                with torch.no_grad():
                    if not one_k:
                        visible = radii > 0
                        valid_ids = ids[visible]
                        visibility_count[valid_ids] += 1
                        gradient_sum[valid_ids] += parameters["means2D"].grad[visible, :2].norm(dim=-1)
                        maximum_radius[valid_ids] = torch.maximum(maximum_radius[valid_ids], radii[visible])
                    batch_loss += loss.detach()
                    batch_psnr += -10 * (image.detach().clamp(0, 1) - target).square().mean().log10() / 4
                del image, radii, parameters, loss, target, ids, result
            with torch.no_grad():
                if iteration < options.densify_until_iter:
                    visible = visibility_count > 0
                    model.max_radii2D[visible] = torch.maximum(model.max_radii2D[visible], maximum_radius[visible])
                    correction = 4 / visibility_count.clamp_min(1)
                    model.add_densification_stats_grad(
                        (gradient_sum * correction)[:, None], visible,
                        model._t.grad.detach() * correction[:, None])
                    del correction
                    # Densification replaces large parameter/Adam tensors. Release
                    # the completed batch and allocator cache before concatenation.
                    del visibility_count, gradient_sum, maximum_radius, visible, valid_ids
                    if iteration > options.densify_from_iter and iteration % options.densification_interval == 0:
                        # Densification replaces Parameters; release their
                        # completed gradients before allocating replacements.
                        model.optimizer.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()
                        model.densify_and_prune(
                            options.densify_grad_threshold, options.thresh_opa_prune,
                            extent, 20 if iteration > options.opacity_reset_interval else None,
                            options.densify_grad_t_threshold)
                    if iteration % options.opacity_reset_interval == 0:
                        model.reset_opacity()
                if one_k or iteration < options.iterations:
                    if offload:
                        offloaded_adam_step(model.optimizer)
                    else:
                        model.optimizer.step()
                model.optimizer.zero_grad(set_to_none=True)
                if iteration % 10 == 0:
                    print("step", iteration, "loss", round(float(batch_loss), 6),
                          "training PSNR", round(float(batch_psnr), 4), "Gaussians", len(model._xyz),
                          "seconds/step", round((time.monotonic() - started) / (iteration - first), 3),
                          "peak MiB", round(torch.cuda.max_memory_allocated() / 1024**2), flush=True)
                if iteration % 500 == 0 or iteration == until:
                    temporary = checkpoint.with_suffix(".pth.part")
                    if one_k:
                        metadata["training_state"] = dict(cursor=cursor,
                            permutation=torch.from_numpy(permutation.copy()),
                            numpy_rng=rng.bit_generator.state)
                    payload = (model.capture(), iteration)
                    torch.save((*payload, metadata) if one_k else payload, temporary)
                    temporary.replace(checkpoint)
                    # A saved capture otherwise retains pre-densification
                    # Parameters until the next checkpoint, wasting GPU memory.
                    del payload
                    print("saved", checkpoint, "step", iteration, flush=True)
    return checkpoint



def export_checkpoint(source, destination):
    """Export trained 1K fields and masks, omitting optimizer and sampler state."""
    from pathlib import Path
    from gspim.prepare import sha256

    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("training and inference checkpoint paths must differ")
    fields, iteration, metadata = read_payload(source)
    if iteration != metadata["source_iteration"] + metadata["finetune_steps"]:
        raise ValueError("finish the recorded fine-tuning budget before export")
    fields = list(fields)
    for index in (7, 8, 9, 10, 11):
        fields[index] = None
    fields = tuple(value.detach().cpu().clone() if torch.is_tensor(value) else value for value in fields)
    metadata = {key: value for key, value in metadata.items() if key != "training_state"}
    metadata.update(inference_only=True, training_checkpoint_sha256=sha256(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    torch.save((fields, iteration, metadata), temporary)
    temporary.replace(destination)
    return destination
