# Training Guide

GSPIM uses the same training pipeline as 4D Gaussian Splatting (4DGS).
Training optimises the 4D Gaussian primitives from a set of multi-view
video frames; the GSPIM rendering layer is applied at **inference** time.

---

## Quick start

```bash
conda activate 4dgs

python train.py \
    -s <path/to/scene> \
    -m <path/to/output_model> \
    --eval
```

---

## Data preparation

See [`datasets.md`](datasets.md) for scene-specific instructions.

The scene directory must contain one of:

- **COLMAP format** (`sparse/0/cameras.bin`, `images/`, etc.)
- **NeRF-synthetic format** (`transforms_train.json`, `transforms_test.json`)

---

## Key training arguments

### Model parameters (`ModelParams`)

| Argument             | Default | Description                                              |
|----------------------|---------|----------------------------------------------------------|
| `-s / --source_path` | —       | Path to the scene directory                              |
| `-m / --model_path`  | —       | Output directory for checkpoints and logs                |
| `--sh_degree`        | `3`     | Max SH degree for view-dependent colour (0 = diffuse)   |
| `--images`           | `images`| Sub-folder name containing image frames                  |
| `--resolution`       | `-1`    | Downscale factor (`-1` = auto from image width)          |

### Optimisation parameters (`OptimizationParams`)

| Argument                    | Default      | Description                                   |
|-----------------------------|--------------|-----------------------------------------------|
| `--iterations`              | `30000`      | Total training iterations                     |
| `--position_lr_init`        | `1.6e-4`     | Initial position learning rate                |
| `--position_lr_final`       | `1.6e-6`     | Final position learning rate (after decay)    |
| `--densify_until_iter`      | `15000`      | Stop densification after this iteration       |
| `--densification_interval`  | `100`        | Densify every N iterations                    |
| `--opacity_reset_interval`  | `3000`       | Reset opacities every N iterations            |
| `--lambda_dssim`            | `0.2`        | Weight for SSIM loss term                     |

### Pipeline parameters (`PipelineParams`)

| Argument             | Default | Description                                              |
|----------------------|---------|----------------------------------------------------------|
| `--convert_SHs_python` | `False` | Compute SH in Python (slower but debuggable)           |
| `--compute_cov3D_python` | `False` | Compute 3D covariance in Python                       |
| `--debug`            | `False` | Enable CUDA debug asserts                                |

---

## Example: Neural 3D Video (N3V) scene

```bash
python train.py \
    -s data/N3V/flame_salmon \
    -m output/flame_salmon \
    --sh_degree 3 \
    --iterations 30000 \
    --eval
```

Training logs are saved to `output/flame_salmon/`:

```
output/flame_salmon/
├── cameras.json          # camera calibration
├── cfg_args              # argument dump
├── input.ply             # initial point cloud
├── point_cloud/          # checkpoints (every 7000 iters)
│   ├── iteration_7000/point_cloud.ply
│   ├── iteration_14000/point_cloud.ply
│   └── iteration_30000/point_cloud.ply
└── train_log.txt         # loss curve
```

---

## Training for GSPIM inference

No special training flags are required to later use GSPIM rendering.
GSPIM inference is compatible with any trained 4DGS model as long as
`gaussian_dim=4` and `rot_4d=True` (both are the defaults).

After training is complete, proceed to [`rendering.md`](rendering.md).

---

## Monitoring

Training outputs PSNR and loss every 100 iterations.  Use TensorBoard to
visualise the full loss curves:

```bash
tensorboard --logdir output/flame_salmon
```

---

## Tips for large scenes

- Start with `--sh_degree 1` to reduce memory usage; increase to `3` after
  densification is stable.
- Use `--resolution 2` to halve image resolution during early training.
- The default `--densify_until_iter 15000` works well; for very large scenes
  you may extend to `20000`.
- If GPU memory is tight, lower `--densification_interval` to 200 to reduce
  the peak Gaussian count.
