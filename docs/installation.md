# Installation Guide

## System Requirements

| Component        | Minimum              | Recommended            |
|-----------------|----------------------|------------------------|
| OS              | Ubuntu 20.04         | Ubuntu 22.04           |
| GPU             | NVIDIA RTX 3090 (24 GB) | NVIDIA A100 (80 GB) |
| CUDA            | 11.7                 | 12.1                   |
| Driver          | 515+                 | 535+                   |
| RAM             | 32 GB                | 64 GB                  |
| Disk            | 50 GB                | 200 GB (for datasets)  |

> **PIM hardware note.** The GSPIM PPIM layer is simulated in software on
> conventional GPUs.  If you have access to a PPIM-capable device (custom
> HBM controller with bank-level filter logic), set `enable_pim_sim=True`
> and provide the device driver handle to `GSPIMRenderer`.

---

## 1. Clone the repository

```bash
git clone https://github.com/your-org/GSPIM.git
cd GSPIM
```

### Initialise submodules

```bash
git submodule update --init --recursive
```

This fetches:

- `submodules/diff-gaussian-rasterization` — differentiable rasterizer
- `submodules/simple-knn` — fast K-nearest-neighbour initialisation
- `submodules/vggt` — Visual Geometry Grounded Transformer (optional, for point-cloud bootstrap)
- `submodules/samsung-pim` — Samsung SATIPIMSimulator for cycle-accurate PPIM timing validation

> **Samsung PIM note.** The simulator requires a Samsung research license
> (non-commercial/academic use only).  If not available, `gspim/ppim_backend.py`
> falls back to the analytic timing mode automatically — see
> [`ppim_hardware.md`](ppim_hardware.md) for details.

---

## 2. Create the Conda environment

```bash
conda env create -f environment.yml
conda activate 4dgs
```

The environment is named **`4dgs`** and installs PyTorch 2.0, torchvision,
CUDA toolkit headers, and all Python dependencies.

---

## 3. Build CUDA extensions

### 3a. Differentiable Gaussian rasterizer

```bash
cd submodules/diff-gaussian-rasterization
pip install -e .
cd ../..
```

### 3b. Simple-KNN

```bash
cd submodules/simple-knn
pip install -e .
cd ../..
```

### 3c. PointOps2 (optional — used for SH coefficient compression)

```bash
cd pointops2
pip install -e .
cd ..
```

### 3d. Samsung SATIPIMSimulator (optional — for simulation mode)

```bash
# Requires Samsung research license; skip if unavailable
cd submodules/samsung-pim
sudo apt install scons libgtest-dev
scons
cd ../..
```

If the binary is absent, `gspim/ppim_backend.py` automatically uses the
analytic timing model (no build required).

### 3e. GSPIM preprocess kernel (optional JIT kernel)

The `cuda_extensions/` directory contains JIT-compiled CUDA kernels for
Stages 2–4.  They are compiled automatically on first use.  To pre-compile:

```bash
cd cuda_extensions
python setup_gspim_preprocess.py build_ext --inplace
cd ..
```

If compilation fails, `gspim_cuda` will fall back to a PyTorch
implementation transparently.

---

## 4. Verify the installation

```bash
conda activate 4dgs
python -c "
from gspim import PIMTimeFilter, GSPIMDataflow, GSPIMRenderer
print('PIMTimeFilter : OK')
print('GSPIMDataflow : OK')
print('GSPIMRenderer : OK')
"
```

Expected output:

```
PIMTimeFilter : OK
GSPIMDataflow : OK
GSPIMRenderer : OK
```

---

## 5. Directory layout after setup

```
GSPIM/
├── train.py                   # Training entry point
├── render_gspim.py            # Rendering entry point
├── benchmark_gspim.py         # Benchmarking entry point
├── run.sh                     # Convenience benchmark script
├── environment.yml            # Conda environment specification
│
├── gspim/                     # GSPIM core package
│   ├── __init__.py
│   ├── pim_filter.py          # PPIM temporal filter simulation
│   ├── dataflow.py            # Multi-frame scheduling dataflow
│   ├── renderer.py            # Batch-major renderer + async primitives
│   ├── profiler.py            # Performance profiler & bandwidth analyser
│   └── ablation.py            # Ablation study estimator (G1–G5)
│
├── gspim_cuda/                # CUDA kernel loader shim
├── cuda_extensions/           # JIT CUDA kernels (Stages 2–4)
│
├── gaussian_renderer/         # Base 4DGS rasterizer interface
├── scene/                     # Scene loading & Gaussian model
├── arguments/                 # CLI argument groups
├── utils/                     # Image, loss, general utilities
│
├── submodules/                # Git submodules
│   ├── diff-gaussian-rasterization/
│   ├── simple-knn/
│   └── vggt/
│
├── pointops2/                 # PointOps2 (local install)
├── lpipsPyTorch/              # LPIPS perceptual metric
├── configs/                   # Configuration presets
├── scripts/                   # Data conversion utilities
└── docs/                      # This documentation
```

---

## Troubleshooting

### `ImportError: No module named 'diff_gaussian_rasterization'`

Re-run the rasterizer build:

```bash
cd submodules/diff-gaussian-rasterization && pip install -e . && cd ../..
```

### CUDA version mismatch

Make sure the CUDA toolkit in `environment.yml` matches the system driver:

```bash
nvcc --version       # toolkit version
nvidia-smi           # driver/CUDA version
```

### Out-of-memory during training

Reduce the number of Gaussians or lower the spherical-harmonics degree:

```bash
python train.py ... --sh_degree 1 --densification_interval 200
```
