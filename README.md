# GSPIM: A 4DGS Rendering Architecture with Primitive-Aware PIM Temporal Filtering and Multi-Frame Dataflow

> **GSPIM: A 4DGS Rendering Architecture with Primitive-Aware PIM Temporal Filtering and Multi-Frame Dataflow**  
> Anonymous Author(s)  
> *Under review*

## Overview

GSPIM is a three-layer architecture–runtime co-design for 4D Gaussian Splatting (4DGS) inference on heterogeneous GPU–PIM platforms. It addresses two systematic HBM bandwidth inefficiencies in native 4DGS:

1. **Temporal filtering waste**: Over 70% of primitives are discarded after loading, wasting 75%+ of HBM-to-L2 traffic.
2. **Frame-isolated rendering**: High cross-frame Gaussian set overlap is not exploited, causing repeated primitive eviction and cache thrashing.

### Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Programming Interface Layer                            │
│  gspim_filter_async | gspim_compact_async | gspim_sync  │
├─────────────────────────────────────────────────────────┤
│  Runtime System Layer                                   │
│  Batch-major multi-frame scheduler + window adaptation  │
│  Depth stability score + differentiated sorting         │
├─────────────────────────────────────────────────────────┤
│  Memory Architecture Layer (PPIM)                       │
│  Near-memory 5-stage temporal contribution filtering    │
│  Active Map → Active Loader → Active Buffer compaction  │
└─────────────────────────────────────────────────────────┘
```

### Five-Stage GSPIM Dataflow

| Stage | Location | Description |
|-------|----------|-------------|
| S1: PPIM Filtering | HBM/PIM | Each bank evaluates p_max(t) in parallel; Active Map → Active Buffer compaction |
| S2: Multi-Frame 4D Projection | GPU | Single data-loading pass projects active primitives to 3D for all frames in window |
| S3: Depth Stability Classification | GPU | Depth stability score SS = exp(−Σ|d_{i+1}−d_i|) classifies stable/unstable sets |
| S4: Sorting & Batch Scheduling | GPU | Stable primitives sorted once globally; unstable sorted per-frame; merged via parallel algorithm |
| S5: Batch-Major Multi-Frame Rendering | GPU | GCC-based gaussian-wise rasterization; shared tile structure across frames in batch |

### Key Results

- **2.36× average FPS speedup** on N3V dataset (small-motion scenes)
- **3.75× average FPS speedup** on EasyVolcap dataset (large-motion scenes)
- **Negligible quality loss**: PSNR/SSIM changes < 10⁻³
- **Lightweight hardware**: Only 1.91% area and 0.89% power added to 6-stack HBM3

---

## Repository Structure

```
GSPIM/
├── train.py                    # 4DGS training (4DGS-1K configuration)
├── render_gspim.py             # Render trained model with GSPIM dataflow
├── benchmark_gspim.py          # End-to-end benchmark vs baseline
├── gspim_renderer.py           # GSPIMRenderer: batch-major multi-frame renderer
├── gspim_dataflow.py           # GSPIMDataflow: 5-stage dataflow implementation
├── gspim_profiler.py           # GSPIMProfiler: bandwidth and timing analysis
├── pim_time_filter.py          # PIMTimeFilter: PPIM temporal filter simulation
├── ablation_estimator.py       # Ablation study FPS estimator
├── run.sh                      # Quick benchmark runner
├── arguments/                  # Argument dataclasses
├── scene/                      # Scene, cameras, Gaussian model (4D)
├── gaussian_renderer/          # Differentiable rasterization wrapper
├── utils/                      # Loss, image, camera, SH utilities
├── cuda_extensions/            # JIT-compiled CUDA kernels
│   ├── gspim_preprocess.cu     # Stage 2–3: 4D→3D projection + depth stability
│   ├── gspim_ops.cu            # Time contribution kernel, compaction, depth entropy
│   └── merge_sorted.cu         # Thrust-based parallel merge for Stage 4
├── gspim_cuda/                 # Compatibility shim → cuda_extensions
├── pointops2/                  # Point cloud operations (local)
├── lpipsPyTorch/               # LPIPS perceptual metric
├── configs/
│   ├── dynerf/                 # N3V scene configs (coffee_martini, sear_steak, …)
│   └── dnerf/                  # D-NeRF scene configs
├── scripts/
│   └── n3v2blender.py          # N3V → Blender format converter
└── submodules/
    ├── diff-gaussian-rasterization/   # Differentiable Gaussian rasterizer
    ├── simple-knn/                    # GPU KNN for Gaussian initialization
    └── vggt/                          # VGGT: initial point cloud construction
```

---

## Setup

### 1. Clone the Repository

```bash
git clone <repo-url> GSPIM
cd GSPIM
git submodule update --init --recursive
```

### 2. Create the Conda Environment

```bash
conda env create -f environment.yml
conda activate 4dgs
```

### 3. Install Editable Extensions

```bash
pip install -e submodules/diff-gaussian-rasterization
pip install -e submodules/simple-knn
pip install -e pointops2
```

### 4. VGGT for Initial Point Cloud (Recommended)

GSPIM uses [VGGT](https://github.com/facebookresearch/vggt) to construct high-quality initial point clouds before training. Follow VGGT's installation instructions, then:

```bash
# Generate initial point cloud for a scene
python submodules/vggt/demo.py \
    --input data/N3V/<scene>/images \
    --output data/N3V/<scene>/sparse/0/points3D.ply
```

---

## Data Preparation

### Neural 3D Video (N3V)

Download the N3V dataset and convert to the required format:

```bash
python scripts/n3v2blender.py \
    --source data/N3V_raw/<scene> \
    --output data/N3V/<scene>
```

Supported scenes: `coffee_martini`, `cook_spinach`, `cut_roasted_beef`, `flame_salmon`, `flame_steak`, `sear_steak`

### EasyVolcap (ZJU)

Download EasyVolcap scenes and place under `data/EasyVolcap/<scene>`. Large-motion scenes (`dance3`, `actor1_4`) are used to evaluate high-motion performance.

---

## Training

Train a 4DGS-1K model on a scene using the provided config files:

```bash
# N3V scene (e.g., coffee_martini)
python train.py --config configs/dynerf/coffee_martini.yaml

# D-NeRF scene
python train.py --config configs/dnerf/jumpingjacks.yaml
```

Training produces a checkpoint under `output/<dataset>/<scene>/`.

---

## Rendering with GSPIM

Run rendering with the GSPIM multi-frame dataflow and PIM simulation:

```bash
python render_gspim.py \
    --model_path output/N3V/coffee_martini \
    --source_path data/N3V/coffee_martini \
    --window_size 5 \
    --enable_pim_sim \
    --stability_threshold 0.98
```

Key arguments:
| Argument | Default | Description |
|----------|---------|-------------|
| `--window_size` | 5 | Temporal window width W |
| `--enable_pim_sim` | False | Enable PPIM near-memory filter simulation |
| `--stability_threshold` | 0.98 | Depth stability score threshold τ_s for Stage 3 |
| `--batch_size` | Auto | Batch size B (auto-sized to L2 capacity) |
| `--adaptive_window` | True | Enable adaptive window width (G5 optimization) |

---

## Benchmarking

Run the full GSPIM vs. baseline benchmark:

```bash
# Quick run (N3V scene)
bash run.sh coffee_martini

# Full benchmark with all ablation groups G1–G5
python benchmark_gspim.py \
    --model_path output/N3V/coffee_martini \
    --source_path data/N3V/coffee_martini \
    --num_frames 300 \
    --window_size 5 \
    --ablation_all
```

The benchmark reports:
- Per-frame FPS and speedup vs. baseline
- Average Gaussian load count per frame
- HBM bandwidth utilization
- PSNR/SSIM quality metrics

---

## PPIM Temporal Filtering (PIM Simulation)

`pim_time_filter.py` implements the software simulation of the five-stage PPIM near-memory filter:

```python
from pim_time_filter import PIMTimeFilter

filter = PIMTimeFilter(threshold=0.05, enable_pim_sim=True)
active_mask, active_indices, p_t_max = filter.filter_gaussians(
    scales, rotation_l, rotation_r, mu_t, t_min, t_max
)
```

**Five-stage computation** (matching Fig. 7 in the paper):
1. **Temporal offset & scale** (FP16 → 26-bit fixed-point, compute dt)
2. **4D rotation matrix construction** (quaternion base products via 4-multiplier FU)
3. **Scaled rotation factor** (temporal row/column M₃,: and M:,₃)
4. **Temporal covariance** (Σ₃₃ = M₃,: · M:,₃ inner product)
5. **Filter decision** (algebraic test: 0.5·dt² < Σ₃₃·ln(20), avoids exponentiation)

---

## Runtime Async Primitives

The three composable primitives from Section 6.1 of the paper are exposed via `gspim_renderer.py`:

```python
from gspim_renderer import GSPIMRenderer

renderer = GSPIMRenderer(model, window_size=5)

# Pipeline: configure + filter + compact (async), then sync before GPU rendering
renderer.gspim_filter_async(t_min, t_max, threshold=0.05)
renderer.gspim_compact_async(dst_buffer)
renderer.gspim_sync()                        # memory fence; blocks until PIM done
frames = renderer.render_window(cameras)
```

These map to the PPIM ISA instructions: `PIM_CFG_WRITE`, `PIM_FILTER`, `PIM_COMPACT`, `PIM_FENCE`.

---

## CUDA Extensions

The `cuda_extensions/` directory provides JIT-compiled CUDA kernels that accelerate Stages 2–4:

| File | Kernel | Purpose |
|------|--------|---------|
| `gspim_preprocess.cu` | `gspim_preprocess_multiframe` | Stage 2–3: batched 4D→3D projection + depth sequence + stability score |
| `gspim_ops.cu` | `compute_time_contribution_kernel`, `compact_active_gaussians_kernel` | Stage 1 simulation, active set compaction |
| `merge_sorted.cu` | `merge_sorted_sequences` | Stage 4: Thrust-based parallel merge of stable+unstable sorted lists |

Extensions load lazily via JIT; if compilation fails (CUDA version mismatch), PyTorch fallbacks activate automatically.

---

## Hardware Configuration

Evaluated on NVIDIA H100 80GB + 6-stack HBM3 (Table 2 of the paper):

| GPU | PIM-HBM3 |
|-----|----------|
| Core: 1755 MHz, 132 SMs | 6 stacks × 8 dies × 32 banks |
| L2: 50 MB | 1,536 total PIM units @ 300 MHz |
| HBM3 BW: 3.35 TB/s | FU: 4× multipliers, 3× adders, 16-entry regfile |

PPIM functional units implemented in Chisel HDL, synthesized at TSMC 28nm. Performance modeled via extended Samsung PIMSimulator.

---

## License

- GSPIM-specific code: see [LICENSE](LICENSE)
- diff-gaussian-rasterization: see [LICENSE_gaussian_splatting.md](LICENSE_gaussian_splatting.md) (Inria, non-commercial research use)
- Gaussian Splatting base: [LICENSE_gaussian_splatting.md](LICENSE_gaussian_splatting.md)
