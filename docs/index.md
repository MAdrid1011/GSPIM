# GSPIM Documentation

**GSPIM** (Gaussian Splatting Processing-In-Memory) is an open-source
implementation of the three-layer hardware-software co-design described in
the paper:

> *GSPIM: Accelerating 4D Gaussian Splatting Inference with Primitive-Aware
> Processing-In-Memory*

The system targets real-time dynamic novel-view synthesis (4DGS) by
eliminating ineffective HBM-to-GPU data transfers through near-memory
temporal filtering (PPIM) and a batch-major multi-frame rendering dataflow.

---

## Quick navigation

| Guide                               | Description                                              |
|-------------------------------------|----------------------------------------------------------|
| [Installation](installation.md)     | Environment setup, submodule build, verification         |
| [Architecture](architecture.md)     | Three-layer design, PPIM, pipeline stages, data-flow     |
| [PPIM Hardware](ppim_hardware.md)   | Samsung SATIPIMSimulator integration & closed-loop proof |
| [Training](training.md)             | Training 4DGS models for GSPIM inference                 |
| [Rendering](rendering.md)           | GSPIM renderer, benchmark, Samsung PIM timing reports    |
| [Datasets](datasets.md)             | N3V, Plenoptic Video, HyperNeRF, D-NeRF setup           |
| [API Reference](api_reference.md)   | Full class and function documentation                    |

---

## Repository layout at a glance

```
GSPIM/
├── train.py              Training entry point
├── render_gspim.py       GSPIM rendering entry point
├── benchmark_gspim.py    Performance benchmark tool
├── run.sh                Quick benchmark script
│
├── gspim/                Core GSPIM package (import from here)
│   ├── pim_filter.py     PPIM temporal filter simulation
│   ├── ppim_backend.py   Samsung SATIPIMSimulator Python bridge
│   ├── dataflow.py       Multi-frame batch scheduling
│   ├── renderer.py       Batch-major renderer + async primitives
│   ├── profiler.py       Profiler and bandwidth analyser
│   └── ablation.py       Ablation study estimator (G1–G5)
│
├── gspim_cuda/           CUDA kernel loader shim
├── cuda_extensions/      JIT CUDA kernels for Stages 2–4
├── gaussian_renderer/    Base 4DGS rasterizer
├── scene/                Scene loading, GaussianModel
├── utils/                General utilities
├── submodules/           External dependencies (git submodules)
│   ├── samsung-pim/      Samsung SATIPIMSimulator (HBM2-PIM cycle model)
│   ├── diff-gaussian-rasterization/
│   ├── simple-knn/
│   └── vggt/
└── docs/                 This documentation
```

---

## Minimal usage example

```python
from gspim import GSPIMRenderer, print_stage_timing_statistics

renderer = GSPIMRenderer(
    time_threshold      = 0.05,  # PPIM filter threshold τ
    initial_window_size = 5,     # temporal window width W
)

rendered_images, stats = renderer.render_multiframe_batch(
    gaussians, cameras, pipe, background
)

print_stage_timing_statistics(stats)
```

---

## Citation

```bibtex
@article{gspim2025,
  title   = {GSPIM: Accelerating 4D Gaussian Splatting Inference with
             Primitive-Aware Processing-In-Memory},
  author  = {Author(s)},
  journal = {Proceedings of ...},
  year    = {2025},
}
```
