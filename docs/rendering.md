# Rendering Guide

This guide covers rendering with GSPIM's batch-major multi-frame pipeline
and the standalone benchmarking tool.

---

## Basic rendering

```bash
conda activate 4dgs

python render_gspim.py \
    --model_path <path/to/trained_model> \
    --window_size 5 \
    --time_threshold 0.05
```

Rendered frames are saved to `<model_path>/renders/`.

---

## Rendering arguments (`render_gspim.py`)

| Argument                  | Default    | Description                                              |
|---------------------------|------------|----------------------------------------------------------|
| `--model_path`            | required   | Path to the trained model directory                      |
| `--iteration`             | `-1`       | Checkpoint iteration to load (`-1` = best/final)         |
| `--window_size`           | `5`        | Temporal window width W (frames per batch)               |
| `--time_threshold`        | `0.05`     | PPIM temporal contribution threshold τ                   |
| `--l2_cache_size`         | `40`       | GPU L2 cache budget in MB (controls batch sizes)         |
| `--enable_pim_sim`        | `False`    | Enable Samsung PIM cycle-accurate timing reports         |
| `--skip_train`            | `False`    | Skip rendering the training split                        |
| `--skip_test`             | `False`    | Skip rendering the test split                            |
| `--output_path`           | `None`     | Override output directory                                |
| `--fps`                   | `30`       | Frame rate for any exported video                        |
| `--compute_cov3D_python`  | `False`    | Compute covariance in Python (slower, for debugging)     |
| `--convert_SHs_python`    | `False`    | Convert SH coefficients in Python                        |
| `--debug`                 | `False`    | Enable CUDA debug assertions                             |

---

## Benchmarking (`benchmark_gspim.py`)

The benchmark script measures FPS, quality metrics, and per-stage timing.

```bash
python benchmark_gspim.py \
    --model_path <path/to/trained_model> \
    --num_frames 50 \
    --window_size 5 \
    --time_threshold 0.05
```

### Benchmark arguments

| Argument            | Default | Description                                                     |
|---------------------|---------|------------------------------------------------------------------|
| `--model_path`      | required | Path to the trained model directory                            |
| `--source_path`     | None    | Scene data path (if different from model_path)                   |
| `--num_frames`      | `50`    | Number of frames to benchmark                                    |
| `--window_size`     | `5`     | GSPIM temporal window width W                                    |
| `--l2_cache_size`   | `50`    | L2 cache budget in MB                                           |
| `--time_threshold`  | `0.05`  | Temporal threshold τ                                             |
| `--output`          | None    | Path to save JSON result file                                    |
| `--skip_baseline`   | `False` | Skip baseline (per-frame) benchmark                              |
| `--quality`         | `False` | Compute PSNR / SSIM metrics                                      |
| `--save_video`      | `False` | Save rendered video comparison                                   |
| `--no_cuda_kernel`  | `False` | Disable JIT CUDA kernels; use PyTorch fallback for Stages 2–4   |

### Sample output

```
================================================================================
GSPIM Pipeline Stage Timing Statistics
================================================================================

Stage                                    Avg (ms)     Total (ms)   Percentage
--------------------------------------------------------------------------------
PIM Stages (Near-Memory Processing):
  Stage 1: PIM Time Filter               2.34         234.1        8.1%
  Stage 1.5: Active Loader               1.87         187.3        6.5%

GPU Stages (Preprocessing + Rendering):
  Stage 2: 4D Projection                 3.11         311.2        10.8%
  Stage 3: Depth Analysis                1.22         122.4        4.2%
  Stage 4: Sorting & Batching            4.88         488.1        16.9%
  Stage 5: Rendering                    16.45        1645.3        57.1%

Summary:
  PIM Total (Stage 1 + 1.5)             4.21         421.4        14.6%
  GPU Total (Stage 2-5)                25.66        2566.9        85.4%
  Overall Total                        29.87        2988.3       100.0%
  Number of Windows                       10

Pipeline Parallelism Analysis:
  Total PIM Time:                   421.4 ms
  Total GPU Time:                  2566.9 ms
  Parallel Total (PIM || GPU):     2620.5 ms    (PIM latency hidden)
  Speedup (parallel/serial):          1.14x
```

---

## Convenience script

`run.sh` wraps the benchmark for common configurations:

```bash
bash run.sh <model_path> [window_size] [threshold]
```

---

## Samsung PIM timing validation

When `--enable_pim_sim` is set, GSPIM calls the Samsung SATIPIMSimulator
backend (`gspim/ppim_backend.py`) after each temporal window to report
cycle-accurate PPIM timing validated against Samsung HBM2 parameters:

```bash
python render_gspim.py \
    --model_path output/flame_salmon \
    --enable_pim_sim
```

Additional output per window:

```
======================================================================
  PPIM Temporal Filter Timing Report  [Samsung SATIPIMSimulator]
======================================================================
  Hardware model:  Samsung HBM2-PIM (8 PIM blocks/ch)
  Channels:        8
  Timing source:   analytic
  Input primitives:       500,000
  Active (pass):          150,000  (30.0% pass rate)
  S1  PIM Filter:         156.29 µs  (156,288 cycles)
  S1.5 Active Load:       212.00 µs  (212,000 cycles)
  Total PPIM time:        368.29 µs
  HBM read (PPIM):        124.3 MB
  HBM saved vs GPU:       201.3 MB  (61.8% reduction)
======================================================================
```

---

## Programmatic API

```python
from gspim import GSPIMRenderer, print_stage_timing_statistics, validate_ppim_bandwidth_claim

# Bandwidth claim validation (standalone)
validate_ppim_bandwidth_claim(num_primitives=500_000)

# Create renderer with Samsung PIM timing enabled
renderer = GSPIMRenderer(
    L2_cache_size       = 40 * 1024 * 1024,
    time_threshold      = 0.05,
    initial_window_size = 5,
    enable_pim_sim      = True,     # enable Samsung PIM backend
    enable_profiling    = True,
)

# Render and get statistics
rendered_images, stats = renderer.render_multiframe_batch(
    gaussians, cameras, pipe, background
)

print_stage_timing_statistics(stats)
```

---

## Runtime async primitives

```python
renderer = GSPIMRenderer(enable_pim_sim=True)

# (A) Fire PPIM filter — non-blocking on real hardware
renderer.gspim_filter_async(t_min=0.0, t_max=0.2, threshold=0.05)

# (B) Overlap: GPU renders previous window concurrently with PIM filter

# (C) Start Active Loader compaction
renderer.gspim_compact_async()

# (D) Memory fence
renderer.gspim_sync()

# (E) GPU rendering using compacted Active Buffer
rendered, stats = renderer.render_multiframe_batch(gaussians, cameras, pipe, bg)
```

| Primitive              | Real HW action                        |
|------------------------|---------------------------------------|
| `gspim_filter_async`   | `PIM_CFG_WRITE; PIM_FILTER`           |
| `gspim_compact_async`  | `PIM_COMPACT`                         |
| `gspim_sync`           | `PIM_FENCE; wait PIM_STATUS.busy==0`  |
