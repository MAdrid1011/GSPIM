# GSPIM Architecture

GSPIM is a three-layer hardware-software co-design that accelerates 4DGS
(4D Gaussian Splatting) inference by eliminating ineffective data transfers
between HBM and the GPU compute fabric.

---

## Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   Programming Interface Layer               │
│  GSPIMRenderer · gspim_filter_async · gspim_compact_async   │
│                      gspim_sync                             │
├─────────────────────────────────────────────────────────────┤
│                    Runtime System Layer                     │
│  GSPIMDataflow · WindowController · DifferentiatedSorter    │
│  BatchScheduler · DepthStabilityClassifier                  │
├─────────────────────────────────────────────────────────────┤
│                  Memory Architecture Layer (PPIM)           │
│  PIMTimeFilter · Active Loader · HBM Bank-Parallel Filter   │
└─────────────────────────────────────────────────────────────┘
```

---

## Layer 1 — Memory Architecture (PPIM)

The **Primitive-Aware PIM (PPIM)** layer implements near-memory temporal
filtering inside HBM banks.  Each HBM bank is augmented with a small fixed
logic unit that evaluates the temporal contribution probability
p(t | μ_t, σ_t) for every 4D Gaussian primitive before any data leaves
the memory subsystem.

### Why near-memory filtering?

A 4D Gaussian Splatting scene typically contains **500 k – 2 M primitives**.
For a single rendering timestamp, more than 70 % of primitives have a
temporal contribution below a threshold τ (default: 0.05) and contribute
nothing visible to the image.  Without PPIM, all primitive attributes
(~636 bytes each) still traverse the HBM bus, wasting bandwidth.

### PPIM pipeline

```
Bank 0 │ Filter ──► Active Map[0]
Bank 1 │ Filter ──► Active Map[1]   (all banks parallel)
  …    │  …
Bank k │ Filter ──► Active Map[k]
         └─────────────────────────► Active Buffer (compacted)
                   Active Loader
```

| Component     | Description                                          |
|--------------|------------------------------------------------------|
| Filter logic  | Reads 52 bytes per primitive; evaluates p(t)         |
| Active Map    | 1-bit mask, one entry per primitive, per bank        |
| Active Loader | Scatter-gather: copies active primitives to GPU DRAM |
| Active Buffer | Compact, contiguous representation for GPU use       |

### Simulated in software

In this open-source release, PPIM is simulated on GPU using `PIMTimeFilter`
(`gspim/pim_filter.py`).  The simulation mirrors the 5-stage filter pipeline
and accounts for:

- Bank-parallel filter latency (proportional to N / #banks)
- Active Loader scatter latency (proportional to N_active)
- HBM bandwidth model (configurable peak bandwidth)

---

## Layer 2 — Runtime System

The **Runtime System** implements a 5-stage multi-frame rendering pipeline
that reuses Gaussian data across temporal frames within a window.

### Pipeline stages

| Stage | Name                          | Unit | Description                                          |
|-------|-------------------------------|------|------------------------------------------------------|
| S1    | PIM Time Filter               | PPIM | Bank-parallel p(t) evaluation → Active Map           |
| S1.5  | Active Loader                 | PPIM | Compact Active Buffer for GPU                        |
| S2    | 4D Projection                 | GPU  | Map 4D Gaussians → 3D Gaussians for each timestamp   |
| S3    | Depth Stability Analysis      | GPU  | Compute DSS; classify stable vs. dynamic primitives  |
| S4    | Differentiated Sorting        | GPU  | Stable: reuse sorted order; dynamic: re-sort         |
| S5    | Batch-Major Rendering         | GPU  | Outer loop on batches; inner loop on frames          |

### Adaptive temporal window

The window controller (`WindowController`) adjusts the window width W
dynamically based on the **Depth Stability Score (DSS)**:

```
DSS = (# stable Gaussians) / (# active Gaussians)

DSS ≥ 0.8  →  enlarge W  (frames are similar; batch reuse is high)
DSS < 0.5  →  shrink W   (fast motion; batches diverge across frames)
```

### Differentiated sorting (Stage 4)

Stable primitives keep their sort order from the previous window.
Dynamic primitives are re-sorted by per-frame depth.
This avoids a full O(N log N) sort on every frame.

```
Stable (DSS ≥ threshold)  ──► reuse depth order from window t-1
Dynamic (DSS <  threshold) ──► argsort by |z| for current timestamp
                               merge stable + dynamic → ROL (Render Order List)
```

### Batch-major rendering (Stage 5)

Traditional per-frame rendering loops: for each frame, load all Gaussians.
GSPIM inverts the loop structure:

```
for batch B in batches:                      # B fits in GPU L2 cache
    load union(active Gaussians for B) once  # single HBM transfer
    for frame f in window:                   # reuse data across frames
        render frame f using B's Gaussians
        accumulate (alpha compositing)
```

The union of active Gaussians for batch B is the set of primitives that are
active in at least one frame of the window within that spatial batch.
On average, 60–80 % of Gaussians are reused across frames.

---

## Layer 3 — Programming Interface

The programming interface exposes three **asynchronous runtime primitives**
that map GPU commands to PIM controller registers.

| Primitive              | Semantics                                              |
|------------------------|--------------------------------------------------------|
| `gspim_filter_async`   | Set temporal window bounds + threshold; start filter   |
| `gspim_compact_async`  | Start Active Loader: write Active Buffer to GPU DRAM   |
| `gspim_sync`           | Memory fence: block until PIM operation completes      |

Example usage:

```python
renderer = GSPIMRenderer(enable_pim_sim=True)

# Issue filter command (non-blocking on real hardware)
renderer.gspim_filter_async(t_min=0.0, t_max=0.2, threshold=0.05)

# Overlap GPU work here (e.g. previous window rendering)
# ...

# Wait for PPIM to finish; start data transfer
renderer.gspim_compact_async()
renderer.gspim_sync()

# Now GPU can safely read Active Buffer
rendered, stats = renderer.render_multiframe_batch(
    gaussians, cameras, pipe, bg_color
)
```

---

## Data-flow diagram

```
                     HBM (4D Gaussian attributes)
                            │
              ┌─────────────▼─────────────┐
              │  Stage 1: PPIM Filter     │  52 B/prim read
              │  p(t) > τ → Active Map    │
              └─────────────┬─────────────┘
                            │ Active Map (1 bit/prim)
              ┌─────────────▼─────────────┐
              │  Stage 1.5: Active Loader │  ~636 B × N_active
              │  Compact → Active Buffer  │
              └─────────────┬─────────────┘
                            │ Active Buffer (GPU DRAM)
         ┌──────────────────┴──────────────────┐
         │                                     │
┌────────▼────────┐                   ┌────────▼────────┐
│ Stage 2         │                   │ Stage 2 (f+1)   │
│ 4D Projection   │      W frames     │ 4D Projection   │
└────────┬────────┘                   └────────┬────────┘
         │                                     │
┌────────▼────────┐                   ┌────────▼────────┐
│ Stage 3         │                   │ Stage 3 (f+1)   │
│ Depth Stability │                   │ Depth Stability │
└────────┬────────┘                   └────────┬────────┘
         │                                     │
┌────────▼─────────────────────────────────────▼────────┐
│         Stage 4: Differentiated Sorting & Batching     │
│  Batch 0: union({A,B,C,D})  Batch 1: union({E,F,G})   │
└────────────────────────┬───────────────────────────────┘
                         │
┌────────────────────────▼───────────────────────────────┐
│         Stage 5: Batch-Major Rendering                  │
│  Batch 0: render frames 0…W-1 → accumulate             │
│  Batch 1: render frames 0…W-1 → accumulate             │
│  Final: add background → output images                  │
└────────────────────────────────────────────────────────┘
```

---

## Performance model

End-to-end latency per temporal window:

```
T_window = T_PIM[0]                                  # serial startup
         + Σ_{i=0}^{W-2} max(T_GPU[i], T_PIM[i+1])  # PIM ‖ GPU pipeline
         + T_GPU[W-1]                                # last window
```

Where:
- `T_PIM[i]` = Stage 1 + Stage 1.5 time for window i
- `T_GPU[i]` = Stage 2–5 time for window i

Because T_PIM ≈ 0.3 × T_GPU (typical), the pipeline almost fully hides
PIM latency after the first window.
