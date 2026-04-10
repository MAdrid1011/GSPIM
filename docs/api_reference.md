# API Reference

All public symbols are accessible from the top-level `gspim` package:

```python
from gspim import (
    # Memory layer
    PIMTimeFilter,
    SamsungPIMBackend,
    PPIMFilterResult,
    get_backend,
    validate_ppim_bandwidth_claim,
    # Runtime layer
    GSPIMDataflow,
    # Interface layer
    GSPIMRenderer,
    print_stage_timing_statistics,
    # Analysis
    GSPIMProfiler,
    BandwidthAnalyzer,
    # Ablation
    AblationEstimator,
    create_ablation_report,
)
```

---

## `gspim.pim_filter` — PPIM Temporal Filter

### `PIMTimeFilter`

GPU simulation of the bank-parallel near-memory temporal filter.
Integrates with `SamsungPIMBackend` for cycle-accurate timing validation.

```python
class PIMTimeFilter:
    def __init__(
        self,
        threshold:      float = 0.05,    # τ: contribution threshold
        enable_pim_sim: bool  = False,   # enable Samsung PIM timing
        hbm_channels:   int   = 8,       # number of HBM2 channels
        pim_clock_ghz:  float = 1.0,
        hbm_bw_gbps:    float = 900.0,
    ): ...

    def filter_gaussians(
        self,
        scales:     Tensor,   # [N, 4]  four-dimensional scales
        rotation_l: Tensor,   # [N, 4]  left quaternion (w,x,y,z)
        rotation_r: Tensor,   # [N, 4]  right quaternion
        mu_t:       Tensor,   # [N, 1]  temporal mean μ_t
        t_min:      float,    # window start timestamp
        t_max:      float,    # window end timestamp
    ) -> tuple[BoolTensor, LongTensor, FloatTensor]:
        """
        Returns
        -------
        active_mask    : BoolTensor [N]   True = primitive passes filter
        active_indices : LongTensor [M]   indices of active primitives
        p_t_max        : FloatTensor [N]  float mask (1.0 or 0.0)
        """

    def get_statistics(self) -> dict: ...
    def reset_statistics(self) -> None: ...
    def validate_bandwidth_claim(self, num_primitives: int = 500_000) -> None:
        """
        Print a Samsung PIM validation report for the PPIM bandwidth claim.
        Shows ≥60% bandwidth reduction vs. full GPU read.
        """
```

---

## `gspim.ppim_backend` — Samsung SATIPIMSimulator Bridge

### `SamsungPIMBackend`

Cycle-accurate PPIM timing using Samsung HBM2 parameters from
`submodules/samsung-pim/ini/HBM2_samsung_2M_16B_x64.ini`.

```python
class SamsungPIMBackend:
    def __init__(
        self,
        num_channels:   int   = 8,
        pim_clock_ghz:  float = 1.0,
        hbm_bw_gbps:    float = 900.0,
        ini_path:       Path | None = None,
        try_simulation: bool = False,   # True = call ./sim binary if available
    ): ...

    def compute_filter_timing(
        self,
        num_primitives: int,
        num_active:     int,
        threshold:      float = 0.05,
    ) -> PPIMFilterResult: ...

    def print_timing_report(self, result: PPIMFilterResult) -> None: ...

    def validate_pim_filter(
        self,
        num_primitives: int,
        active_ratio:   float = 0.30,
        threshold:      float = 0.05,
    ) -> dict:
        """Returns {'validated': bool, 'savings_pct': float, 'result': PPIMFilterResult}"""
```

### `PPIMFilterResult`

```python
@dataclass
class PPIMFilterResult:
    num_primitives:      int
    num_active:          int
    filter_cycles:       int      # Samsung PIM cycles for Stage 1
    compaction_cycles:   int      # cycles for Stage 1.5 Active Loader
    filter_time_us:      float    # Stage 1 wall-clock time (µs)
    compaction_time_us:  float    # Stage 1.5 wall-clock time (µs)
    total_time_us:       float    # total PPIM time (µs)
    bandwidth_read_gb:   float    # HBM bytes read by PPIM (GB)
    bandwidth_saved_gb:  float    # bytes saved vs. baseline GPU read (GB)
    savings_pct:         float    # bandwidth reduction percentage
    timing_source:       str      # "analytic" | "simulation"
```

### Convenience functions

```python
def get_backend(
    num_channels:  int   = 8,
    pim_clock_ghz: float = 1.0,
    hbm_bw_gbps:   float = 900.0,
) -> SamsungPIMBackend: ...

def validate_ppim_bandwidth_claim(num_primitives: int = 500_000) -> None:
    """Quick self-test: verify ≥60% bandwidth reduction for given N."""
```

---

## `gspim.dataflow` — Multi-Frame Scheduling

### Key classes

```python
class ActiveLoader:
    """PIM-side compaction: scatter-gather active attributes to Active Buffer."""
    def compact_active_gaussians(
        self, gaussians, active_indices: LongTensor
    ) -> dict | None:
        """Returns compact tensor dict or None if no active primitives."""

class DepthEntropyAnalyzer:
    """Stage 3: compute Depth Stability Score (DSS) per window."""

class BatchScheduler:
    """Stage 4: assign primitives to L2-cache-fitting batches."""

class WindowAdaptiveController:
    """Adjust window width W based on DSS feedback."""

class GSPIMDataflow:
    def __init__(
        self,
        L2_cache_size:       int   = 40 * 1024 * 1024,
        time_threshold:      float = 0.05,
        initial_window_size: int   = 5,
        enable_pim_sim:      bool  = False,
        use_cuda_kernel:     bool  = True,
    ): ...

    def process_time_window(
        self,
        gaussians,                     # GaussianModel
        timestamps: list[float],       # W timestamps in the window
        viewpoint_camera,              # reference camera for DSS
    ) -> tuple[list[dict], LongTensor, dict]:
        """
        Run Stages 1–4 for one temporal window.

        Returns
        -------
        batches : list of batch dicts, each containing:
            'union_indices'       : LongTensor [M_batch]
            'render_order'        : list[LongTensor]  length W
            'is_sequential_order' : bool
        active_indices : LongTensor [N_active]
        window_stats : dict
            'window_size'            : int
            'num_batches'            : int
            'pim_time_ms'            : float
            'gpu_preprocess_time_ms' : float
            'compact_data'           : dict of active tensors
            'stage_times'            : dict of per-stage floats (ms)
        """
```

---

## `gspim.renderer` — Batch-Major Renderer

### `GSPIMRenderer`

```python
class GSPIMRenderer:
    def __init__(
        self,
        L2_cache_size:       int   = 40 * 1024 * 1024,
        time_threshold:      float = 0.05,
        initial_window_size: int   = 5,
        enable_pim_sim:      bool  = False,
        enable_profiling:    bool  = True,
        use_cuda_kernel:     bool  = True,
    ): ...

    # Async runtime primitives
    def gspim_filter_async(self, t_min: float, t_max: float, threshold: float = 0.05) -> None: ...
    def gspim_compact_async(self, dst_buffer=None) -> None: ...
    def gspim_sync(self) -> None: ...

    # Rendering
    def render_multiframe_batch(
        self,
        gaussians,
        viewpoint_cameras: list,
        pipe,
        bg_color: Tensor,
        scaling_modifier: float = 1.0,
    ) -> tuple[list[dict], dict]:
        """
        Returns
        -------
        rendered_images : list of dicts (one per frame)
            Each: {'render': [3,H,W], 'alpha': [1,H,W],
                   'depth': [1,H,W], 'transmittance': [1,H,W]}
        window_stats : dict  (same as GSPIMDataflow.process_time_window)
        """

    def render_video_sequence(
        self,
        gaussians,
        all_cameras: list,
        pipe,
        bg_color: Tensor,
        scaling_modifier: float = 1.0,
    ) -> tuple[list[dict], dict]:
        """Render full sequence by splitting into windows of size W."""
```

### `print_stage_timing_statistics`

```python
def print_stage_timing_statistics(
    statistics: dict,
    baseline_fps: float | None = None,
) -> None:
    """
    Print per-stage timing, parallelism analysis, SOTA comparison,
    and ablation estimation to stdout.
    """
```

---

## `gspim.profiler` — Performance Profiler

### `GSPIMProfiler`

```python
class GSPIMProfiler:
    def __init__(self, enable_cuda_profiler: bool = True): ...

    def start_event(self, name: str) -> object: ...
    def end_event(self, name: str, start_event) -> float: ...   # returns ms
    def record_memory(self, name: str) -> None: ...
    def estimate_bandwidth(self, data_size_mb: float, time_ms: float) -> float: ...

    def get_summary(self) -> dict: ...
    def print_summary(self) -> None: ...
    def get_fps(self, num_frames: int | None = None) -> float: ...
    def save_to_file(self, filepath) -> None: ...
    def compare_with_baseline(self, baseline_file) -> None: ...

    def generate_ablation_study(
        self,
        baseline_fps:          float | None = None,
        k_fps:                 float | None = None,
        full_fps:              float | None = None,
        scene_motion:          str = 'medium',
        auto_estimate_baseline: bool = True,
    ) -> AblationEstimator | None: ...
```

### `BandwidthAnalyzer`

```python
class BandwidthAnalyzer:
    def record_transfer(self, name: str, data_size_bytes: int) -> None: ...
    def estimate_gaussian_transfer(
        self,
        num_gaussians: int,
        with_time_filter: bool = False,
    ) -> int: ...
    def get_bandwidth_savings(self) -> dict:
        """{'gspim_bytes', 'baseline_bytes', 'savings_percent'}"""
```

---

## `gspim.ablation` — Ablation Study Estimator

### `AblationEstimator`

```python
@dataclass
class AblationEstimator:
    baseline_fps : float
    k_fps        : float
    full_fps     : float
    scene_motion : str   # 'low' | 'medium' | 'high'
    results      : dict  # populated by .estimate()

    def estimate(self) -> dict: ...
    def print_report(self) -> None: ...
```

### Ablation groups

| Group | Optimisation                              |
|-------|-------------------------------------------|
| G1    | PPIM near-memory temporal filtering       |
| G2    | Batch-major multi-frame dataflow           |
| G3    | Depth-stability differentiated sorting    |
| G4    | GPU batch-wise Gaussian prefetching       |
| G5    | Adaptive window width (`WindowAdaptiveController`) |

```python
def create_ablation_report(
    baseline_fps: float,
    k_fps:        float,
    full_fps:     float,
    scene_motion: str = 'medium',
) -> AblationEstimator: ...
```

---

## CUDA extensions (`cuda_extensions/`)

Loaded automatically on first import via `torch.utils.cpp_extension.load`:

| Symbol                     | Source file           | Stages covered |
|----------------------------|-----------------------|----------------|
| `gspim_preprocess_multiframe` | `gspim_preprocess.cu` | Stage 2–3      |
| `merge_sorted_sequences`   | `merge_sorted.cu`     | Stage 4        |

Both fall back to PyTorch equivalents if JIT compilation fails.

```python
from cuda_extensions import gspim_preprocess_multiframe, merge_sorted_sequences
```
