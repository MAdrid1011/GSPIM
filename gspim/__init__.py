"""
gspim — GSPIM core package.

Three-layer architecture for 4DGS inference on GPU–PIM platforms:
  Memory layer:      PIMTimeFilter (pim_filter.py)
  Runtime layer:     GSPIMDataflow (dataflow.py)
  Interface layer:   GSPIMRenderer + async primitives (renderer.py)
  Analysis:          GSPIMProfiler, BandwidthAnalyzer (profiler.py)
  Ablation study:    AblationEstimator (ablation.py)
"""

from .pim_filter   import PIMTimeFilter
from .ppim_backend import SamsungPIMBackend, PPIMFilterResult, get_backend, validate_ppim_bandwidth_claim
from .dataflow     import GSPIMDataflow
from .renderer     import GSPIMRenderer, print_stage_timing_statistics
from .profiler     import GSPIMProfiler, BandwidthAnalyzer
from .ablation     import AblationEstimator, create_ablation_report

__all__ = [
    # Memory layer
    'PIMTimeFilter',
    'SamsungPIMBackend',
    'PPIMFilterResult',
    'get_backend',
    'validate_ppim_bandwidth_claim',
    # Runtime layer
    'GSPIMDataflow',
    # Interface layer
    'GSPIMRenderer',
    'print_stage_timing_statistics',
    # Analysis
    'GSPIMProfiler',
    'BandwidthAnalyzer',
    # Ablation
    'AblationEstimator',
    'create_ablation_report',
]
