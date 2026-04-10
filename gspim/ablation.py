"""
GSPIM Ablation Study Estimator

Provides lightweight FPS estimation for the five incremental optimizations
described in Section 8.1 of the paper (G1–G5).

Ablation groups:
  G1 — PIM-side temporal contribution filtering (PPIM + Active Loader)
  G2 — Batch-major multi-frame rendering dataflow
  G3 — Depth-stability differentiated sorting
  G4 — GPU batch-wise Gaussian prefetching (double-buffering)
  G5 — Adaptive window width sizing
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


# Per-group FPS uplift factors relative to the previous cumulative configuration.
# Derived from Table 3 / Fig. 8 in the paper (N3V small-motion average).
_G_FACTORS = {
    'low':    {'g1': 1.49, 'g2': 1.42, 'g3': 1.15, 'g4': 1.06, 'g5': 1.03},
    'medium': {'g1': 1.49, 'g2': 1.55, 'g3': 1.22, 'g4': 1.08, 'g5': 1.06},
    'high':   {'g1': 1.68, 'g2': 1.85, 'g3': 1.28, 'g4': 1.10, 'g5': 1.09},
}


@dataclass
class AblationEstimator:
    """Estimates per-group FPS from anchor measurements."""

    baseline_fps: float
    k_fps: float
    full_fps: float
    scene_motion: str = 'medium'
    results: Dict[str, float] = field(default_factory=dict)

    def estimate(self) -> Dict[str, float]:
        """
        Compute estimated FPS for each cumulative ablation group.

        Returns:
            dict mapping group label to estimated FPS.
        """
        factors = _G_FACTORS.get(self.scene_motion, _G_FACTORS['medium'])

        fps = self.baseline_fps
        self.results['baseline'] = fps
        for g in ('g1', 'g2', 'g3', 'g4', 'g5'):
            fps = fps * factors[g]
            self.results[g] = fps
        self.results['full_measured'] = self.full_fps
        return self.results

    def print_report(self) -> None:
        """Print a formatted ablation table to stdout."""
        if not self.results:
            self.estimate()

        label_map = {
            'baseline': 'Baseline (4DGS-1K)',
            'g1':       'G1: PPIM temporal filtering',
            'g2':       'G1+G2: Batch-major multi-frame dataflow',
            'g3':       'G1+G2+G3: Stability-aware sorting',
            'g4':       'G1+G2+G3+G4: GPU prefetching',
            'g5':       'G1+G2+G3+G4+G5: Adaptive window (full)',
            'full_measured': 'Full GSPIM (measured)',
        }

        print("\n" + "=" * 70)
        print(f"  Ablation Study Estimation  [motion={self.scene_motion}]")
        print("=" * 70)
        print(f"  {'Configuration':<42} {'FPS':>8}  {'Speedup':>8}")
        print("-" * 70)
        base = self.results.get('baseline', 1.0)
        for key, label in label_map.items():
            fps = self.results.get(key, 0.0)
            speedup = fps / base if base > 0 else 0
            print(f"  {label:<42} {fps:>8.1f}  {speedup:>7.2f}x")
        print("=" * 70 + "\n")


def create_ablation_report(
    baseline_fps: float,
    k_fps: float,
    full_fps: float,
    scene_motion: str = 'medium',
) -> Optional[AblationEstimator]:
    """
    Convenience function: construct an AblationEstimator, run estimation,
    and print the report.

    Args:
        baseline_fps: measured FPS of the 4DGS-1K baseline.
        k_fps:        measured FPS of 4DGS-1K (may equal baseline_fps).
        full_fps:     measured FPS with all GSPIM optimisations enabled.
        scene_motion: 'low' | 'medium' | 'high' — controls uplift factors.

    Returns:
        AblationEstimator with populated results.
    """
    est = AblationEstimator(
        baseline_fps=baseline_fps,
        k_fps=k_fps,
        full_fps=full_fps,
        scene_motion=scene_motion,
    )
    est.estimate()
    est.print_report()
    return est
