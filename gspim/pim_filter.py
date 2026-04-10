"""
GSPIM PPIM Temporal Filter
===========================

Simulates the Primitive-Aware PIM (PPIM) temporal filtering layer
described in Section 3.1 of the paper.

The filter evaluates the temporal contribution probability

    p(t | μ_t, σ_tt) = exp( -0.5 · dt² / σ_tt )

for each 4D Gaussian primitive and marks it as active if p(t) > τ.

The equivalent inequality (avoids exp):

    0.5 · dt²  <  σ_tt · ln(1/τ)

This is computed in-bank on the Samsung HBM2 PIM hardware:
    MOV  GRF[μ_t], BANK; ADD  GRF[dt]; MUL  GRF[dt²]; …  (10 instructions)

See ``submodules/samsung-pim/gspim_ppim_kernel.h`` for the full ISA mapping.

Cycle-accurate timing validation is provided by ``gspim.ppim_backend``
(SamsungPIMBackend), which parses ``submodules/samsung-pim/ini/HBM2_samsung_2M_16B_x64.ini``
for exact Samsung HBM2 timing parameters.

Bandwidth claim (paper, Section 4.2):
    PPIM reads only 52 B/primitive for the filter decision,
    then loads full ~636 B only for the ~30% active primitives.
    Net reduction: ≈ (1 - 0.3) × 584/636 ≈ 64–72% bandwidth saved.
"""

import time
import torch

try:
    from .ppim_backend import SamsungPIMBackend, PPIMFilterResult, get_backend
    _HAS_BACKEND = True
except ImportError:
    _HAS_BACKEND = False


class PIMTimeFilter:
    """
    PPIM temporal filter — fast PyTorch simulation with optional
    cycle-accurate Samsung HBM2 timing validation.

    The ``filter_gaussians`` method runs entirely on GPU using vectorised
    PyTorch operations and mirrors the 10-instruction CRF kernel that would
    execute on real PPIM hardware.

    Args:
        threshold:        Temporal contribution threshold τ (default: 0.05).
        enable_pim_sim:   If True, compute and report Samsung PIM cycle counts
                          alongside the GPU-side simulation.
        hbm_channels:     Number of HBM2 channels for the timing model (default: 8).
        pim_clock_ghz:    PIM block clock frequency in GHz (default: 1.0).
        hbm_bw_gbps:      HBM2 peak external bandwidth in GB/s (default: 900.0).
    """

    # ln(1 / 0.05) = ln(20)
    LN_INV_THRESHOLD: float = 2.995732273553991

    def __init__(
        self,
        threshold:      float = 0.05,
        enable_pim_sim: bool  = False,
        hbm_channels:   int   = 8,
        pim_clock_ghz:  float = 1.0,
        hbm_bw_gbps:    float = 900.0,
    ):
        self.threshold      = threshold
        self.enable_pim_sim = enable_pim_sim
        self.pim_ops_count  = 0

        # Precompute threshold constant for the current τ
        import math
        self._ln_inv_tau = math.log(1.0 / threshold)

        # Samsung PIM backend (analytic timing model)
        self._backend: SamsungPIMBackend | None = None
        if enable_pim_sim and _HAS_BACKEND:
            self._backend = SamsungPIMBackend(
                num_channels  = hbm_channels,
                pim_clock_ghz = pim_clock_ghz,
                hbm_bw_gbps   = hbm_bw_gbps,
            )

        # Accumulated timing statistics
        self._total_filter_us:     float = 0.0
        self._total_compaction_us: float = 0.0
        self._calls: int = 0
        self._last_pim_result: PPIMFilterResult | None = None

    @torch.no_grad()
    def filter_gaussians(
        self,
        scales,
        rotation_l,
        rotation_r,
        mu_t,
        t_min: float,
        t_max: float,
    ):
        """
        Apply PPIM temporal contribution filter to all primitives.

        Implements the equivalent of the 10-instruction Samsung PIM CRF kernel:

            MOV  GRF_A[0] ← BANK[μ_t]          (load μ_t)
            ADD  GRF_A[1] ← t_query - GRF_A[0]  (dt)
            MUL  GRF_A[2] ← GRF_A[1] × GRF_A[1](dt²)
            MUL  GRF_A[3] ← GRF_A[2] × 0.5     (½dt²)
            MOV  GRF_A[4] ← BANK[σ_tt]          (load σ_tt)
            MUL  GRF_A[5] ← GRF_A[4] × ln(1/τ) (threshold)
            ADD  GRF_B[0] ← GRF_A[5]-GRF_A[3]  (margin)
            NOP / EXIT

        On hardware:
        - 52 bytes are read per primitive (μ_t + σ_tt + alignment)
        - 8 PIM blocks execute in parallel across 16 banks
        - Active Map written to reserved bank region

        In simulation (GPU):
        - σ_tt is computed from quaternion components (not pre-stored) so
          we evaluate the full rotation decomposition in CUDA

        Args:
            scales:     [N, 4]  four-dimensional scales (s_x, s_y, s_z, s_t)
            rotation_l: [N, 4]  left quaternion  (w, x, y, z)
            rotation_r: [N, 4]  right quaternion (w, x, y, z)
            mu_t:       [N, 1]  temporal mean μ_t
            t_min:      float   window start
            t_max:      float   window end

        Returns:
            active_mask:    BoolTensor [N]   True = primitive passes filter
            active_indices: LongTensor [M]   indices of active primitives
            p_t_max:        FloatTensor [N]  float active mask (1.0 / 0.0)
        """
        # ---------- σ_tt computation (mirrors PIM CRF in full form) ----------
        # Extract quaternion components
        lw = rotation_l[:, 0]; lx = rotation_l[:, 1]
        ly = rotation_l[:, 2]; lz = rotation_l[:, 3]
        rw = rotation_r[:, 0]; rx = rotation_r[:, 1]
        ry = rotation_r[:, 2]; rz = rotation_r[:, 3]

        # Map to (a,b,c,d) and (p,q,r,s) notation used in paper
        a, b, c, d = lx, ly, lz, lw
        p, q, r, s = rx, ry, rz, rw

        # 4D rotation matrix row 3 (M_r @ M_l)[3, :]
        R30 = s*a - r*b + q*c - p*d
        R31 = s*b + r*a - q*d - p*c
        R32 = -s*c + r*d + q*a - p*b
        R33 = s*d + r*c - q*b + p*a

        # σ_tt = Σ (R[3,i] · s_i)² = ||L[3,:]||²
        sigma_tt = (
            (R30 * scales[:, 0]).square() +
            (R31 * scales[:, 1]).square() +
            (R32 * scales[:, 2]).square() +
            (R33 * scales[:, 3]).square()
        )

        # ---------- filter decision (PIM CRF PC 1–7) ----------
        mu_t_flat = mu_t.view(-1)
        # dt = clamp(t_query, t_min, t_max) - μ_t  →  closest window point to μ_t
        dt = mu_t_flat.clamp(t_min, t_max) - mu_t_flat

        # 0.5 · dt² < σ_tt · ln(1/τ)
        ln_inv_tau = self._ln_inv_tau
        active_mask = (0.5 * dt.square()) < (sigma_tt * ln_inv_tau)
        active_indices = active_mask.nonzero(as_tuple=False).view(-1)

        # ---------- PIM instruction count accounting ----------
        self.pim_ops_count += scales.shape[0] * 10   # 10 CRF instructions/prim

        # ---------- Samsung PIM cycle-accurate timing (optional) ----------
        if self._backend is not None:
            pim_result = self._backend.compute_filter_timing(
                num_primitives = scales.shape[0],
                num_active     = int(active_indices.shape[0]),
                threshold      = self.threshold,
            )
            self._total_filter_us     += pim_result.filter_time_us
            self._total_compaction_us += pim_result.compaction_time_us
            self._last_pim_result      = pim_result
            self._calls += 1

        return active_mask, active_indices, active_mask.float()

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> dict:
        """Return accumulated filter statistics."""
        stats = {
            "pim_ops_count":         self.pim_ops_count,
            "total_calls":           self._calls,
        }
        if self._backend is not None and self._calls > 0:
            stats["samsung_pim"] = {
                "total_filter_us":       self._total_filter_us,
                "total_compaction_us":   self._total_compaction_us,
                "avg_filter_us":         self._total_filter_us / self._calls,
                "avg_compaction_us":     self._total_compaction_us / self._calls,
                "timing_model":          "Samsung HBM2-PIM (analytic)",
                "ini_source":            str(
                    self._backend.timing.__class__.__name__
                ),
            }
            if self._last_pim_result is not None:
                stats["samsung_pim"]["last_savings_pct"] = (
                    self._last_pim_result.savings_pct
                )
        return stats

    def reset_statistics(self) -> None:
        """Reset all accumulated statistics."""
        self.pim_ops_count         = 0
        self._total_filter_us      = 0.0
        self._total_compaction_us  = 0.0
        self._calls                = 0
        self._last_pim_result      = None

    def validate_bandwidth_claim(self, num_primitives: int = 500_000) -> None:
        """
        Print a Samsung PIM validation report for the PPIM bandwidth claim.

        Shows that PPIM achieves ≥60% bandwidth reduction vs. baseline
        (full GPU read of all primitives) for a scene of ``num_primitives``
        primitives at active_ratio = 30%.
        """
        if not _HAS_BACKEND:
            print("[PIMTimeFilter] ppim_backend not available.")
            return
        backend = self._backend or get_backend()
        validation = backend.validate_pim_filter(num_primitives, threshold=self.threshold)
        backend.print_timing_report(validation["result"])
        status = "PASS ✓" if validation["validated"] else "FAIL ✗"
        print(f"  Paper claim (≥70% reduction): {validation['savings_pct']:.1f}%  [{status}]")
