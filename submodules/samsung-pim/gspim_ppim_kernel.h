/**
 * GSPIM PPIM Temporal Filter Kernel — Samsung PIM ISA Mapping
 * ============================================================
 *
 * This file documents how the GSPIM temporal contribution filter
 * (Section 3.1 of the paper) maps onto the Samsung SATIPIMSimulator
 * PIM instruction set architecture.
 *
 * Memory layout per primitive in HBM bank (52 bytes = 13 × float32):
 *   Offset  Field          Size  Description
 *   ------  -----          ----  -----------
 *    0      mu_t            4 B  Temporal mean μ_t
 *    4      sigma_tt        4 B  Precomputed σ_tt = Σ(R3i·si)²
 *    8      R30_s0          4 B  R[3,0] * s_0  (for σ_tt recompute)
 *   12      R31_s1          4 B  R[3,1] * s_1
 *   16      R32_s2          4 B  R[3,2] * s_2
 *   20      R33_s3          4 B  R[3,3] * s_3
 *   24      opacity         4 B  Base opacity (for early-exit)
 *   28–51   (reserved / pad)     Future: Δxyz or covariance prefix
 *
 * SRF (scalar register) layout (broadcast before PIM mode):
 *   SRF[0] = t_query          — current rendering timestamp
 *   SRF[1] = 0.5              — half constant
 *   SRF[2] = LN_THRESHOLD     — ln(1/τ) ≈ 2.996 for τ=0.05
 *   SRF[3] = 0.0              — zero (for sign checks)
 *
 * CRF (command register) program:
 *   PC 0:  MOV  GRF_A[0], BANK[mu_t]          ; load μ_t
 *   PC 1:  ADD  GRF_A[1], SRF[0], -GRF_A[0]   ; dt = t_query - μ_t
 *   PC 2:  MUL  GRF_A[2], GRF_A[1], GRF_A[1]  ; dt² = dt * dt
 *   PC 3:  MUL  GRF_A[3], GRF_A[2], SRF[1]    ; half_dt² = 0.5 * dt²
 *   PC 4:  MOV  GRF_A[4], BANK[sigma_tt]       ; load σ_tt
 *   PC 5:  MUL  GRF_A[5], GRF_A[4], SRF[2]    ; thresh = σ_tt * ln(1/τ)
 *   PC 6:  ADD  GRF_B[0], GRF_A[5], -GRF_A[3] ; margin = thresh - half_dt²
 *   PC 7:  NOP                                  ; wait for comparison
 *   PC 8:  EXIT                                 ; end kernel
 *
 * Active Map write (after GRF_B[0] sign check):
 *   If GRF_B[0] > 0  →  write 1 to Active Map row for this primitive
 *   Else             →  write 0 (primitive filtered out)
 *
 * Instruction counts per primitive:
 *   2 × MOV  (memory reads: μ_t, σ_tt = 8 B = 2 burst reads)
 *   4 × MUL  (dt², half_dt², thresh, — MUL instruction)
 *   2 × ADD  (dt, margin)
 *   1 × NOP  (pipeline drain)
 *   1 × EXIT
 *   Total: 10 PIM instructions per primitive
 *   Memory reads: 2 × 4 B = 8 B from bank row buffer
 *   Memory writes: 1 bit to Active Map (batched)
 *
 * Throughput analysis (Samsung HBM2, 1 GHz PIM clock, 8 PIM blocks):
 *   Per-PIM-block primitives: N / (NUM_BANKS / NUM_PIM_BLOCKS) = N/2
 *   Cycles per primitive: max(RL, 10 instructions) ≈ 20 cycles
 *   Bank-parallel: 8 PIM blocks process in parallel
 *   Total cycles: (N/8) * 20 + tRCD + tRP overhead
 *
 * Comparison with baseline (GPU reads all primitives):
 *   Baseline: N × 636 B/prim × HBM_latency
 *   PPIM:     N × 52 B read (filter) + N_active × 636 B (active load)
 *   Savings:  (1 - active_ratio) × N × 584 B ≈ 70% BW reduction at τ=0.05
 */

#pragma once

#include <cstdint>
#include <cmath>

namespace gspim_ppim {

// --- Constants mirroring pim_filter.py ---
constexpr float DEFAULT_THRESHOLD = 0.05f;
constexpr float LN_INV_THRESHOLD  = 2.995732274f;  // ln(1/0.05) = ln(20)
constexpr int   BYTES_PER_PRIM_FILTER = 52;         // bytes read per primitive
constexpr int   BYTES_PER_PRIM_FULL   = 636;        // full attribute bytes

// --- Samsung HBM2 PIM block constants (from ini/HBM2_samsung_2M_16B_x64.ini) ---
constexpr int NUM_BANKS      = 16;
constexpr int NUM_PIM_BLOCKS = 8;           // 1 PIM block per 2 banks
constexpr int RL             = 20;          // read latency (cycles)
constexpr int BL             = 4;           // burst length
constexpr int DEVICE_WIDTH   = 64;          // bits per device
constexpr int BURST_BYTES    = BL * DEVICE_WIDTH / 8;  // 32 bytes per burst

// PIM instructions per primitive for the temporal filter kernel
constexpr int PIM_INSTRUCTIONS_PER_PRIM = 10;

/**
 * Compute PPIM filter cycle count for N primitives on K channels.
 *
 * Models the Samsung PIM cycle-accurate timing:
 *   - Bank activation: tRCDRD = 14 cycles
 *   - PIM instruction throughput: 1 instr/cycle per PIM block
 *   - Bank-parallel execution: NUM_PIM_BLOCKS / NUM_BANKS banks share one PIM block
 *
 * @param num_primitives  Total number of 4D Gaussian primitives
 * @param num_channels    Number of HBM channels (default: 8 for 8-stack HBM2)
 * @param pim_clock_ghz   PIM block clock frequency in GHz (default: 1.0)
 * @return                Estimated wall-clock time in microseconds
 */
inline float compute_ppim_filter_time_us(
    uint64_t num_primitives,
    int      num_channels   = 8,
    float    pim_clock_ghz  = 1.0f)
{
    // Primitives distributed across channels × banks × PIM blocks
    int total_pim_blocks = num_channels * NUM_PIM_BLOCKS;

    // Primitives per PIM block (ceiling division)
    uint64_t prims_per_block = (num_primitives + total_pim_blocks - 1) / total_pim_blocks;

    // Cycles: bank activation + PIM instruction throughput
    int bank_overhead_cycles = 14 + 14;  // tRCDRD × 2 (activate + precharge)
    uint64_t compute_cycles  = prims_per_block * PIM_INSTRUCTIONS_PER_PRIM;
    uint64_t total_cycles    = compute_cycles + bank_overhead_cycles;

    // Wall-clock time in microseconds
    float time_us = static_cast<float>(total_cycles) / (pim_clock_ghz * 1000.0f);
    return time_us;
}

/**
 * Compute Active Loader (compaction) time in microseconds.
 *
 * After filtering, the Active Loader performs a scatter-gather:
 *   - Reads 636 B per active primitive from HBM
 *   - Writes to contiguous Active Buffer in HBM
 *
 * @param num_active      Number of active (non-filtered) primitives
 * @param hbm_bw_gbps     HBM peak bandwidth in GB/s (Samsung HBM2: ~900 GB/s per stack)
 * @return                Estimated compaction time in microseconds
 */
inline float compute_active_loader_time_us(
    uint64_t num_active,
    float    hbm_bw_gbps = 900.0f)
{
    // Bytes read + bytes written (double transfer for compaction)
    float total_bytes = static_cast<float>(num_active) * BYTES_PER_PRIM_FULL * 2.0f;
    // GB/s → B/us = GB/s × 1000
    float bw_b_per_us = hbm_bw_gbps * 1000.0f;
    return total_bytes / bw_b_per_us;
}

}  // namespace gspim_ppim
