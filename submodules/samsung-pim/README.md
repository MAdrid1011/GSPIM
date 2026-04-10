# Samsung SATIPIMSimulator — GSPIM Submodule

This submodule provides the cycle-accurate HBM2 PIM simulator used to
validate GSPIM's PPIM (Primitive-Aware PIM) timing claims.

## Source

**Samsung Electronics SATIPIMSimulator**  
Copyright (C) 2021 Samsung Electronics Co. LTD  
[Restricted to non-commercial / academic research use]

Original repository: internal Samsung research distribution.  
Contact: Samsung Memory Research Team (see original distribution).

## How to obtain the full simulator

The full C++ source (`src/`, `lib/`, `Sconstruct`) is available under a
Samsung research license.  To obtain it:

1. Contact the Samsung authors listed above.
2. Clone and place the full simulator tree at this path:
   `submodules/samsung-pim/`
3. Build with `scons`:
   ```bash
   cd submodules/samsung-pim
   sudo apt install scons libgtest-dev
   scons
   ```

The `gspim/ppim_backend.py` Python bridge works in two modes:

- **Analytic mode** (default, no build required): Uses the HBM2 timing
  parameters from `ini/HBM2_samsung_2M_16B_x64.ini` to compute
  cycle-accurate PPIM latency analytically.
- **Simulation mode** (requires built binary): Calls `./sim` via subprocess
  and parses the cycle-count output for exact validation.

## Role in GSPIM

```
GSPIM paper claim          Samsung PIM mapping
──────────────────────────────────────────────────────
PPIM bank-parallel filter  → PIM block per 2 banks (NUM_PIM_BLOCKS=8)
52 B read per primitive     → 2 × HBM2 bursts (BL=4, bus=256 bit)
MAC-based σ_tt computation  → MAC / MUL PIM ISA instructions
Active Map write            → PIM_WRITE to reserved bank region
Active Loader compaction    → sequential PIM_READ of active rows
```

## ini files included

| File                             | Description                          |
|----------------------------------|--------------------------------------|
| `ini/HBM2_samsung_2M_16B_x64.ini` | Samsung HBM2 timing parameters     |
| `system_hbm.ini`                 | System-level HBM2 configuration      |
| `system_hbm_1ch.ini`             | Single-channel configuration         |
| `system_hbm_64ch.ini`            | 64-channel configuration             |
