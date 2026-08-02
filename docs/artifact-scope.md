# Artifact Scope

## Reproducible Here

This artifact reproduces the **mechanisms** described in the paper:

- Executable TEMP_ACTIVITY, KEYFRAME_RANGE, ANCHOR_OVERLAP, and DEPTH_STABLE
  PPIM programs with signed 26-bit operands, rounded 52-bit products, and
  conservative threshold behavior.
- Mask-controlled S1 and S3 selection, bank/die compaction, reserved regions,
  task descriptors, completions, and cache-visibility protocol.
- The five-stage dataflow and hybrid block, batch, and cross-window task graph.
- Functional representations for explicit 4D/4DGS-1K, Ex4DGS, and Anchored
  4DGS, including payload movement and deterministic raster output.
- Inspectable Chisel modules for the PPIM/HCF/task-control architecture.

The deterministic fixture family contains six timestamps. Its window tests use
two to five frames, while the sequence test exercises the six-frame loopback.
They validate selection, record layouts, ROLs, batch boundaries, task traces,
and end-to-end state transitions. They do not require an AGX Orin, a dataset,
or a trained checkpoint.

## Not Reproduced Here

The checkout has no Jetson AGX Orin or PPIM-enabled LPDDR5 hardware. Therefore
it does not recalculate or claim to validate manuscript FPS, speedup, energy,
LPDDR5 command timing, area, or power. Local timing, if a caller collects it,
is labeled as local-host timing only and is never converted into paper claims.

The repository has no cycle, energy, Ramulator, or PPA substitute.
Chisel/Verilator proves RTL transaction behavior, not OpenROAD PPA.
The generated RTL top is one die-level tile. The paper's package and die counts
are configuration context, not an instantiated multi-die memory-system model.

The paper's Methodology is not reproduced as an experiment package. In
particular, this checkout contains no native workload checkpoints or scenes,
unified CUDA/CUB/gsplat backend, AGX Orin tuning or measurements, specialized
accelerator reconstruction, Ramulator integration, energy traces, or physical
implementation flow. The three checked-in fixtures cover only the temporal
representation categories required to exercise the mechanisms.

## Data and External Code

Paper source files and the two local PDFs are private local material. They are
ignored by Git and are neither modified nor required by the public artifact.
No dataset, checkpoint, generated RTL, benchmark result, or third-party
workload source is committed. The Ex4DGS integration fetches an official
checkout outside version control; its license remains authoritative. The
release gate checks that these local-only roots are untracked, so a local
checkout or generated output never requires deletion before a public check.
All tracked public files are ASCII English; the same release gate rejects a
non-ASCII byte before publication.
