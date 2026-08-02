# GSPIM: A Temporal-Locality-Aware 4DGS Accelerator with Polymorphic PIM and Multi-Frame Rendering

This repository is an executable mechanism artifact for GSPIM. It provides
deterministic functional and Chisel transaction models for the paper's five
stages, Polymorphic PIM (PPIM), Hierarchical Compaction Fabric (HCF), and
hybrid runtime. It is intended for inspection and regression testing.

## What This Artifact Proves

- S1 runs an executable representation-specific PPIM program and compacts its
  selected, aligned payloads into the Active Buffer that S2 actually consumes.
- S2 generates 3D means and covariances for all active records and timestamps.
- S3 derives camera-space depth stability and compacts stable and unstable
  sorting metadata into separate regions that S4 actually consumes.
- S4 builds stable-once and unstable-per-frame orders, merged frame ROLs, and
  shared batches whose actual footprint fits the configured effective L2.
- S5 performs deterministic 2D projection, tile assignment, and alpha
  compositing directly from the current HCF-packed ping-pong batch buffer.
- `PIM_SELECT` and `PIM_REORG` carry program, region, block, purpose,
  dependency, block-completion, and final-count information.

The artifact does not reproduce or claim AGX Orin performance, energy,
quality, Ramulator timing, OpenROAD PPA, datasets, or paper headline numbers.
See [artifact scope](docs/artifact-scope.md).

## Quick Start

```bash
python -m pip install -e '.[test]'
python -m pytest
python -m unittest discover -s tests -v
python -m gspim.demo --workload all --out out/fixture-run
python tools/check_public_tree.py
```

The hardware model is independent from the Python model:

```bash
cd hardware
sbt test
./verify-rtl.sh
```

Generated SystemVerilog remains in ignored `hardware/generated/`.

## Workloads

- **Explicit 4D / 4DGS-1K** uses the temporal-covariance PPIM activity
  program from the GSPIM programming example; it does not use a neighboring
  keyframe-mask OR shortcut.
- **Ex4DGS** uses static/keyframe-support selection, keyframe interpolation,
  and an explicit aligned record location.
- **Anchored 4DGS** uses interval overlap and same-die `binding_start` plus
  `binding_count` expansion into explicitly located payload records.

The deterministic fixture family contains six timestamps; individual window
runs use two to five frames, while the sequence test exercises six-frame
loopback. External worktrees and datasets are optional local inputs and are
never committed. Start review with
[the paper contract](docs/paper-contract.md) and [the traceability matrix](docs/traceability.md).

## Repository Map

- `gspim/`: executable functional model and CLI.
- `reference_model/`: independent oracle used by regression tests.
- `tests/`: fixtures, golden outputs, and integration tests.
- `hardware/`: Chisel PPIM/HCF/task implementation and RTL tests.
- `integrations/`: validated external workload adapters without vendored code.
- `docs/`: public contract, scope, and paper-to-code traceability.
