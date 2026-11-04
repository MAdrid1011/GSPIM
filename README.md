# GSPIM

GSPIM combines a CUDA rendering dataflow for dynamic Gaussian splatting with
Polymorphic PIM (PPIM), a Hierarchical Compaction Fabric (HCF), and task scheduling
models. It supports explicit 4D Gaussians with contribution masks, Ex4DGS keyframes,
and four-dimensional anchors with neural decoders across four Neural 3D Video scenes.

## Install and run

Use a CUDA-enabled Python environment with FFmpeg and a compatible C++ compiler.
See [setup](docs/setup.md) for the toolchain and dependency versions.

```bash
python -m pip install -e '.[cuda]'
python -m gspim.demo --prepare
python -m gspim.demo
```

Preparation downloads public data and Ex4DGS models, preprocesses images, and
checks the supplied 1K and Anchored checkpoints in `checkpoints/`. Model paths and
sources are described in [workloads](docs/workloads.md). The default run evaluates
all twelve workload/scene combinations and writes Markdown reports to `outputs/`.

## Repository

| Directory | Contents |
| --- | --- |
| `gspim/` | CUDA pipeline, preparation, runtime, and model construction functions |
| `integrations/` | Checkpoint adapters and shared workload manifest |
| `hardware/` | PPIM, HCF, control logic, and RTL tests |
| `reference_model/` | Independent algorithm references for design tests |
| `tests/` | Functional design tests |
| `docs/` | Design and operation documentation |
| `datasets/`, `checkpoints/` | Local data and model files, ignored by Git |
| `third_party/` | Pinned upstream source checkouts, ignored by Git |
| `outputs/` | Generated Markdown reports, ignored by Git |

## Documentation

- [Setup](docs/setup.md): environment and installation.
- [Workloads](docs/workloads.md): representations, sources, model formats, and training recipes.
- [Data](docs/data.md): preparation, cameras, pixels, and splits.
- [Dataflow](docs/dataflow.md): GPU stages, ordering, and batching.
- [Running](docs/running.md): commands, options, reports, and metrics.
- [Architecture](docs/architecture.md): PPIM and HCF organization.
- [Runtime](docs/runtime.md): tasks, buffers, dependencies, and synchronization.

The repository license is in [LICENSE](LICENSE). Dataset, model, and dependency
licenses are listed in [third-party notices](THIRD_PARTY_NOTICES.md).
