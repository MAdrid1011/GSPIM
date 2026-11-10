# GSPIM

GSPIM combines a CUDA rendering dataflow for dynamic Gaussian splatting with
Polymorphic PIM (PPIM), a Hierarchical Compaction Fabric (HCF), and task scheduling
models. It supports explicit 4D Gaussians with contribution masks, Ex4DGS keyframes,
and four-dimensional anchors with neural decoders across six camera scenes.

## Inputs

Download the GSPIM checkpoint bundle from the project
[Releases](https://github.com/MAdrid1011/GSPIM/releases) and extract it at the
repository root. The resulting `checkpoints/` tree contains one checkpoint for
each of the eighteen workload/scene combinations.

The complete six-scene run also uses the official Dance3 and Actor1_4 packages.
Request them from [Mobile-Stage](https://docs.google.com/forms/d/e/1FAIpQLSeEbjuTV7w0lfryl-9FPX1VteuPGkbqjDvxXebY02Tm6BMejQ/viewform?usp=sf_link)
and [ENeRF-Outdoor](https://docs.google.com/forms/d/e/1FAIpQLSeFAgcnJEDYEaco4vA2QsD-bM8V8CRFuiR0QWbi_3uXmXtUKg/viewform?usp=sf_link),
then place `dance3_libx265.tar.gz` and `actor1_4_libx265.tar.gz` in
`~/.cache/gspim/downloads/`. The remaining four scene packages are downloaded
by the preparation command.

## Install and run

Use a CUDA-enabled Python environment with FFmpeg and a compatible C++ compiler.
See [setup](docs/setup.md) for the toolchain and dependency versions.

```bash
python -m pip install -e '.[cuda]'
python -m gspim.demo --prepare
python -m gspim.demo
```

Preparation downloads, converts, and verifies the registered inputs. Model paths
and formats are described in [workloads](docs/workloads.md). The default run
evaluates all eighteen workload/scene combinations with a ten-frame quick pass
and writes the result tables and charts to `outputs/`.

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
| `third_party/` | Pinned upstream source checkouts and the Golden-Ratio plotting submodule |
| `outputs/` | Generated Markdown reports, ignored by Git |
| `traces/` | Regenerated compact PIM traces, ignored by Git |

## Documentation

- [Setup](docs/setup.md): environment and installation.
- [Workloads](docs/workloads.md): representations, checkpoint formats, and model parameters.
- [Data](docs/data.md): preparation, cameras, pixels, and splits.
- [Dataflow](docs/dataflow.md): GPU stages, ordering, and batching.
- [Running](docs/running.md): commands, options, and outputs.
- [Architecture](docs/architecture.md): PPIM and HCF organization.
- [Runtime](docs/runtime.md): tasks, buffers, dependencies, and synchronization.

The repository license is in [LICENSE](LICENSE). Dataset, model, and dependency
licenses are listed in [third-party notices](THIRD_PARTY_NOTICES.md).
