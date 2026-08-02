# Public File Map

This map makes the public review surface explicit. Every tracked source,
test, and document is either paper-mechanism evidence or release-scope
infrastructure. It does not elevate the artifact into a performance,
dataset, or PPA reproduction.

| Path | Role |
| --- | --- |
| `.gitignore` | Excludes private paper material, local checkouts, and generated outputs. |
| `README.md` | Public artifact entry point and scope. |
| `THIRD_PARTY_NOTICES.md` | External workload provenance boundary. |
| `pyproject.toml` | Python package and test entry point. |
| `docs/artifact-scope.md` | Claims and non-claims. |
| `docs/architecture.md` | PPIM/HCF architecture contract. |
| `docs/git-msg-tag.md` | Public commit-message tag convention. |
| `docs/index.md` | Documentation index. |
| `docs/paper-contract.md` | S1--S5, PPIM, HCF, and runtime contract. |
| `docs/public-file-map.md` | This release-audit map. |
| `docs/runtime.md` | Task, visibility, and hybrid scheduling contract. |
| `docs/traceability.md` | Paper mechanism to implementation and evidence links. |
| `docs/workloads.md` | Three paper workload adapter contract. |
| `gspim/__init__.py` | Public Python API. |
| `gspim/README.md` | Functional model module map. |
| `gspim/config.py` | Paper topology and artifact configuration. |
| `gspim/dataflow.py` | S3/S4 depth, order, and batch semantics. |
| `gspim/demo.py` | Deterministic fixture CLI. |
| `gspim/hcf.py` | Functional L1/L2 compaction and live ranges. |
| `gspim/layouts/__init__.py` | Workload adapter exports. |
| `gspim/layouts/anchored.py` | Anchored S1/S2 adapter. |
| `gspim/layouts/base.py` | Shared window validation. |
| `gspim/layouts/ex4dgs.py` | Ex4DGS S1/S2 adapter. |
| `gspim/layouts/explicit4d.py` | Explicit-4D S1/S2 adapter. |
| `gspim/model.py` | Typed model, frame, layout, and geometry interface. |
| `gspim/pipeline.py` | Five-stage functional composition. |
| `gspim/ppim.py` | PPIM fixed-point program semantics. |
| `gspim/renderer.py` | S5 projection, tiles, and compositing. |
| `gspim/runtime.py` | Functional task and visibility protocol. |
| `gspim/scheduling.py` | Ping-pong Batch Buffer ownership state. |
| `gspim/trace.py` | Dependency trace representation. |
| `hardware/README.md` | RTL transaction-model scope and verification command. |
| `hardware/build.sbt` | Chisel build definition. |
| `hardware/project/build.properties` | SBT version pin. |
| `hardware/verify-rtl.sh` | Generate-and-lint verification entry point. |
| `hardware/src/main/scala/gspim/Bundles.scala` | Host task and completion wire contracts. |
| `hardware/src/main/scala/gspim/Generate.scala` | SystemVerilog generation entry point. |
| `hardware/src/main/scala/gspim/GspimRank.scala` | Die-level PPIM/HCF/task composition. |
| `hardware/src/main/scala/gspim/Hcf.scala` | L1/L2 HCF transaction modules. |
| `hardware/src/main/scala/gspim/LpddrTaskAdapter.scala` | Reserved-address LPDDR WRITE/READ task boundary. |
| `hardware/src/main/scala/gspim/PPIMDatapath.scala` | PPIM fixed-point microprogram execution. |
| `hardware/src/main/scala/gspim/PPIMProgramBuffer.scala` | Bank-local microprogram storage. |
| `hardware/src/main/scala/gspim/Params.scala` | RTL topology and arithmetic parameters. |
| `hardware/src/main/scala/gspim/TaskController.scala` | PIM task lifecycle and completion. |
| `hardware/src/test/scala/gspim/GeneratedRtlSpec.scala` | Verilator rank-level behavior test. |
| `hardware/src/test/scala/gspim/HcfSpec.scala` | HCF unit behavior test. |
| `hardware/src/test/scala/gspim/LpddrTaskAdapterSpec.scala` | LPDDR command and completion-read behavior test. |
| `hardware/src/test/scala/gspim/PpimSpec.scala` | PPIM unit behavior test. |
| `hardware/src/test/scala/gspim/TaskControllerSpec.scala` | Task-control unit behavior test. |
| `integrations/__init__.py` | External integration package boundary. |
| `integrations/ex4dgs.py` | Strict Ex4DGS normalized-input adapter. |
| `integrations/manifest.py` | Pinned external provenance. |
| `reference_model/README.md` | Independent oracle scope. |
| `reference_model/__init__.py` | Oracle package marker. |
| `reference_model/contracts.py` | Oracle result contract. |
| `reference_model/dataflow.py` | Independent S3 equation. |
| `reference_model/pipeline.py` | Independent three-workload oracle. |
| `reference_model/renderer.py` | Independent checksum renderer. |
| `reference_model/selection.py` | Independent explicit-4D equations. |
| `scripts/fetch_workloads.py` | Opt-in ignored upstream checkout helper. |
| `tests/README.md` | Regression-suite scope. |
| `tests/__init__.py` | Test package marker. |
| `tests/fixtures.py` | Versioned deterministic inputs. |
| `tests/golden/README.md` | Golden evidence scope. |
| `tests/golden/explicit4d_window.json` | Explicit-4D expected state. |
| `tests/golden/ex4dgs_window.json` | Ex4DGS expected state. |
| `tests/golden/anchored_window.json` | Anchored expected state. |
| `tests/golden/explicit4d_six_frame_sequence.json` | Six-frame window adjustment, loopback, and image checksums. |
| `tests/test_dataflow.py` | S3/S4 regression tests. |
| `tests/test_hcf.py` | HCF regression tests. |
| `tests/test_integrations.py` | Integration and release-gate tests. |
| `tests/test_layouts.py` | Workload adapter tests. |
| `tests/test_pipeline.py` | End-to-end five-stage tests. |
| `tests/test_ppim.py` | PPIM semantics tests. |
| `tests/test_renderer.py` | S5 tests. |
| `tests/test_runtime.py` | Runtime and scheduling tests. |
| `tools/check_public_tree.py` | Release-boundary gate. |
