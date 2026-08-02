# Paper-to-Code Traceability

| Paper contract item | Public implementation | Primary evidence |
| --- | --- | --- |
| S1 explicit temporal activity | `gspim/ppim.py`, `gspim/layouts/explicit4d.py` | `tests/test_ppim.py`, `tests/test_layouts.py` |
| S1 keyframe and anchor activity, with Ex4DGS all-frame S2 support admission | `gspim/layouts/ex4dgs.py`, `gspim/layouts/anchored.py`, `gspim/pipeline.py` | `tests/test_layouts.py`, `tests/test_pipeline.py`, `tests/test_ppim.py` |
| S1 Active Buffer compaction consumed by S2 | `gspim/hcf.py`, `gspim/pipeline.py`, `hardware/src/main/scala/gspim/Hcf.scala` | `tests/test_hcf.py`, `tests/test_pipeline.py`, `hardware/src/test/scala/gspim/HcfSpec.scala` |
| S2 conditional explicit slice and keyframe interpolation | `gspim/layouts/explicit4d.py`, `gspim/layouts/ex4dgs.py` | `tests/test_layouts.py`, `tests/test_pipeline.py` |
| S3 camera-space SS and split metadata consumed by S4 | `gspim/dataflow.py`, `gspim/pipeline.py`, `gspim/ppim.py` | `tests/test_dataflow.py`, `tests/test_pipeline.py`, `tests/test_ppim.py` |
| S4 differentiated sort, ROL, and bounded batch | `gspim/dataflow.py` | `tests/test_dataflow.py`, `tests/test_pipeline.py` |
| S5 projection, tiles, and Batch Buffer compositing | `gspim/renderer.py`, `gspim/pipeline.py` | `tests/test_renderer.py`, `tests/test_pipeline.py`, `tests/golden/explicit4d_window.json`, `tests/golden/ex4dgs_window.json`, `tests/golden/anchored_window.json` |
| PPIM microprogram, 26/52-bit, and mask-controlled access | `gspim/ppim.py`, `hardware/src/main/scala/gspim/PPIMProgramBuffer.scala`, `hardware/src/main/scala/gspim/PPIMDatapath.scala` | `tests/test_ppim.py`, `hardware/src/test/scala/gspim/PpimSpec.scala`, `hardware/src/test/scala/gspim/GeneratedRtlSpec.scala` |
| HCF FIFO, PPIM-mask gather, shared live ranges, and overflow | `gspim/hcf.py`, `hardware/src/main/scala/gspim/Hcf.scala`, `hardware/src/main/scala/gspim/GspimRank.scala` | `tests/test_hcf.py`, `tests/test_pipeline.py`, `hardware/src/test/scala/gspim/HcfSpec.scala`, `hardware/src/test/scala/gspim/GeneratedRtlSpec.scala` |
| LPDDR task WRITE/READ, commands, visibility, and completion | `gspim/runtime.py`, `hardware/src/main/scala/gspim/LpddrTaskAdapter.scala`, `hardware/src/main/scala/gspim/TaskController.scala` | `tests/test_runtime.py`, `hardware/src/test/scala/gspim/LpddrTaskAdapterSpec.scala`, `hardware/src/test/scala/gspim/TaskControllerSpec.scala` |
| Hybrid physical-block/batch/window scheduling | `gspim/pipeline.py`, `gspim/runtime.py`, `gspim/scheduling.py` | `tests/test_runtime.py`, `tests/test_pipeline.py`, `tests/golden/explicit4d_six_frame_sequence.json` |
| Generated RTL behavior | `hardware/src/main/scala/gspim/Generate.scala` | `hardware/src/test/scala/gspim/GeneratedRtlSpec.scala`, `hardware/verify-rtl.sh` |
| Workload-category mechanism coverage and three independent fixture oracles (not Methodology reproduction) | `integrations/`, `reference_model/`, `docs/workloads.md`, `docs/artifact-scope.md` | `tests/test_integrations.py`, `tests/test_pipeline.py`, `tests/golden/ex4dgs_window.json`, `tests/golden/anchored_window.json`, `tools/check_public_tree.py` |

The release gate verifies that every listed source and evidence path exists and
is tracked in a release. It rejects private paper material, legacy code, and
untracked required public artifacts.
