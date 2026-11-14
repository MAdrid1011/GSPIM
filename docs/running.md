# Running

## Evaluate the workloads

Prepare the inputs and evaluate all 18 model/scene combinations:

```bash
python -m gspim.demo --prepare
python -m gspim.demo
```

For efficient execution and lower trace-storage requirements, the default uses
10 consecutive held-out frames, 100 warmup windows, and four repetitions.
The paper evaluates 300-frame sequences, with 200 frames for Actor1_4. Use a ±5%
relative speedup difference as the reference tolerance when comparing this
shorter run with the paper's evaluation.

Use the same entry for one model or scene:

```bash
python -m gspim.demo --workload ex4dgs --scene sear_steak
python -m gspim.demo --workload 4dgs1k
```

The workload names are `4dgs1k`, `ex4dgs`, and `anchored`. Omitting a selector
includes all models or scenes. Selectors also apply to preparation.

## Full-sequence quality

```bash
python -m gspim.demo --frames 300 --out outputs/quality
```

PSNR and SSIM average the requested held-out frames. Five scenes contain 300
frames, and Actor1_4 contains 200. The report lists the evaluated frame count
for every model and scene.

| Evaluation | Standard run | `--frames 300` |
| --- | --- | --- |
| GPU measurements | First 10 frames | Full available sequence |
| PSNR / SSIM | First 10 frames | Full available sequence |
| PIM trace and schedule | First 10 frames | First 10 frames |

The full-sequence quality pass reuses the compact 10-frame traces to keep trace
construction and storage small. PIM selection and data reorganization preserve
the Dataflow images, so Full PSNR/SSIM uses the Dataflow rendering. Trace RGB
hashes and window decisions are checked against current rendering before each
measurement.

## Timing and progress

Baseline and Dataflow measurement order alternates across repetitions. CUDA
events measure rendering after warmup; input loading, compilation, image decoding,
and quality evaluation take place outside this interval. Speedups use per-frame
GPU measurements and per-frame PIM schedule times.

The terminal shows the current model and scene, trace preparation, CUDA warmup,
measurement progress, quality frame counts, and PIM scheduling. Each completed
combination prints Dataflow, PIM-only, Serial, and Full speedups.

Traces are associated with the inputs, device, and recording implementation;
schedules are associated with the trace and scheduler. Matching traces and
schedules are reused, while GPU measurements and quality are collected on every
run. Compilation products, raw measurements, and reusable traces stay in the
external cache.

## Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--frames` | 10 | First consecutive held-out frames, from 2 through 300 |
| `--repeats` | 4 | Measurement repetitions |
| `--warmup` | 100 | Warmup windows |
| `--checkpoint` | Manifest path | Checkpoint override for one combination |
| `--images` | Manifest path | Held-out images override for one combination |
| `--cache` | `~/.cache/gspim/` | External download, build, trace, schedule and measurement cache |
| `--out` | `outputs/` | Report destination |

For multiple combinations, outputs are grouped by model and scene. For a single
combination, an explicit `--out` is used directly. Repeating an output path
refreshes its reports. The [AGX Orin setup guide](setup.md#external-storage)
shows how to place inputs, caches, and results on an external SSD.

## Results

`verification.md` contains the four speedups and Baseline/Dataflow/Full quality
columns. `performance.{png,pdf}` and `ablation.{png,pdf}` contain normalized
speedups; plot labels abbreviate 4DGS-1K as `1K`. Each combination also has a
`summary.md`.
