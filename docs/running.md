# Running

## All combinations or one combination

From the checkout, prepare inputs and run all twelve combinations:

```bash
python -m gspim.demo --prepare
python -m gspim.demo
```

Preparation downloads and preprocesses public inputs and checks supplied models.
It does not start training. Rendering checks every selected input before loading
CUDA models. Missing inputs produce a list of paths and preparation instructions.

A single combination uses the same entry point:

```bash
python -m gspim.demo --workload ex4dgs --scene sear_steak
```

`--workload explicit4d` and `--workload anchored` select the other representations.
Omitting `--scene` runs all four scenes for the selected workload; omitting
`--workload` runs all three representations for the selected scene. These selectors
also apply to `--prepare`.

## Options and outputs

| Option | Default | Meaning |
| --- | --- | --- |
| `--frames` | 300 | First consecutive held-out frames, from 2 through 300 |
| `--repeats` | 4 | Timed sequence repetitions per path |
| `--warmup` | 20 | Warmup windows before timing |
| `--checkpoint` | Manifest path | Model override for a single combination |
| `--images` | Manifest path | Held-out image directory override for a single combination |
| `--cache` | `~/.cache/gspim/` | External CUDA compilation and download cache |
| `--out` | `outputs/` | Report destination |

Default reports are `outputs/<workload>/<scene>/summary.md`. For a single
combination, an explicit `--out` is used directly. Multiple combinations append
`<workload>/<scene>/` to the supplied output root. Repeating a run replaces its
report. The default output tree is ignored by Git.

## Timing and quality metrics

The two paths are independent-frame rendering (`baseline`) and windowed rendering
(`dataflow`). Both use FP32, identical camera and background settings, alpha cutoff
1/255, alpha cap 0.999, and transmittance early stopping at 1e-4.

An untimed prepass determines the adaptive ready-window partition. Both paths
receive that partition; Dataflow repeats its window control during timing. The
measured interval includes selection, geometry generation, projection, ordering,
and composition. Disk access, checkpoint loading, warmup, the prepass, and quality
calculation are outside this interval. Repetitions alternate the execution order
of the two paths.

CUDA events measure sequence durations and frame completion latency. Baseline
records completion after each frame; Dataflow records all window frames complete
after its batch drain. FPS is total rendered frames divided by summed sequence
durations. Speedup is baseline duration divided by Dataflow duration. P95 and P99
use linear quantiles of the recorded per-frame latencies from ready-window submission.

Quality is calculated separately for every requested held-out frame. PSNR uses
mean squared RGB error; SSIM uses an 11 by 11 Gaussian window with zero padding.
Rendered RGB is clamped to [0, 1] for quality calculation. Reports contain actual
run parameters, per-path means, quality differences, repetition durations, and
individual quality and latency samples.
