# Running

## All combinations or one combination

From the checkout, prepare inputs and run all eighteen combinations:

```bash
python -m gspim.demo --prepare
python -m gspim.demo
```

Preparation downloads and preprocesses public inputs and checks supplied models.
It does not start training. Rendering checks every selected input before loading
CUDA models. Missing inputs produce a list of paths and preparation instructions.

The first run compiles GPU publication kernels and builds compact PIM traces
from the selected model and images. Subsequent runs reuse matching traces and
schedules from the external cache. Raw capture files are temporary.

A single combination uses the same entry point:

```bash
python -m gspim.demo --workload ex4dgs --scene sear_steak
```

`--workload explicit4d` and `--workload anchored` select the other representations.
Omitting `--scene` runs all six scenes for the selected workload; omitting
`--workload` runs all three representations for the selected scene. These selectors
also apply to `--prepare`.

Use the full available image sequence with:

```bash
python -m gspim.demo --frames 300
```

Actor1_4 uses its 200 available frames. Reports list the rendered frame count
and the PIM trace frame count.

## Options and outputs

| Option | Default | Meaning |
| --- | --- | --- |
| `--frames` | 10 | First consecutive held-out frames, from 2 through 300 |
| `--repeats` | 4 | Number of execution repetitions |
| `--warmup` | 100 | Number of warmup windows |
| `--checkpoint` | Manifest path | Model override for a single combination |
| `--images` | Manifest path | Held-out image directory override for a single combination |
| `--cache` | `~/.cache/gspim/` | External compilation, download, trace and schedule cache |
| `--out` | `outputs/` | Report destination |

Default reports are `outputs/<workload>/<scene>/summary.md`. For a single
combination, an explicit `--out` is used directly. Multiple combinations append
`<workload>/<scene>/` to the supplied output root. Repeating a run replaces its
report. The default output tree is ignored by Git.

## Outputs

The default run writes one summary for each model and scene under `outputs/`,
the combined `outputs/verification.md` file, and the two charts
`outputs/performance.{pdf,png}` and `outputs/ablation.{pdf,png}`. The combined
report contains:

| Section | Contents |
| --- | --- |
| Performance | GPU+Dataflow and Full GSPIM speedup, time, P95 and P99 |
| Ablation | The five execution paths used by the schedule |
| Quality | Baseline, Dataflow, and GSPIM PSNR/SSIM |
