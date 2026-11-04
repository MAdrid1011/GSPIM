# Workloads

## Representations and scenes

The workload manifest is [`integrations/manifest.py`](../integrations/manifest.py).
It defines all twelve combinations, their default paths, public sources, SHA256
values, upstream revisions, and model parameters. Scene names are
`coffee_martini`, `cook_spinach`, `flame_steak`, and `sear_steak`.

| Workload identifier | Representation | Checkpoint path |
| --- | --- | --- |
| `explicit4d` | 4DGS-1K: explicit 4D Gaussians and contribution masks | `checkpoints/4dgs1k/<scene>.pth` |
| `ex4dgs` | Static Gaussians and interpolated dynamic keyframes | `checkpoints/ex4dgs/<scene>.zip` |
| `anchored` | 4D anchors, ten bindings each, and neural decoders | `checkpoints/anchored4dgs/<scene>.pth` |

Ex4DGS v0.1 archives are downloaded from the
[upstream release](https://github.com/juno181/Ex4DGS/releases/tag/v0.1).
They contain PLY fields, configuration, and cameras. The adapter evaluates the
archive directly, using cubic position interpolation and quaternion slerp. The
archives record 40000 training iterations, 300 frames, keyframe interval ten,
time padding two, variance padding three, resolution divisor two, SH degree
three, clipping bounds four to 300, and projection kernel size 0.1. Place the supplied compact 1K and Anchored models at the paths
above; their public download URLs are unset in the manifest. `--prepare` verifies
available models and lists missing ones.

## Explicit4D checkpoint and construction

The [4DGS-1K method](https://papers.neurips.cc/paper_files/paper/2025/hash/6c39d1a7eecfd98570d74bf7efec1be7-Abstract-Conference.html)
uses the original 4DGS parameterization with contribution pruning. The `.pth`
payload is `(fields, iteration, metadata)`. Its nineteen fields retain spatial
means, spherical harmonics of spatial degree three and temporal degree two,
log scales, left/right quaternion rotations, opacity logits, temporal means, and
temporal scales. Optimizer and accumulator slots are `None` in compact files.
Metadata contains source provenance, retained IDs, sixteen Boolean masks with
keyframes from 0 through 300 at stride twenty, and the fine-tuning budget.

`gspim.training.explicit4d` contains the construction functions:

- `continue_training(scene, until=None, one_k=False)` continues an original 4DGS
  source; `one_k=True` performs the masked fine-tuning recorded in its metadata.
- `prepare_1k(scene, finetune_steps=1500)` accumulates alpha-times-transmittance
  contributions on training cameras, retains the highest-scoring 20%, and builds
  keyframe masks. Temporal and spatial time sums are multiplied, with volume
  exponent 0.1 and the descending 90% reference volume.
- `export_checkpoint(source, destination)` removes optimizer and sampler state
  while preserving inference fields and masks.

Original source files belong in `checkpoints/training/4dgs/<scene>.pth` and continued
sources use `<scene>-trained.pth`. Pruned training checkpoints live in
`checkpoints/training/4dgs1k/`. Exports go to the manifest inference paths. Training
functions write separate training files; the default render always reads compact
inference files. The retained models use these source and final training steps:

| Scene | Source step | Final step |
| --- | ---: | ---: |
| coffee_martini | 20000 | 21500 |
| cook_spinach | 21000 | 22500 |
| flame_steak | 6000 | 7500 |
| sear_steak | 16500 | 19500 |

The Sear source includes continued original 4DGS training before pruning. Masked
fine-tuning uses batches of four, the original learning-rate schedules, frozen
population size, black background, and a loss of 0.8 L1 plus 0.2 SSIM. Source
models follow [original 4DGS](https://github.com/LogosIntelligenceLab/4d-gaussian-splatting)
checkpoint conventions; source provenance is also stored in each compact model.

## Anchored checkpoint and construction

The [Anchored 4DGS algorithm](https://doi.org/10.1145/3757377.3763898) is implemented
with 32-dimensional anchor features and 32-unit hidden layers. Ten bindings per
anchor carry unscaled 4D offsets and base scales. Opacity uses a tanh decoder;
covariance uses feature inputs, sigmoid scale factors, and paired quaternion
rotations; color uses features and view direction with sigmoid RGB output.
Temporal support uses threshold 0.05. Static anchors retain their mean at the
temporal center and apply no temporal decay.

The compact `.pth` dictionary contains `method`, `iteration`, `training_steps`,
`choices`, `model`, and conservative `intervals`, together with source metadata.
The loader checks tensor dimensions, finite values, decoder conventions, and the
recorded training budget.

`gspim.training.anchored` contains `AnchoredModel`, `anchored_inputs(scene)`,
`train_anchored(scene, until=2000)`, and `export_checkpoint(source, destination)`.
Initialization uses the Ex4DGS archive's COLMAP point cloud in its original
coordinate frame. Anchors are quantized at 0.001 spatial and 0.033 temporal units;
initial temporal scale is 1.414. Training uses batches of two, 0.8 L1 plus 0.2 SSIM
and a 0.001 volume penalty. Growth uses spatial levels 16/4/1 with 4D centers,
velocity-aware selection, and accumulated visibility/opacity statistics. The
static classification begins at step 1500. Each supplied model records 2000
updates and its decoder choices.

Training checkpoints live in `checkpoints/training/anchored4dgs/`; exports retain
anchor parameters, decoder weights, static flags, and conservative time intervals
at the manifest inference paths. Training and export are Python functions; preparation
checks or downloads inputs without invoking these functions.

## Pinned sources

| Source | Revision | Use |
| --- | --- | --- |
| Ex4DGS | `7fac64997164fde64732c80261e56adf66ca14df` | Keyframe conventions and released models |
| Original 4DGS | `121e360ce23d6e9ca7734072393414c844dfaea2` | Parameterization, camera math, training utilities and rasterizer |
| fused-ssim | `a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38` | Differentiable training loss |

[Third-party notices](../THIRD_PARTY_NOTICES.md) describe the applicable licenses
and additional algorithm sources.
