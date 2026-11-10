# Workloads

## Representations and scenes

The workload manifest is [`integrations/manifest.py`](../integrations/manifest.py).
It defines all eighteen combinations, their default paths, SHA256 values, method
sources, upstream revisions, and model parameters. Scene names are
`coffee_martini`, `cook_spinach`, `flame_steak`, `sear_steak`, `dance3`, and
`actor1_4`.

| Workload identifier | Representation | Checkpoint path |
| --- | --- | --- |
| `explicit4d` | 4DGS-1K: explicit 4D Gaussians and contribution masks | `checkpoints/4dgs1k/<scene>.pth` |
| `ex4dgs` | Static Gaussians and interpolated dynamic keyframes | `checkpoints/ex4dgs/<scene>.zip` |
| `anchored` | 4D anchors, ten bindings each, and neural decoders | `checkpoints/anchored4dgs/<scene>.pth` |

The GSPIM checkpoint bundle in the project
[Releases](https://github.com/MAdrid1011/GSPIM/releases) contains the selected
checkpoint for every row. Extract the bundle at the repository root so that it
produces the paths above. The manifest records each file's SHA256; `--prepare`
verifies present files and lists every missing path.

The Ex4DGS archives contain PLY fields, configuration, and cameras. The adapter
evaluates each archive directly, using cubic position interpolation and quaternion
slerp. The four Neural 3D Video checkpoints record 40000 training iterations, 300
frames, keyframe interval ten, time padding two, variance padding three, resolution
divisor two, SH degree three, clipping bounds four to 300, and projection kernel
size 0.1.

Dance3 and Actor1_4 use the official Mobile-Stage and ENeRF-Outdoor camera
packages. Their access-controlled source packages are converted to the same
`poses_bounds.npy`, `images/`, and `cam00/` layout before being placed under
`datasets/n3v/<scene>/`.

The supplied Dance3 and Actor1_4 models are constructed from native 4D sources
trained on their calibrated training cameras. Their Ex4DGS keyframe models use
2837 and 2720 fitting updates, resolution divisor one, near/far planes 0.2/300,
and a black background. Dance3 covers 300 frames and Actor1_4 covers 200. The manifest's
`parameters_for(scene)` combines each representation's parameters with the
values recorded in the selected scene checkpoint.

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
| dance3 | 8537 | 10507 |
| actor1_4 | 7495 | 10760 |

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
`train_anchored(scene, until=2000)`,
`train_anchored_from_source(scene, source, until=None)`, and
`export_checkpoint(source, destination)`.
The supplied models initialize binding positions, scales, color features, and
paired rotations from the matching compact explicit 4D source. Ten spatially
adjacent primitives form each anchor. Training uses synchronized two-camera
batches, shared time-conditioned geometry, a 0.8 L1 plus 0.2 SSIM image loss,
and a 0.001 volume penalty. Decoder inputs and outputs are initialized from a
32-dimensional low-rank fit of the bound source fields before image training.
The manifest records the source iteration and completed image-training updates
for every scene.

Training checkpoints live in `checkpoints/training/anchored4dgs/`; exports retain
anchor parameters, decoder weights, static flags, and conservative time intervals
at the manifest inference paths. The rendering path reads these compact exports.

## Pinned sources

| Source | Revision | Use |
| --- | --- | --- |
| Ex4DGS | `7fac64997164fde64732c80261e56adf66ca14df` | Keyframe conventions and released models |
| Original 4DGS | `121e360ce23d6e9ca7734072393414c844dfaea2` | Parameterization, camera math, training utilities and rasterizer |
| fused-ssim | `a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38` | Differentiable training loss |

[Third-party notices](../THIRD_PARTY_NOTICES.md) describe the applicable licenses
and additional algorithm sources.
