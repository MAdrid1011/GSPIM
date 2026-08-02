# Paper Contract

This repository is a functional and RTL transaction artifact for the GSPIM
mechanisms. It has one source of truth: the five-stage dataflow, PPIM/HCF
architecture, task interface, hybrid scheduling, and workload categories
described by the GSPIM paper. It does not reproduce the paper's platform
measurements.

## Canonical Dataflow

S1 runs a representation-selected PPIM program over temporal decision fields
for an explicit half-open interval `[t_min, t_max)`. The public pipeline API
requires both boundaries; it never infers `t_max` from the last frame timestamp.
HCF compacts the selected aligned payloads into an Active Buffer. S2
conditions explicit 4D Gaussians on each timestamp and interpolates every
modeled Ex4DGS keyframe field to generate a 3D mean and covariance.
S3 transforms means to camera space, computes the paper's normalized depth
stability score, and compacts separate stable and unstable metadata streams.
S4 uses fixed-width radix keys to sort the stable set once by window minimum
depth and the unstable set per frame, then uses a partitioned merge-path to
form render-order lists (ROLs) and chooses shared
stable batch boundaries. S5 reads each batch in ROL order, transforms the full
3D covariance into camera space, applies the projection Jacobian to form the
2D covariance, assigns tiles, and forward-composites while retaining each
frame's color and transmittance state across batches.

The fixture renderer is a deterministic 3DGS semantic model. Its camera,
tile size, image size, and storage granularity are artifact configuration, not
claims about an evaluated platform.

## PPIM Programs

`TEMP_ACTIVITY` is the explicit-4D S1 program. For each temporal distribution
it executes `ROT_SLICE`, `SCALE`, `DOT` to obtain `Sigma_tt`, `WINDOW_DIST`,
and `CMP_LE(0.5*d_t^2, Sigma_tt*ln(20))`. A decision within one output LSB
retains the record. The paper fixes the signed 26-bit operand, 52-bit product,
and nearest writeback behavior. The artifact's Q16 binary point is a declared
implementation parameter for executable tests, not a paper parameter; it
compares raw one-LSB margins under that parameter.

`KEYFRAME_RANGE` selects a static Ex4DGS record or a dynamic record whose
normalized keyframe support intersects the half-open temporal window. The
adapter represents support endpoints as closed: a support starting at `t_max`
is not selected, while one ending at `t_min` is selected. `ANCHOR_OVERLAP`
uses the same adapter endpoint convention, then HCF expands its declared
binding range. `DEPTH_STABLE` compares the GPU-produced score to
`tau_s`; a decision within one LSB is unstable. These are PPIM programs, not
host-side shortcuts. Before S1 begins, the Ex4DGS adapter also verifies that
every selected dynamic record brackets every requested S2 timestamp. Window
intersection alone cannot admit a record that would leave a later S2 frame
undefined. The Anchored adapter similarly verifies every selected binding has
explicit representation-generated geometry for every requested S2 timestamp.

## HCF and Runtime

Every selected record has an aligned `(die, bank, block, slot)` location,
decision field, and payload. A model load writes both the local field-layout
table and microprogram into every participating PIM bank. HCF's L1 path filters fields by a PPIM mask; its
L2 path queues block masks, reserves a contiguous live range from one shared
workspace, gathers payloads, and reports block and final completions. Active,
stable, unstable, and batch ranges coexist only while live and are explicitly
released by their consumer; overflow is reported and processed by an explicit
GPU fallback that retains the original physical block plus its rejected tail.
Anchor binding reads carry a valid response for every declared payload; missing
responses fail instead of synthesizing default data. S1 assigns each retained Gaussian one global Active Index in
physical HCF order, and S4 rejects duplicate or non-contiguous indices.

The only host tasks are `PIM_SELECT` and `PIM_REORG`. The functional runtime
uses typed descriptors; the Chisel `LpddrTaskAdapter` maps each descriptor and
model-load write to a standard LPDDR `WRITE` at a reserved control address, and
a reserved-address `READ` obtains its final completion. Program kind is
model-load metadata rather than an assumed program-slot number. A SELECT task
identifies a program-buffer entry, source/destination regions, die, PIM-bank
scope, physical block scope, purpose, and dependencies. S1/S3 REORG snapshots
the completed SELECT map and its program origin; batch REORG uses GPU masks.
The runtime requires GPU writeback before a PIM read and invalidation plus a
completion event before GPU consumption. It records dependencies and the set
of tasks live at each completion; it does not estimate time.

## Hybrid Scheduling

S1, S2, and S3 advance per completed physical `(die, bank, block)` entry. For
each S2 output, a GPU `S3_GPU` task transforms centers to camera space and
reduces the depth sequence before PPIM receives the stability score. S4 waits
for the complete window. S5 packs batch zero before render zero, then
prepares batch `k + 1` while the GPU renders batch `k`; render `k + 1` waits
for both. Once S3 chooses the next window width, next-window S1 selection is
submitted against that completed decision while current-window S5 render zero
remains live. The runtime records this live overlap, not only a trace edge.

## Evidence

`docs/traceability.md` maps every contract item to source and a test.
`reference_model/` contains independent Explicit4D, Ex4DGS, and Anchored
selection/dataflow/rendering oracles. Fixed fixtures compare explicit window
bounds, masks, geometry, ROLs, batches, and output image hashes; targeted
functional and RTL tests separately check HCF payload movement, task
dependencies, live ranges, and overlap.
The generated Chisel top is one die-level PPIM/HCF tile. The paper's 8-package,
4-die topology remains parameter context rather than an instantiated full-rank
RTL wrapper. AGX Orin, datasets, checkpoint quality, timing, energy, Ramulator,
and PPA are outside this artifact.
