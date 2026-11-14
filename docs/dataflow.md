# GPU dataflow

## Geometry and projection

`CUDAPipeline` compares independent-frame execution with temporal window execution.
Each call selects active inputs, generates geometry, projects Gaussians, constructs
render order, and composites pixels. 4DGS-1K selection unions the masks needed
by the requested frames. Ex4DGS interpolates dynamic keyframes and retains static
attributes. Its window selection derives conservative temporal support from each
record's opacity amplitude, envelope centers, and variances. Selected dynamic IDs
retain their original interpolation keys, and generation fuses camera depth and
window statistics. Static records remain selected. Anchored selection tests
conservative anchor intervals before decoding binding positions, covariances,
opacities, and colors. 4DGS-1K and Anchored use their selection and decoding
functions for both independent-frame and window execution.

The independent-frame path generates and projects one frame at a time, then uses
gsplat tile sorting and rasterization. The Dataflow path shares generation and
projection across a window. Anchored windows compact decoded bindings whose opacity
can reach 1/255 in at least one requested frame before projection. Spherical
harmonic evaluation can be deferred until visible tile work is known.

The Ex4DGS independent-frame path reconstructs activated attributes and evaluates
spherical harmonics for each frame. Its Dataflow path reuses static attributes
and shares dynamic selection and generation within the window. Both paths read
the same checkpoint and use the same camera and compositing parameters.

## Windows, stability, and order

Windows contain two through five frames and begin at five frames. The next size
changes by one according to the window stability classification, bounded to that
range. Sequence partitioning adjusts a tail that would otherwise contain one frame.
The classification uses camera-space depth normalized by finite stability bounds.
For a primitive with depths `z[0..W-1]`, its score is
`exp(-sum(abs(z[i+1]-z[i])) / ((far-near)*(W-1)))`. The stable threshold is 0.98;
window adjustment increases the size when the mean score is at least 0.98 and
decreases it otherwise.

Stable primitives sort once by minimum window depth. Unstable primitives sort by
each frame's depth. CUB radix sorting and merge-path construction combine those
sequences, retaining primitive ID ordering for ties. A fully stable window can
share tile metadata across frames.

Visible depth crossing detection checks the shared stable order. Crossings split
composition into spans of at most three frames with mean-depth keys and locally
repaired order. The original ready-window submission stays intact. The second
span's metadata preparation can overlap the first span's composition; span
composition drains serially. Anchored windows with at most 1% unstable bindings
can embed frame-owned unstable occurrences into a shared stable sequence while
retaining each frame's render order.

## Batches and composition

Batch planning obtains the GPU's L2 size and reserves space for resident tile
state. It accounts for packed attributes, descriptors, render-order IDs, tile IDs,
and per-frame state. Greedy boundaries keep each estimated batch footprint within
the remaining budget. Shared-order paths reuse culled tile records after batch
assignment. Frame-owned duplicates are charged separately.

Composition processes batches in order, retaining accumulated color and
transmittance for each pixel. Tile jobs with shared frame visibility use masked
records; the two-pixel compositor factors the common horizontal contribution in
FP32. Alpha cutoff is 1/255, alpha cap is 0.999, and transmittance early stopping is
1e-4. All paths retain the workload's camera, clipping bounds, and background.

The CPU pipeline and independent references exercise selection, stable/unstable
ordering, batch accounting, and compositing on small design inputs. PPIM/HCF
organization is described in [architecture](architecture.md), and the task-level
schedule is described in [runtime](runtime.md).
