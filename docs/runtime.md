# Runtime Contract

The functional Python runtime uses typed task descriptors to model dependency,
cache-visibility, block-completion, and final-completion semantics. The
Chisel `LpddrTaskAdapter` maps that same public descriptor contract to
standard LPDDR `WRITE` transactions at reserved control addresses and returns
final completion through a reserved-address `READ`. Program writes may target
one bank or broadcast to every PIM bank; layout writes target one bank, so a
model loader emits the corresponding layout write for each participating bank.
`PIM_SELECT` names a PPIM program, source region, die, explicit PIM-bank scope,
and physical block scope; it produces block masks. `PIM_REORG` names a
source/destination region, compaction purpose, bank and block scopes, and a
selection dependency; it drives HCF compaction or binding expansion. This is a
transaction protocol only: it does not model LPDDR timing.

Every descriptor is submitted to the die and PIM banks that own its source
blocks. A completion includes the task ID, die, destination line/range,
overflow state, and final count; HCF exposes each completed block range through
a decoupled transaction separately. S1 and S3 REORG requests identify the
preceding SELECT completion; the tile snapshots that map and program origin so
the next SELECT may overwrite its selection storage while HCF drains the prior
map. Batch REORG instead consumes GPU-provided masks. The Chisel top is one
die-level tile; cross-die dependencies are represented by the functional trace,
not by an instantiated multi-die RTL wrapper.

Before PIM consumes a GPU-written source, the GPU writes it back. Before the
GPU reads a PIM-written destination, it invalidates the region and acquires the
task completion. Violations fail deterministically.

S1 -> S2 -> S3 is `(die, bank, block)`-granular. S1 uses the explicit
`[t_min, t_max)` supplied for that window, and its HCF output assigns a global
Active Index before S2 begins. The block-local GPU `S3_GPU` task performs
camera-space depth transformation and SS reduction before the dependent PPIM
stability SELECT and reorganization. S4 waits for the full window. S5 has two
batch buffers: pack zero precedes render zero; later packing can overlap the
preceding render, and each later render waits for both its pack and the prior
render. The runtime records which render remains live when the later pack
completes. S3's window decision releases next-window S1 without waiting for the
current window's S4/S5; the shared runtime records current S5 render as live
when that next-window S1 completion occurs. No elapsed-time estimate is
produced.
