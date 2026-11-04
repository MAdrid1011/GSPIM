# Runtime and synchronization

## Task descriptors

The Python runtime models task dependencies, cache visibility, block completion,
and final completion with typed descriptors. `LpddrTaskAdapter` maps descriptors to
LPDDR `WRITE` transactions at reserved control addresses and reads final completion
through a reserved-address `READ`.

Program writes target one bank or broadcast to all PIM banks. Layout writes target
one bank. A model loader installs layouts on every participating bank.
`PIM_SELECT` supplies a program, source region, die, PIM-bank scope, and physical
block scope to produce masks. `PIM_REORG` supplies source and destination regions,
a compaction purpose, bank/block scopes, and a selection dependency to invoke HCF
packing or binding expansion.

Descriptors go to the die and banks owning their source blocks. Completion includes
task ID, die, destination range, overflow state, and final count. HCF reports each
completed block range separately through a decoupled transaction.

## Dependencies and buffers

Active and stability reorganization requests identify their preceding selection
completion. The tile snapshots the masks and originating program while HCF drains
them, allowing the next selection to reuse selection storage. Batch reorganization
consumes GPU-generated masks.

The window pipeline has five stages: active selection (S1), geometry generation
(S2), stability classification (S3), render-order construction (S4), and batched
composition (S5). S1 through S3 proceed at `(die, bank, block)` granularity.
S1 uses the explicit half-open window `[t_min, t_max)`; HCF assigns a global active
index before S2. Block-local `S3_GPU` transforms camera-space depth and reduces
stability before dependent PPIM classification and packing. S4 waits for the whole
window.

S5 alternates two batch buffers. The first pack precedes the first render. Later
packing can overlap the preceding render; each later render waits for its own pack
and the prior render. The next window's S1 depends on the current S3 window decision
and can overlap current S4/S5. Functional traces retain the live-task relationships
at completion events.

## Visibility and completion

Before PIM consumes GPU-written data, the GPU writes back the source region.
Before the GPU reads PIM-written data, it invalidates the destination region and
acquires completion. The runtime validates region ownership and completion
order. Physical block completions permit consumers to start before a multi-block
task's final completion. Workspace releases follow the consuming task's lifetime.

The CUDA renderer uses stream events to order projection, metadata preparation,
and composition. Cross-stream tensor lifetimes are recorded on the consuming
stream. Timing uses frame completion events as described in [running](running.md).
