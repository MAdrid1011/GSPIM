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
stream.

## GPU and memory timeline

`pim_record` runs the shared GPU Dataflow pipeline and records generation,
projection, window adjustment, sorting, color publication, culling, and consumer
tasks. Representation adapters supply explicit or anchored generation fields.
Fixed CUDA publication kernels compile in the external cache. The recorder
checks images and window decisions across parallel and serial GPU execution.

`pim_capture` connects recorded GPU tasks to field publications, input reads,
block completions, and frame completion events. `pim_requests` translates active
selection, stability classification, field gathering, and tile packing into
32-byte physical memory requests. The three representations provide their own
active fields and generation tasks to the same request and scheduling interface.

`pim_memory` advances paired LPDDR5 dies through Ramulator while GPU tasks execute
on the same timeline. `pim_schedule` admits GPU tasks according to available SMs,
registers, shared memory, producer dependencies, and window buffers. Each channel
has a 128-request window; field traffic uses twelve bank queues, and each die has
16 KiB of staging storage. A consumer starts after its inputs are published and
physically complete. Completion drains all GPU tasks, memory requests, and staging
allocations.

Bitmap expansion scans that exceed one staging allocation are divided into ordered
capacity-bounded passes. The PPIM lane retains its prefix between passes, and the
final pass carries the descriptors and output sectors. Every physical bitmap read
and publication dependency remains on the resulting timeline.

Ready consumers prefer the oldest window, then the largest published input
workload within that window. Producer tasks retain their stage ordering.

Independent-frame PIM uses one-frame activity and depth keys. It preserves GPU
frame order while overlapping the next frame's PIM preparation with the current
frame's GPU stages. Serial Dataflow+PIM drains each batch before admitting the next
batch or window. Full GSPIM permits packing, GPU consumption, and the next window's
preparation to overlap according to the same input and buffer dependencies.
