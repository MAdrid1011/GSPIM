# PPIM and HCF Architecture

The paper's evaluated topology is eight packages, four dies per package,
twelve PIM banks and four reserved banks per die. `ArchitectureConfig` and
`GspimParams` retain those counts as configuration context. The instantiated
`GspimRank` RTL is one die-level PPIM/HCF tile, not a complete 32-die package
or rank wrapper, and does not assign timing, area, or power to those counts.

## Polymorphic PIM

PPIM reads only temporal decision fields from an aligned record block and emits
one decision bit per record. Each PIM bank owns a program buffer and a
load-time field-layout table. The table maps temporal mean, rotation base,
scale base, static flag, interval endpoints, and stability score to columns in
the bank-local row record; it is not a fixed union of representation ports.
`PIM_SELECT` names a program-buffer entry; model load records that entry's
representation kind beside its instructions, and the selected bank fetches
both. Its shared arithmetic executes the instruction sequence. TEMP uses `ROT_SLICE`, `SCALE`, `DOT`, `WINDOW_DIST`, and
`CMP_LE`; overlap uses two comparisons plus `AND`; depth stability uses a
comparison. Arithmetic uses signed 26-bit values; products use 52-bit precision
before nearest-value two's-complement register-file writeback. S1 retains a
one-LSB boundary record and S3 classifies it as unstable.

The mask-controlled access contract forwards every record for a GPU access and
only PPIM-selected records for compaction. RTL represents one selected record
as a 32-bit payload transaction: `MaskControlledColumnPath` carries the valid
decision and `BankCompactor` selects and packs the corresponding payload word.
It is not a transistor-level LPDDR column-decoder model. A model block records
its `(die, bank, block, slot)` location, temporal fields, payload, and
attribute mapping, so a PPIM decision always controls the matching payload.
The transaction interface carries the source and destination line for each
physical task block.

## Hierarchical Compaction Fabric

L1 filters an individual bank block. L2 accepts the PPIM-produced ActiveMap
entries in FIFO order, exposes per-bank prefix accounting, reserves a contiguous
live range from one shared per-die workspace using the complete task count,
generates source/destination addresses, gathers payloads, and writes packed
records. A range remains allocated until its consumer releases it; insufficient
shared space emits a `GPU_REORG` fallback transaction containing the original
physical block, mask, payloads, and binding metadata. It reports a block range
immediately and a final output count when the task completes. The GPU wins only
a same-bank conflict; requests for another bank do not stall FIFO dequeue.

The reserved-bank workspace is shared by `active`, `stable`, `unstable`, and
two in-flight `batch` allocations rather than statically partitioned by
purpose. S1 writes active payloads, S3 writes separate sorting metadata, and
S4 selections pack shared attributes and frame-local data. For Anchored 4DGS,
an active anchor's binding count reserves and emits every declared bound
payload from a validated binding-table response; a missing response asserts in
RTL rather than becoming a zero-filled binding. Capacity overflow is visible and
triggers GPU reorganization; no record is silently discarded.

`hardware/` implements this transaction contract and `gspim/hcf.py` is its
functional counterpart. The RTL tests execute all four microprograms through a
program counter, use a shuffled static field layout, verify bound-payload
expansion, prove two-block completion, emit a fallback transaction, and check
same-bank GPU priority.
