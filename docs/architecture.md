# PPIM and HCF architecture

## Organization

`ArchitectureConfig` and `GspimParams` describe eight packages, four dies per
package, twelve PIM banks and four reserved banks per die. `GspimRank` instantiates
one die-level PPIM/HCF tile. Each tile combines bank-local program and layout
storage, PPIM arithmetic, selection masks, the HCF workspace, and task control.

## Polymorphic PIM

PPIM reads temporal decision fields from aligned record blocks and emits one bit
per record. Each PIM bank owns a program buffer and a load-time field-layout table.
The table maps temporal mean, rotation base, scale base, static flag, interval
endpoints, and stability score to columns in the bank-local row record.

`PIM_SELECT` names a program-buffer entry. Model loading records the representation
kind beside its instructions; the selected bank fetches both. The shared arithmetic
executes `ROT_SLICE`, `SCALE`, `DOT`, `WINDOW_DIST`, and `CMP_LE` for temporal
activity, two comparisons plus `AND` for interval overlap, or a comparison for
depth stability. Signed 26-bit operands use 52-bit products before nearest-value
two's-complement writeback. Temporal selection retains a one-LSB boundary record;
stability selection classifies the boundary as unstable.

`MaskControlledColumnPath` forwards GPU accesses and applies PPIM mask bits to
compaction accesses. A selected record is represented by a 32-bit payload
transaction. `BankCompactor` packs selected payload words. Physical record locations
use `(die, bank, block, slot)` so each decision controls the matching payload.
Transactions carry source and destination lines for each task block.

## Hierarchical Compaction Fabric

L1 filters an individual bank block. L2 consumes ActiveMap entries in FIFO order,
maintains per-bank prefixes, reserves a contiguous live range from one shared
per-die workspace, generates addresses, gathers payloads, and writes packed records.
It reports each completed block range and a final output count. A range stays
allocated until its consumer releases it.

The reserved-bank workspace serves active, stable, unstable, and two in-flight
batch allocations. Active selection packs payloads; stability selection packs
sorting metadata; batch selection packs shared attributes and frame-local data.
For Anchored layouts, binding counts reserve and emit every declared bound payload
from validated binding-table responses.

Insufficient workspace emits a `GPU_REORG` transaction with the original block,
mask, payloads, and binding metadata. Same-bank GPU requests have priority; requests
to other banks can proceed alongside FIFO dequeue. `gspim/hcf.py` provides the
functional counterpart to the RTL transactions.

## Design verification

Python tests compare independent selection, sorting, batching, and rendering
algorithms directly. HCF tests cover allocation lifetimes, payload movement,
binding expansion, overflow, and bank arbitration. RTL tests execute all four
microprograms, shuffled field layouts, multi-block completion, fallback, and task
control. [Setup](setup.md) describes how to run the Python and RTL checks.
