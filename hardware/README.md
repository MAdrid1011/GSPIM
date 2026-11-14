# GSPIM hardware

The Chisel source implements the die-level PPIM/HCF tile, bank-local program
and layout storage, compaction workspace, and LPDDR task transactions.
See [Architecture](../docs/architecture.md) and [Runtime](../docs/runtime.md).

Hardware design checks cover FIFO ownership, task transactions, compaction,
and generated SystemVerilog. `GeneratedRtlSpec` exercises a reduced legal
topology. Elaboration supports the default topology; generated RTL, build
products, and waveforms are ignored by Git.

The hardware source is retained as part of the design. Reproduction uses the
single [main entry](../README.md#evaluation) and its PIM scheduling model.
