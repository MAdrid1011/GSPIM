# GSPIM Chisel Transaction Model

This project implements one die-level PPIM/HCF tile and its paper-visible task
transaction contract. It is RTL behavior evidence, not a complete 32-die rank,
DRAM timing, power, or PPA model.

Reserved LPDDR control-address `WRITE`s load programs/layouts and submit
`PIM_SELECT`/`PIM_REORG`; a reserved-address `READ` receives the final task
completion. `TaskController` latches a program-buffer entry rather than a
hard-coded operation. Each PIM bank contains a `PPIMProgramBuffer` and a load-time
field-layout table which maps semantic decision fields to a common row-buffer
record. `PPIMProgramExecutor` advances through its TEMP, keyframe, anchor, or
depth-stability instructions using signed 26-bit fixed point. The Q16 binary
point is an artifact parameter rather than a paper parameter. A SELECT
latches one mask bit per record for every physical task block; an S1 or S3
REORG snapshots those exact maps and their source program into the ActiveMap
FIFO. `HcfGatherWriter` reserves and releases one
shared live workspace, expands anchor binding payloads, reports block/final
completion, emits original block payloads on overflow for `GPU_REORG`, and
only stalls the connected gather when a GPU request targets the same bank.

Run the complete reproducible hardware check:

```bash
./verify-rtl.sh
```

It runs ChiselTest, including `GeneratedRtlSpec` on the Verilator backend,
generates SystemVerilog into ignored `generated/`, and runs Verilator lint on
the generated rank. `GeneratedRtlSpec` uses a reduced, legal topology to test
the complete transaction state machine in a practical Verilator run; `sbt run`
and the lint step emit the default paper-context topology. `sbt test` is the
unit and generated-RTL behavior target.
