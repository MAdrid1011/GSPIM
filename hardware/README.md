# GSPIM hardware

This sbt project implements the die-level PPIM/HCF tile, bank-local program and
layout storage, compaction workspace, and LPDDR task transactions. See
[architecture](../docs/architecture.md) and [runtime](../docs/runtime.md).

With JDK 17, sbt, and Verilator installed, run from this directory:

```bash
./verify-rtl.sh
```

The command runs ChiselTest, including generated RTL on the Verilator backend,
emits SystemVerilog into the ignored `generated/` directory, and lints it.
`GeneratedRtlSpec` exercises a reduced legal topology; `sbt run` emits the default
topology. Build products and waveforms are ignored by Git.
