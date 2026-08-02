# GSPIM Functional Model

This directory contains the executable five-stage mechanism model. `model.py`
defines canonical records and frames, `ppim.py` executes selection programs,
`hcf.py` moves aligned payloads, `dataflow.py` implements S2--S4, `renderer.py`
implements S5, and `pipeline.py` connects them through `runtime.py` and the
ping-pong batch-buffer state in `scheduling.py`. The executable runtime trace
is emitted by `pipeline.py`; there is no second, synthetic scheduling model.

Public behavior is defined by `docs/paper-contract.md`; this package is a
semantic model and never reports hardware performance or PPA.
