#!/usr/bin/env bash
set -euo pipefail

sbt test
sbt run
verilator --lint-only --timing generated/GspimRank.sv
