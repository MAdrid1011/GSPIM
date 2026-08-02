package gspim

import circt.stage.ChiselStage

object Generate extends App {
  ChiselStage.emitSystemVerilogFile(new GspimRank(), Array("--target-dir", "generated"), Array.empty)
}
