package gspim

import chisel3._
import chisel3.util._

/**
  * Transaction-level bridge for the paper's standard WRITE/READ task protocol.
  *
  * This module decodes only the reserved control addresses. It deliberately does
  * not model LPDDR timing, scheduling, cache hierarchy, or electrical behavior.
  */
class LpddrTaskAdapter(p: GspimParams) extends Module {
  val io = IO(new Bundle {
    val write = Flipped(Decoupled(new LpddrControlWrite(p)))
    val task = Decoupled(new PimTaskReq(p))
    val programWrite = Decoupled(new PpimProgramWrite(p))
    val layoutWrite = Decoupled(new PpimLayoutWrite(p))
    val completionIn = Flipped(Decoupled(new PimCompletion(p)))
    val readRequest = Input(Bool())
    val readAddress = Input(UInt(32.W))
    val completionRead = Decoupled(new PimCompletion(p))
  })

  val address = io.write.bits.address
  val isProgram = address === LpddrControlAddress.programWrite.U(32.W)
  val isLayout = address === LpddrControlAddress.layoutWrite.U(32.W)
  val isSelect = address === LpddrControlAddress.pimSelect.U(32.W)
  val isReorg = address === LpddrControlAddress.pimReorg.U(32.W)
  val isTask = isSelect || isReorg

  io.task.valid := io.write.valid && isTask
  io.task.bits := io.write.bits.task
  io.programWrite.valid := io.write.valid && isProgram
  io.programWrite.bits := io.write.bits.program
  io.layoutWrite.valid := io.write.valid && isLayout
  io.layoutWrite.bits := io.write.bits.layout
  io.write.ready := MuxCase(false.B, Seq(
    isProgram -> io.programWrite.ready,
    isLayout -> io.layoutWrite.ready,
    isTask -> io.task.ready,
  ))
  when(io.write.fire && isSelect) {
    assert(io.write.bits.task.kind === PimTaskKind.select,
      "PIM_SELECT address requires a PIM_SELECT descriptor")
  }
  when(io.write.fire && isReorg) {
    assert(io.write.bits.task.kind === PimTaskKind.reorg,
      "PIM_REORG address requires a PIM_REORG descriptor")
  }

  val completionValid = RegInit(false.B)
  val completionBits = Reg(new PimCompletion(p))
  io.completionIn.ready := !completionValid
  when(io.completionIn.fire) {
    completionValid := true.B
    completionBits := io.completionIn.bits
  }
  val completionReadAddress = io.readAddress === LpddrControlAddress.completionRead.U(32.W)
  io.completionRead.valid := completionValid && io.readRequest && completionReadAddress
  io.completionRead.bits := completionBits
  when(io.completionRead.fire) {
    completionValid := false.B
  }
}
