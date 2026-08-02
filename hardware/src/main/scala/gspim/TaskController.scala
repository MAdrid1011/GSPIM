package gspim

import chisel3._
import chisel3.util._

/** Doorbell controller that preserves task program/scope and real execution count. */
class TaskController(p: GspimParams) extends Module {
  val io = IO(new Bundle {
    val request = Flipped(Decoupled(new PimTaskReq(p)))
    val sourceWritebackDone = Input(Bool())
    val dependencyComplete = Input(Bool())
    val executionDone = Input(Bool())
    val executionOutputCount = Input(UInt(16.W))
    val executionDestinationStart = Input(UInt(16.W))
    val executionOverflow = Input(Bool())
    val completion = Decoupled(new PimCompletion(p))
    val destinationWritten = Output(Bool())
    val activeProgram = Output(UInt(log2Ceil(p.programSlots).W))
    val activeTaskId = Output(UInt(16.W))
    val activeKind = Output(UInt(1.W))
    val activeDie = Output(UInt(log2Ceil(p.diesPerPackage * p.packages).W))
    val activeBank = Output(UInt(log2Ceil(p.pimBanksPerDie).W))
    val activeBankCount = Output(UInt(log2Ceil(p.pimBanksPerDie + 1).W))
    val activeSourceLine = Output(UInt(32.W))
    val activeDestinationLine = Output(UInt(32.W))
    val activeBlockStart = Output(UInt(16.W))
    val activeBlockCount = Output(UInt(16.W))
    val activePurpose = Output(UInt(2.W))
    val activeDependency = Output(UInt(16.W))
    val active = Output(Bool())
  })
  val busy = RegInit(false.B)
  val latched = Reg(new PimTaskReq(p))
  val resultReady = RegInit(false.B)
  val resultCount = RegInit(0.U(16.W))
  val resultStart = RegInit(0.U(16.W))
  val resultOverflow = RegInit(false.B)
  val dependencySatisfied = io.request.bits.dependency === 0.U || io.dependencyComplete
  io.request.ready := !busy && io.sourceWritebackDone && dependencySatisfied
  when(io.request.fire) {
    assert(io.request.bits.bankCount =/= 0.U, "PIM task bank scope must be nonempty")
    assert(io.request.bits.bank +& io.request.bits.bankCount <= p.pimBanksPerDie.U, "PIM task bank scope exceeds die")
    assert(io.request.bits.blockCount =/= 0.U &&
      io.request.bits.blockStart +& io.request.bits.blockCount <= p.maxBlocksPerTask.U,
      "PIM task block scope is invalid")
    latched := io.request.bits
    busy := true.B
  }
  io.activeProgram := latched.program
  io.activeTaskId := latched.taskId
  io.activeKind := latched.kind
  io.activeDie := latched.die
  io.activeBank := latched.bank
  io.activeBankCount := latched.bankCount
  io.activeSourceLine := latched.sourceLine
  io.activeDestinationLine := latched.destinationLine
  io.activeBlockStart := latched.blockStart
  io.activeBlockCount := latched.blockCount
  io.activePurpose := latched.purpose
  io.activeDependency := latched.dependency
  io.active := busy
  when(busy && io.executionDone && !resultReady) {
    resultReady := true.B
    resultCount := io.executionOutputCount
    resultStart := io.executionDestinationStart
    resultOverflow := io.executionOverflow
  }
  io.completion.valid := resultReady
  io.completion.bits.taskId := latched.taskId
  io.completion.bits.die := latched.die
  io.completion.bits.outputCount := resultCount
  io.completion.bits.destinationLine := latched.destinationLine
  io.completion.bits.destinationStart := resultStart
  io.completion.bits.overflow := resultOverflow
  io.destinationWritten := resultReady
  when(io.completion.fire) {
    busy := false.B
    resultReady := false.B
  }
}
