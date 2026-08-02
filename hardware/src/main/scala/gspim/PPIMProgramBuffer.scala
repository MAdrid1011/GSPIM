package gspim

import chisel3._
import chisel3.util.log2Ceil

/** Static-layout descriptors select PPIM decision fields from a row-buffer record. */
object PpimLayoutField {
  val temporalMean = 0
  val rotationBase = 1
  val scaleBase = 2
  val staticFlag = 3
  val supportStart = 4
  val supportEnd = 5
  val score = 6
  val count = 7
  val defaults: Seq[Int] = Seq(0, 1, 17, 21, 22, 23, 24)
}

/** One bank-local static data-layout table written when a model is loaded. */
class PPIMLayoutBuffer(p: GspimParams) extends Module {
  private val fieldWidth = log2Ceil(p.rowFieldSlots)
  private val descriptorWidth = log2Ceil(PpimLayoutField.count)
  val io = IO(new Bundle {
    val writeEnable = Input(Bool())
    val writeField = Input(UInt(descriptorWidth.W))
    val writeIndex = Input(UInt(fieldWidth.W))
    val fields = Output(Vec(PpimLayoutField.count, UInt(fieldWidth.W)))
  })

  val storage = RegInit(VecInit(PpimLayoutField.defaults.map(_.U(fieldWidth.W))))
  when(io.writeEnable) {
    assert(io.writeField < PpimLayoutField.count.U, "PPIM layout descriptor field is invalid")
    assert(io.writeIndex < p.rowFieldSlots.U, "PPIM layout points outside the row record")
    when(io.writeField === PpimLayoutField.rotationBase.U) {
      assert(io.writeIndex <= (p.rowFieldSlots - 16).U,
        "PPIM rotation base must leave room for a 4x4 matrix")
    }
    when(io.writeField === PpimLayoutField.scaleBase.U) {
      assert(io.writeIndex <= (p.rowFieldSlots - 4).U,
        "PPIM scale base must leave room for four scale values")
    }
    storage(io.writeField) := io.writeIndex
  }
  io.fields := storage
}

/** Stores host-loaded PPIM microinstructions fetched from a task program entry. */
class PPIMProgramBuffer(p: GspimParams) extends Module {
  private val addressWidth = log2Ceil(p.programSlots * p.microProgramLength)
  private val programWidth = log2Ceil(p.programSlots)
  val io = IO(new Bundle {
    val writeEnable = Input(Bool())
    val writeAddress = Input(UInt(addressWidth.W))
    val writeKind = Input(UInt(2.W))
    val writeInstruction = Input(UInt(p.microInstructionWidth.W))
    val readProgram = Input(UInt(programWidth.W))
    val readInstruction = Output(Vec(p.microProgramLength, UInt(p.microInstructionWidth.W)))
    val readKind = Output(UInt(2.W))
  })

  val storage = RegInit(VecInit(Seq.fill(p.programSlots * p.microProgramLength)(PpimMicroOp.nop.pad(p.microInstructionWidth))))
  val programKinds = RegInit(VecInit((0 until p.programSlots).map(_.U(2.W))))
  val writeProgram = (io.writeAddress / p.microProgramLength.U)(programWidth - 1, 0)
  when(io.writeEnable) {
    assert(io.writeAddress < (p.programSlots * p.microProgramLength).U,
      "PPIM program write address is outside the program buffer")
    when(io.writeAddress < (p.programSlots * p.microProgramLength).U) {
      storage(io.writeAddress) := io.writeInstruction
      programKinds(writeProgram) := io.writeKind
    }
  }
  for (index <- 0 until p.microProgramLength) {
    io.readInstruction(index) := storage(io.readProgram * p.microProgramLength.U + index.U)
  }
  io.readKind := programKinds(io.readProgram)
}
