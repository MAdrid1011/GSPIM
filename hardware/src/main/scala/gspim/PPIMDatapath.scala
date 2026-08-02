package gspim

import chisel3._
import chisel3.util._

/** Executes one bank-local microprogram with explicit PC and register state. */
class PPIMProgramExecutor(p: GspimParams) extends Module {
  val io = IO(new Bundle {
    val start = Input(Bool())
    val instructions = Input(Vec(p.microProgramLength, UInt(p.microInstructionWidth.W)))
    val temporalMean = Input(SInt(p.operandWidth.W))
    val temporalRotation = Input(Vec(4, Vec(4, SInt(p.operandWidth.W))))
    val temporalScale = Input(Vec(4, SInt(p.operandWidth.W)))
    val windowStart = Input(SInt(p.operandWidth.W))
    val windowEnd = Input(SInt(p.operandWidth.W))
    val isStatic = Input(Bool())
    val allowStatic = Input(Bool())
    val supportStart = Input(SInt(p.operandWidth.W))
    val supportEnd = Input(SInt(p.operandWidth.W))
    val score = Input(SInt(p.operandWidth.W))
    val tau = Input(SInt(p.operandWidth.W))
    val mask = Output(Bool())
    val sigmaTt = Output(SInt(p.operandWidth.W))
    val done = Output(Bool())
  })

  def roundedProduct(a: SInt, b: SInt): SInt = {
    val product = a * b
    val absolute = Mux(product < 0.S, -product, product)
    val roundedAbsolute = (absolute + (1.S << (p.fractionalBits - 1))) >> p.fractionalBits
    Mux(product < 0.S, -roundedAbsolute, roundedAbsolute).asSInt
  }

  val scaledRow = Wire(Vec(4, SInt(p.operandWidth.W)))
  for (index <- 0 until 4) {
    scaledRow(index) := roundedProduct(io.temporalRotation(3)(index), io.temporalScale(index))
  }
  val dotTerms = (0 until 4).map(index => roundedProduct(scaledRow(index), scaledRow(index)))
  val sigma = dotTerms.reduce(_ +& _).asSInt
  val distance = Wire(SInt(p.operandWidth.W))
  distance := Mux(io.temporalMean < io.windowStart, io.windowStart - io.temporalMean, Mux(io.temporalMean > io.windowEnd, io.temporalMean - io.windowEnd, 0.S))
  val ln20 = 196328.S(p.operandWidth.W)
  val lsb = 1.S(p.operandWidth.W)
  val pc = RegInit(0.U(log2Ceil(p.microProgramLength).W))
  val busy = RegInit(false.B)
  val done = RegInit(false.B)
  val mask = RegInit(false.B)
  val sigmaReg = RegInit(0.S(p.operandWidth.W))
  val distanceReg = RegInit(0.S(p.operandWidth.W))
  val rotSeen = RegInit(false.B)
  val scaleSeen = RegInit(false.B)
  val dotSeen = RegInit(false.B)
  val windowSeen = RegInit(false.B)
  val loadSeen = RegInit(false.B)
  val compareCount = RegInit(0.U(2.W))
  val left = RegInit(false.B)
  val right = RegInit(false.B)

  done := false.B
  when(io.start && !busy) {
    pc := 0.U
    busy := true.B
    mask := false.B
    sigmaReg := 0.S
    distanceReg := 0.S
    rotSeen := false.B
    scaleSeen := false.B
    dotSeen := false.B
    windowSeen := false.B
    loadSeen := false.B
    compareCount := 0.U
    left := false.B
    right := false.B
  }.elsewhen(busy) {
    val instruction = io.instructions(pc)(3, 0)
    switch(instruction) {
      is(PpimMicroOp.rotSlice) {
        rotSeen := true.B
      }
      is(PpimMicroOp.scale) {
        scaleSeen := rotSeen
      }
      is(PpimMicroOp.dot) {
        when(scaleSeen) {
          sigmaReg := sigma
          dotSeen := true.B
        }
      }
      is(PpimMicroOp.windowDist) {
        distanceReg := distance
        windowSeen := true.B
      }
      is(PpimMicroOp.compareLe) {
        val halfDistanceSquared = (roundedProduct(distanceReg, distanceReg) >> 1).asSInt
        val temporalRight = roundedProduct(sigmaReg, ln20)
        mask := dotSeen && windowSeen && halfDistanceSquared <= temporalRight + lsb
      }
      is(PpimMicroOp.load) {
        loadSeen := true.B
        compareCount := 0.U
        left := false.B
        right := false.B
      }
      is(PpimMicroOp.compareLt) {
        when(loadSeen && compareCount === 0.U) {
          // Support endpoints are closed while rendering windows are
          // half-open. Do not add a threshold margin to this set intersection.
          left := io.supportStart < io.windowEnd
          compareCount := 1.U
        }.elsewhen(loadSeen && compareCount === 1.U) {
          right := io.supportEnd >= io.windowStart
          compareCount := 2.U
        }
      }
      is(PpimMicroOp.and) {
        mask := (loadSeen && compareCount === 2.U && left && right) || (io.allowStatic && io.isStatic)
      }
      is(PpimMicroOp.compareGt) {
        mask := io.score > io.tau + lsb
      }
    }
    when(pc === (p.microProgramLength - 1).U) {
      busy := false.B
      done := true.B
    }.otherwise {
      pc := pc + 1.U
    }
  }

  io.mask := mask
  io.sigmaTt := sigmaReg
  io.done := done
}

/** Models the mask-controlled column-selection path after PPIM execution. */
class MaskControlledColumnPath extends Module {
  val io = IO(new Bundle {
    val regularGpuAccess = Input(Bool())
    val ppimMask = Input(Bool())
    val inputValid = Input(Bool())
    val outputValid = Output(Bool())
  })
  io.outputValid := io.inputValid && (io.regularGpuAccess || io.ppimMask)
}
