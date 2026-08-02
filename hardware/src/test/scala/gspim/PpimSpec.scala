package gspim

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class PpimSpec extends AnyFlatSpec with ChiselScalatestTester with Matchers {
  behavior of "PPIM programs"

  private def execute(dut: PPIMProgramExecutor): Unit = {
    dut.io.start.poke(true.B)
    dut.clock.step()
    dut.io.start.poke(false.B)
    dut.clock.step(5)
    dut.io.done.expect(true.B)
  }

  it should "execute TEMP, keyframe, anchor, and stability instructions through PC state" in {
    test(new PPIMProgramExecutor(GspimParams())) { dut =>
      dut.io.instructions.foreach(_.poke(PpimMicroOp.nop))
      dut.io.temporalMean.poke(65536.S)
      dut.io.windowStart.poke(0.S)
      dut.io.windowEnd.poke((4 * 65536).S)
      for (row <- 0 until 4; column <- 0 until 4) dut.io.temporalRotation(row)(column).poke(0.S)
      dut.io.temporalRotation(3)(0).poke(65536.S)
      dut.io.temporalScale.foreach(_.poke(0.S))
      dut.io.temporalScale(0).poke(32768.S)
      dut.io.supportStart.poke(0.S)
      dut.io.supportEnd.poke(1.S)
      dut.io.isStatic.poke(false.B)
      dut.io.allowStatic.poke(false.B)
      dut.io.score.poke(0.S)
      dut.io.tau.poke(0.S)

      dut.io.instructions(0).poke(PpimMicroOp.rotSlice)
      dut.io.instructions(1).poke(PpimMicroOp.scale)
      dut.io.instructions(2).poke(PpimMicroOp.dot)
      dut.io.instructions(3).poke(PpimMicroOp.windowDist)
      dut.io.instructions(4).poke(PpimMicroOp.compareLe)
      execute(dut)
      dut.io.mask.expect(true.B)
      dut.io.sigmaTt.expect(16384.S)

      dut.io.instructions.foreach(_.poke(PpimMicroOp.nop))
      dut.io.instructions(0).poke(PpimMicroOp.load)
      dut.io.instructions(1).poke(PpimMicroOp.compareLt)
      dut.io.instructions(2).poke(PpimMicroOp.compareLt)
      dut.io.instructions(3).poke(PpimMicroOp.and)
      dut.io.supportStart.poke((7 * 65536).S)
      dut.io.supportEnd.poke((8 * 65536).S)
      dut.io.isStatic.poke(true.B)
      dut.io.allowStatic.poke(true.B)
      execute(dut)
      dut.io.mask.expect(true.B)

      dut.io.isStatic.poke(false.B)
      dut.io.allowStatic.poke(false.B)
      dut.io.supportStart.poke(0.S)
      dut.io.supportEnd.poke(65536.S)
      execute(dut)
      dut.io.mask.expect(true.B)

      // A closed support ending at windowStart still contains that frame, but
      // one starting at the half-open windowEnd has no frame to process.
      dut.io.supportStart.poke((4 * 65536).S)
      dut.io.supportEnd.poke((6 * 65536).S)
      execute(dut)
      dut.io.mask.expect(false.B)
      dut.io.supportStart.poke(0.S)
      dut.io.supportEnd.poke(0.S)
      execute(dut)
      dut.io.mask.expect(true.B)

      dut.io.instructions.foreach(_.poke(PpimMicroOp.nop))
      dut.io.instructions(0).poke(PpimMicroOp.compareGt)
      dut.io.score.poke(64225.S)
      dut.io.tau.poke(64225.S)
      execute(dut)
      dut.io.mask.expect(false.B)
    }
    test(new MaskControlledColumnPath) { dut =>
      dut.io.inputValid.poke(true.B)
      dut.io.regularGpuAccess.poke(false.B)
      dut.io.ppimMask.poke(false.B)
      dut.io.outputValid.expect(false.B)
      dut.io.ppimMask.poke(true.B)
      dut.io.outputValid.expect(true.B)
    }
  }
}
