package gspim

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class TaskControllerSpec extends AnyFlatSpec with ChiselScalatestTester with Matchers {
  behavior of "TaskController"

  it should "latch program and block scope, then report the execution count" in {
    test(new TaskController(GspimParams())) { dut =>
      dut.io.request.valid.poke(true.B)
      dut.io.request.bits.taskId.poke(7.U)
      dut.io.request.bits.kind.poke(PimTaskKind.select)
      dut.io.request.bits.die.poke(0.U)
      dut.io.request.bits.bank.poke(2.U)
      dut.io.request.bits.bankCount.poke(3.U)
      dut.io.request.bits.program.poke(PpimProgramId.tempActivity)
      dut.io.request.bits.sourceLine.poke(10.U)
      dut.io.request.bits.destinationLine.poke(20.U)
      dut.io.request.bits.blockStart.poke(1.U)
      dut.io.request.bits.blockCount.poke(3.U)
      dut.io.request.bits.purpose.poke(ReorgPurposeId.active)
      dut.io.request.bits.dependency.poke(0.U)
      dut.io.sourceWritebackDone.poke(true.B)
      dut.io.dependencyComplete.poke(true.B)
      dut.io.executionDone.poke(false.B)
      dut.io.executionOutputCount.poke(5.U)
      dut.io.executionDestinationStart.poke(3.U)
      dut.io.executionOverflow.poke(false.B)
      dut.io.completion.ready.poke(true.B)
      dut.io.request.ready.expect(true.B)
      dut.clock.step()
      dut.io.request.valid.poke(false.B)
      dut.io.activeProgram.expect(PpimProgramId.tempActivity)
      dut.io.activeBank.expect(2.U)
      dut.io.activeBankCount.expect(3.U)
      dut.io.activeBlockStart.expect(1.U)
      dut.io.activeBlockCount.expect(3.U)
      dut.io.activePurpose.expect(ReorgPurposeId.active)
      dut.io.executionDone.poke(true.B)
      dut.clock.step()
      dut.io.completion.valid.expect(true.B)
      dut.io.completion.bits.taskId.expect(7.U)
      dut.io.completion.bits.outputCount.expect(5.U)
      dut.io.completion.bits.destinationStart.expect(3.U)
      dut.io.completion.bits.overflow.expect(false.B)
      dut.io.destinationWritten.expect(true.B)
      dut.clock.step()
      dut.io.destinationWritten.expect(false.B)
    }
  }
}
