package gspim

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class LpddrTaskAdapterSpec extends AnyFlatSpec with ChiselScalatestTester with Matchers {
  private val p = GspimParams()

  private def initialize(dut: LpddrTaskAdapter): Unit = {
    dut.io.write.valid.poke(false.B)
    dut.io.write.bits.address.poke(0.U)
    dut.io.write.bits.task.taskId.poke(0.U)
    dut.io.write.bits.task.kind.poke(PimTaskKind.select)
    dut.io.write.bits.task.die.poke(0.U)
    dut.io.write.bits.task.bank.poke(0.U)
    dut.io.write.bits.task.bankCount.poke(1.U)
    dut.io.write.bits.task.program.poke(0.U)
    dut.io.write.bits.task.sourceLine.poke(0.U)
    dut.io.write.bits.task.destinationLine.poke(0.U)
    dut.io.write.bits.task.blockStart.poke(0.U)
    dut.io.write.bits.task.blockCount.poke(1.U)
    dut.io.write.bits.task.purpose.poke(ReorgPurposeId.active)
    dut.io.write.bits.task.dependency.poke(0.U)
    dut.io.write.bits.program.bank.poke(0.U)
    dut.io.write.bits.program.broadcast.poke(false.B)
    dut.io.write.bits.program.address.poke(0.U)
    dut.io.write.bits.program.kind.poke(PpimProgramId.tempActivity)
    dut.io.write.bits.program.instruction.poke(PpimMicroOp.nop)
    dut.io.write.bits.layout.bank.poke(0.U)
    dut.io.write.bits.layout.field.poke(0.U)
    dut.io.write.bits.layout.index.poke(0.U)
    dut.io.task.ready.poke(true.B)
    dut.io.programWrite.ready.poke(true.B)
    dut.io.layoutWrite.ready.poke(true.B)
    dut.io.completionIn.valid.poke(false.B)
    dut.io.completionIn.bits.taskId.poke(0.U)
    dut.io.completionIn.bits.die.poke(0.U)
    dut.io.completionIn.bits.outputCount.poke(0.U)
    dut.io.completionIn.bits.destinationLine.poke(0.U)
    dut.io.completionIn.bits.destinationStart.poke(0.U)
    dut.io.completionIn.bits.overflow.poke(false.B)
    dut.io.readRequest.poke(false.B)
    dut.io.readAddress.poke(0.U)
    dut.io.completionRead.ready.poke(true.B)
  }

  it should "decode reserved WRITE commands and return a completion only on the reserved READ" in {
    test(new LpddrTaskAdapter(p)) { dut =>
      initialize(dut)

      dut.io.write.valid.poke(true.B)
      dut.io.write.bits.address.poke(LpddrControlAddress.programWrite.U)
      dut.io.write.bits.program.broadcast.poke(true.B)
      dut.io.write.bits.program.address.poke(3.U)
      dut.io.write.bits.program.kind.poke(PpimProgramId.anchorOverlap)
      dut.io.write.bits.program.instruction.poke(PpimMicroOp.compareLe)
      dut.io.programWrite.valid.expect(true.B)
      dut.io.task.valid.expect(false.B)
      dut.clock.step()

      dut.io.write.bits.address.poke(LpddrControlAddress.pimSelect.U)
      dut.io.write.bits.task.taskId.poke(17.U)
      dut.io.write.bits.task.kind.poke(PimTaskKind.select)
      dut.io.write.bits.task.blockStart.poke(2.U)
      dut.io.write.bits.task.blockCount.poke(1.U)
      dut.io.task.valid.expect(true.B)
      dut.io.task.bits.taskId.expect(17.U)
      dut.io.task.bits.blockStart.expect(2.U)
      dut.clock.step()
      dut.io.write.valid.poke(false.B)

      dut.io.completionIn.valid.poke(true.B)
      dut.io.completionIn.bits.taskId.poke(17.U)
      dut.io.completionIn.bits.die.poke(1.U)
      dut.io.completionIn.bits.outputCount.poke(4.U)
      dut.io.completionIn.bits.destinationLine.poke(4096.U)
      dut.io.completionIn.bits.destinationStart.poke(12.U)
      dut.io.completionIn.bits.overflow.poke(false.B)
      dut.io.completionIn.ready.expect(true.B)
      dut.clock.step()
      dut.io.completionIn.valid.poke(false.B)

      dut.io.readRequest.poke(true.B)
      dut.io.readAddress.poke(LpddrControlAddress.completionRead.U)
      dut.io.completionRead.valid.expect(true.B)
      dut.io.completionRead.bits.taskId.expect(17.U)
      dut.io.completionRead.bits.outputCount.expect(4.U)
      dut.io.completionRead.bits.destinationStart.expect(12.U)
      dut.clock.step()
      dut.io.completionRead.valid.expect(false.B)
    }
  }
}
