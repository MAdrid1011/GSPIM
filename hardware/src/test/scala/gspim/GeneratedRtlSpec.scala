package gspim

import chisel3._
import chiseltest._
import chiseltest.simulator.VerilatorBackendAnnotation
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class GeneratedRtlSpec extends AnyFlatSpec with ChiselScalatestTester with Matchers {
  // The Verilator integration fixture keeps all program, task, FIFO, gather,
  // binding, and multi-bank/block states while reducing only replication. The
  // `Generate` target below still emits the default paper-context topology.
  private val p = GspimParams(
    packages = 1,
    diesPerPackage = 2,
    banksPerDie = 4,
    pimBanksPerDie = 2,
    reservedBanksPerDie = 2,
    blockRecords = 4,
    maxBlocksPerTask = 4,
    maxBindingsPerRecord = 2,
  )
  private val defaultLayout = PpimLayoutField.defaults

  private def setField(dut: GspimRank, bank: Int, block: Int, record: Int, field: Int, value: BigInt): Unit = {
    dut.io.rowFields(bank)(block)(record)(field).poke(value.S)
  }

  private def load(dut: GspimRank, program: Int, pc: Int, opcode: UInt, kind: Int = -1): Unit = {
    val effectiveKind = if (kind < 0) program else kind
    dut.io.lpddrWrite.valid.poke(true.B)
    dut.io.lpddrWrite.bits.address.poke(LpddrControlAddress.programWrite.U)
    dut.io.lpddrWrite.bits.program.bank.poke(0.U)
    dut.io.lpddrWrite.bits.program.broadcast.poke(true.B)
    dut.io.lpddrWrite.bits.program.address.poke((program * p.microProgramLength + pc).U)
    dut.io.lpddrWrite.bits.program.kind.poke(effectiveKind.U)
    dut.io.lpddrWrite.bits.program.instruction.poke(opcode)
    dut.io.lpddrWrite.ready.expect(true.B)
    dut.clock.step()
    dut.io.lpddrWrite.valid.poke(false.B)
  }

  private def loadLayout(dut: GspimRank, bank: Int, descriptor: Int, rowField: Int): Unit = {
    dut.io.lpddrWrite.valid.poke(true.B)
    dut.io.lpddrWrite.bits.address.poke(LpddrControlAddress.layoutWrite.U)
    dut.io.lpddrWrite.bits.layout.bank.poke(bank.U)
    dut.io.lpddrWrite.bits.layout.field.poke(descriptor.U)
    dut.io.lpddrWrite.bits.layout.index.poke(rowField.U)
    dut.io.lpddrWrite.ready.expect(true.B)
    dut.clock.step()
    dut.io.lpddrWrite.valid.poke(false.B)
  }

  private def submit(
      dut: GspimRank,
      id: Int,
      kind: UInt,
      program: UInt,
      blocks: Int,
      banks: Int = 1,
      blockStart: Int = 0,
      dependency: Int = 0,
      purpose: UInt = ReorgPurposeId.active,
  ): Unit = {
    val address = if (kind.litValue == PimTaskKind.select.litValue) {
      LpddrControlAddress.pimSelect
    } else {
      LpddrControlAddress.pimReorg
    }
    dut.io.lpddrWrite.valid.poke(true.B)
    dut.io.lpddrWrite.bits.address.poke(address.U)
    dut.io.lpddrWrite.bits.task.taskId.poke(id.U)
    dut.io.lpddrWrite.bits.task.kind.poke(kind)
    dut.io.lpddrWrite.bits.task.die.poke(0.U)
    dut.io.lpddrWrite.bits.task.bank.poke(0.U)
    dut.io.lpddrWrite.bits.task.bankCount.poke(banks.U)
    dut.io.lpddrWrite.bits.task.program.poke(program)
    dut.io.lpddrWrite.bits.task.sourceLine.poke(100.U)
    dut.io.lpddrWrite.bits.task.destinationLine.poke(4096.U)
    dut.io.lpddrWrite.bits.task.blockStart.poke(blockStart.U)
    dut.io.lpddrWrite.bits.task.blockCount.poke(blocks.U)
    dut.io.lpddrWrite.bits.task.purpose.poke(purpose)
    dut.io.lpddrWrite.bits.task.dependency.poke(dependency.U)
    dut.io.lpddrWrite.ready.expect(true.B)
    dut.clock.step()
    dut.io.lpddrWrite.valid.poke(false.B)
  }

  private def waitFor(dut: GspimRank, cycles: Int)(ready: => Boolean): Unit = {
    var seen = ready
    var remaining = cycles
    while (!seen && remaining > 0) {
      dut.clock.step()
      seen = ready
      remaining -= 1
    }
    assert(seen, s"condition did not occur in $cycles cycles")
  }

  private def finishTask(dut: GspimRank): Unit = {
    dut.io.lpddrReadRequest.poke(true.B)
    dut.io.lpddrReadAddress.poke(LpddrControlAddress.completionRead.U)
    waitFor(dut, 120)(dut.io.lpddrCompletionRead.valid.peek().litToBoolean)
    dut.clock.step()
    dut.io.lpddrReadRequest.poke(false.B)
    dut.io.lpddrWrite.ready.expect(true.B)
  }

  private def releaseRange(dut: GspimRank, start: Int, count: Int): Unit = {
    dut.io.hcfReleaseValid.poke(true.B)
    dut.io.hcfReleaseStart.poke(start.U)
    dut.io.hcfReleaseCount.poke(count.U)
    dut.clock.step()
    dut.io.hcfReleaseValid.poke(false.B)
  }

  private def initialize(dut: GspimRank): Unit = {
    dut.io.gpuRequest.poke(false.B)
    dut.io.gpuBank.poke(0.U)
    dut.io.compactRequest.poke(false.B)
    dut.io.compactRequestBank.poke(0.U)
    dut.io.windowStart.poke(0.S)
    dut.io.windowEnd.poke((4 * 65536).S)
    dut.io.tau.poke(0.S)
    dut.io.maskInputValid.poke(true.B)
    dut.io.regularGpuAccess.poke(false.B)
    dut.io.sourceWritebackDone.poke(true.B)
    dut.io.dependencyComplete.poke(true.B)
    dut.io.lpddrCompletionRead.ready.poke(true.B)
    dut.io.lpddrWrite.valid.poke(false.B)
    dut.io.lpddrWrite.bits.address.poke(0.U)
    dut.io.lpddrWrite.bits.task.taskId.poke(0.U)
    dut.io.lpddrWrite.bits.task.kind.poke(PimTaskKind.select)
    dut.io.lpddrWrite.bits.task.die.poke(0.U)
    dut.io.lpddrWrite.bits.task.bank.poke(0.U)
    dut.io.lpddrWrite.bits.task.bankCount.poke(1.U)
    dut.io.lpddrWrite.bits.task.program.poke(0.U)
    dut.io.lpddrWrite.bits.task.sourceLine.poke(0.U)
    dut.io.lpddrWrite.bits.task.destinationLine.poke(0.U)
    dut.io.lpddrWrite.bits.task.blockStart.poke(0.U)
    dut.io.lpddrWrite.bits.task.blockCount.poke(1.U)
    dut.io.lpddrWrite.bits.task.purpose.poke(ReorgPurposeId.active)
    dut.io.lpddrWrite.bits.task.dependency.poke(0.U)
    dut.io.lpddrWrite.bits.program.bank.poke(0.U)
    dut.io.lpddrWrite.bits.program.broadcast.poke(false.B)
    dut.io.lpddrWrite.bits.program.address.poke(0.U)
    dut.io.lpddrWrite.bits.program.kind.poke(0.U)
    dut.io.lpddrWrite.bits.program.instruction.poke(PpimMicroOp.nop)
    dut.io.lpddrWrite.bits.layout.bank.poke(0.U)
    dut.io.lpddrWrite.bits.layout.field.poke(0.U)
    dut.io.lpddrWrite.bits.layout.index.poke(0.U)
    dut.io.lpddrReadRequest.poke(false.B)
    dut.io.lpddrReadAddress.poke(0.U)
    dut.io.hcfReleaseValid.poke(false.B)
    dut.io.hcfReleaseStart.poke(0.U)
    dut.io.hcfReleaseCount.poke(0.U)
    dut.io.hcfGpuFallback.ready.poke(true.B)
    dut.io.hcfBlockCompletion.ready.poke(true.B)
    for (bank <- 0 until p.pimBanksPerDie; block <- 0 until p.maxBlocksPerTask; record <- 0 until p.blockRecords) {
      dut.io.rowFields(bank)(block)(record).foreach(_.poke(0.S))
      setField(dut, bank, block, record, defaultLayout(PpimLayoutField.temporalMean), 10 * 65536)
      setField(dut, bank, block, record, defaultLayout(PpimLayoutField.supportStart), 0)
      setField(dut, bank, block, record, defaultLayout(PpimLayoutField.supportEnd), 0)
      setField(dut, bank, block, record, defaultLayout(PpimLayoutField.score), 0)
      dut.io.bindingStart(bank)(block)(record).poke(0.U)
      dut.io.bindingCount(bank)(block)(record).poke(0.U)
      dut.io.hcfPayloads(bank)(block)(record).poke((bank * 1000 + block * 100 + record).U)
      dut.io.bindingPayloads(bank)(block)(record).foreach(_.poke(0.U))
      dut.io.bindingPayloadsValid(bank)(block)(record).foreach(_.poke(false.B))
      dut.io.gpuReorgMasks(bank)(block)(record).poke(false.B)
    }
  }

  it should "execute all task-selected programs and stall real multi-block gathering for GPU priority" in {
    test(new GspimRank(p)).withAnnotations(Seq(VerilatorBackendAnnotation)) { dut =>
      initialize(dut)
      load(dut, 0, 0, PpimMicroOp.rotSlice)
      load(dut, 0, 1, PpimMicroOp.scale)
      load(dut, 0, 2, PpimMicroOp.dot)
      load(dut, 0, 3, PpimMicroOp.windowDist)
      load(dut, 0, 4, PpimMicroOp.compareLe)
      load(dut, 1, 0, PpimMicroOp.load)
      load(dut, 1, 1, PpimMicroOp.compareLt)
      load(dut, 1, 2, PpimMicroOp.compareLt)
      load(dut, 1, 3, PpimMicroOp.and)
      load(dut, 2, 0, PpimMicroOp.load)
      load(dut, 2, 1, PpimMicroOp.compareLt)
      load(dut, 2, 2, PpimMicroOp.compareLt)
      load(dut, 2, 3, PpimMicroOp.and)
      load(dut, 3, 0, PpimMicroOp.compareGt)

      for (bank <- 0 until 2; block <- 0 until p.maxBlocksPerTask) {
        setField(dut, bank, block, 0, defaultLayout(PpimLayoutField.temporalMean), 65536)
        setField(dut, bank, block, 0, defaultLayout(PpimLayoutField.rotationBase) + 15, 65536)
        setField(dut, bank, block, 0, defaultLayout(PpimLayoutField.scaleBase) + 3, 32768)
      }
      submit(dut, 1, PimTaskKind.select, PpimProgramId.tempActivity, blocks = 1)
      finishTask(dut)
      dut.io.programMask(0).expect(true.B)

      setField(dut, 0, 0, 0, defaultLayout(PpimLayoutField.staticFlag), 1)
      submit(dut, 2, PimTaskKind.select, PpimProgramId.keyframeRange, blocks = 1)
      finishTask(dut)
      dut.io.programMask(0).expect(true.B)

      setField(dut, 0, 0, 0, defaultLayout(PpimLayoutField.staticFlag), 0)
      setField(dut, 0, 0, 0, defaultLayout(PpimLayoutField.supportStart), 0)
      setField(dut, 0, 0, 0, defaultLayout(PpimLayoutField.supportEnd), 65536)
      submit(dut, 3, PimTaskKind.select, PpimProgramId.anchorOverlap, blocks = 1)
      finishTask(dut)
      dut.io.programMask(0).expect(true.B)
      dut.io.bindingStart(0)(0)(0).poke(900.U)
      dut.io.bindingCount(0)(0)(0).poke(2.U)
      dut.io.bindingPayloads(0)(0)(0)(0).poke(501.U)
      dut.io.bindingPayloads(0)(0)(0)(1).poke(502.U)
      dut.io.bindingPayloadsValid(0)(0)(0)(0).poke(true.B)
      dut.io.bindingPayloadsValid(0)(0)(0)(1).poke(true.B)
      // REORG inherits its source program from SELECT task 3.  Its own program
      // field is intentionally ignored at the host-visible task boundary.
      submit(dut, 30, PimTaskKind.reorg, PpimProgramId.tempActivity, blocks = 1, dependency = 3)
      waitFor(dut, 12)(dut.io.hcfFinalComplete.peek().litToBoolean)
      dut.io.hcfFinalOutputCount.expect(2.U)
      dut.io.hcfBindingSourceAddressValid(0).expect(true.B)
      dut.io.hcfBindingSourceAddress(0).expect(900.U)
      dut.io.hcfBindingSourceAddress(1).expect(901.U)
      finishTask(dut)
      // The reduced integration fixture has an eight-entry workspace. Model
      // S2 consuming the two-entry Active Buffer before the next S1 task.
      releaseRange(dut, 0, 2)

      setField(dut, 0, 0, 0, defaultLayout(PpimLayoutField.score), 65536)
      submit(dut, 4, PimTaskKind.select, PpimProgramId.depthStable, blocks = 1)
      finishTask(dut)
      dut.io.programMask(0).expect(true.B)

      submit(dut, 5, PimTaskKind.select, PpimProgramId.tempActivity, blocks = 4, banks = 2)
      finishTask(dut)
      submit(dut, 6, PimTaskKind.reorg, PpimProgramId.tempActivity, blocks = 4, banks = 2, dependency = 5)
      dut.io.gpuRequest.poke(true.B)
      dut.io.compactRequest.poke(true.B)
      dut.clock.step(3)
      dut.io.grantGpu.expect(true.B)
      dut.io.hcfBlockComplete.expect(false.B)
      dut.io.gpuRequest.poke(false.B)
      waitFor(dut, 80)(dut.io.hcfFinalComplete.peek().litToBoolean)
      dut.io.hcfFinalOutputCount.expect(8.U)
      dut.io.hcfSourceLine.expect(107.U)
      dut.io.hcfOverflow.expect(false.B)
      finishTask(dut)
      // S2/S3 have consumed the multi-block Active Buffer before later HCF
      // transactions reuse the same shared reserved-bank workspace.
      releaseRange(dut, 0, 8)

      // A model layout may place the same semantic fields in different row
      // columns.  The program must still select through the loaded descriptor.
      val shuffled = Seq(5, 6, 1, 25, 26, 27, 28)
      shuffled.zipWithIndex.foreach { case (rowField, descriptor) =>
        loadLayout(dut, 0, descriptor, rowField)
      }
      // A layout rewrite changes which physical column supplies temporal mean.
      // Keep all non-target records outside the window in that new column so
      // the one-LSB TEMP rule does not legitimately retain default zero rows.
      for (block <- 0 until 2; record <- 0 until p.blockRecords) {
        setField(dut, 0, block, record, shuffled(PpimLayoutField.temporalMean), 10 * 65536)
      }
      setField(dut, 0, 0, 0, 5, 65536)
      setField(dut, 0, 0, 0, 6 + 15, 65536)
      setField(dut, 0, 0, 0, 1 + 3, 32768)
      submit(dut, 7, PimTaskKind.select, PpimProgramId.tempActivity, blocks = 1)
      finishTask(dut)
      dut.io.programMask(0).expect(true.B)

      // The block range is physical: selection, HCF input, source line, and
      // block completion must all refer to block one rather than block zero.
      setField(dut, 0, 1, 0, shuffled(PpimLayoutField.temporalMean), 65536)
      setField(dut, 0, 1, 0, shuffled(PpimLayoutField.rotationBase) + 15, 65536)
      setField(dut, 0, 1, 0, shuffled(PpimLayoutField.scaleBase) + 3, 32768)
      dut.io.hcfPayloads(0)(1)(0).poke(777.U)
      submit(dut, 8, PimTaskKind.select, PpimProgramId.tempActivity, blocks = 1, blockStart = 1)
      finishTask(dut)
      dut.io.programMask(0).expect(true.B)
      dut.io.hcfBlockCompletion.ready.poke(false.B)
      submit(dut, 9, PimTaskKind.reorg, PpimProgramId.tempActivity, blocks = 1, blockStart = 1, dependency = 8)
      waitFor(dut, 20)(dut.io.hcfBlockCompletion.valid.peek().litToBoolean)
      dut.io.hcfBlockCompletion.bits.taskId.expect(9.U)
      dut.io.hcfBlockCompletion.bits.bank.expect(0.U)
      dut.io.hcfBlockCompletion.bits.block.expect(1.U)
      dut.io.hcfBlockCompletion.bits.sourceLine.expect(101.U)
      dut.io.hcfBlockCompletion.bits.outputCount.expect(1.U)
      dut.io.hcfFirstPayload(0).expect(777.U)
      dut.io.hcfBlockCompletion.ready.poke(true.B)
      dut.clock.step()
      finishTask(dut)

      // Program slot and representation kind are independent model-load
      // metadata. An anchor program in slot three must still trigger indirect
      // binding expansion even though slot three was originally depth-stable.
      load(dut, 3, 0, PpimMicroOp.load, PpimProgramId.anchorOverlap.litValue.toInt)
      load(dut, 3, 1, PpimMicroOp.compareLt, PpimProgramId.anchorOverlap.litValue.toInt)
      load(dut, 3, 2, PpimMicroOp.compareLt, PpimProgramId.anchorOverlap.litValue.toInt)
      load(dut, 3, 3, PpimMicroOp.and, PpimProgramId.anchorOverlap.litValue.toInt)
      setField(dut, 0, 0, 0, shuffled(PpimLayoutField.supportStart), 0)
      setField(dut, 0, 0, 0, shuffled(PpimLayoutField.supportEnd), 65536)
      dut.io.bindingStart(0)(0)(0).poke(950.U)
      dut.io.bindingCount(0)(0)(0).poke(2.U)
      dut.io.bindingPayloads(0)(0)(0)(0).poke(601.U)
      dut.io.bindingPayloads(0)(0)(0)(1).poke(602.U)
      dut.io.bindingPayloadsValid(0)(0)(0)(0).poke(true.B)
      dut.io.bindingPayloadsValid(0)(0)(0)(1).poke(true.B)
      submit(dut, 10, PimTaskKind.select, 3.U, blocks = 1)
      finishTask(dut)
      submit(dut, 11, PimTaskKind.reorg, PpimProgramId.tempActivity, blocks = 1, dependency = 10)
      waitFor(dut, 20)(dut.io.hcfFinalComplete.peek().litToBoolean)
      dut.io.hcfFinalOutputCount.expect(2.U)
      dut.io.hcfBindingSourceAddress(0).expect(950.U)
      dut.io.hcfBindingSourceAddress(1).expect(951.U)
      finishTask(dut)

      // A batch PIM_REORG always uses its GPU selection mask.  It must not
      // inherit anchor binding expansion from the preceding S1 task.
      dut.io.gpuReorgMasks(0)(0)(0).poke(true.B)
      dut.io.hcfPayloads(0)(0)(0).poke(701.U)
      submit(dut, 12, PimTaskKind.reorg, PpimProgramId.tempActivity, blocks = 1, purpose = ReorgPurposeId.batch)
      waitFor(dut, 20)(dut.io.hcfFinalComplete.peek().litToBoolean)
      dut.io.hcfFinalOutputCount.expect(1.U)
      dut.io.hcfBindingSourceAddressValid(0).expect(false.B)
      dut.io.hcfFirstPayload(0).expect(701.U)
      finishTask(dut)
    }
  }
}
