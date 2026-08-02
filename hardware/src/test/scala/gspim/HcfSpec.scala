package gspim

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class HcfSpec extends AnyFlatSpec with ChiselScalatestTester with Matchers {
  behavior of "HCF"

  // HCF behavior is independent of the paper topology's replication factor.
  // Keeping this unit fixture small makes the transaction test practical on
  // the default Chisel simulator; `Generate` retains the paper configuration.
  private val unitParams = GspimParams(
    packages = 1,
    diesPerPackage = 2,
    banksPerDie = 3,
    pimBanksPerDie = 2,
    reservedBanksPerDie = 1,
    blockRecords = 4,
    maxBlocksPerTask = 2,
    maxBindingsPerRecord = 2,
  )

  private def driveBlock(
      dut: HcfGatherWriter,
      p: GspimParams,
      task: Int,
      purpose: UInt,
      first: Boolean,
      last: Boolean,
      total: Int,
      selected: Set[Int],
  ): Unit = {
    dut.io.block.valid.poke(true.B)
    dut.io.block.bits.taskId.poke(task.U)
    dut.io.block.bits.die.poke(0.U)
    dut.io.block.bits.bank.poke(1.U)
    dut.io.block.bits.block.poke(task.U)
    dut.io.block.bits.purpose.poke(purpose)
    dut.io.block.bits.first.poke(first.B)
    dut.io.block.bits.last.poke(last.B)
    dut.io.block.bits.totalSelected.poke(total.U)
    dut.io.block.bits.sourceLine.poke((100 + task).U)
    dut.io.block.bits.expandBindings.poke(false.B)
    for (index <- 0 until p.blockRecords) {
      dut.io.block.bits.mask(index).poke(selected.contains(index).B)
      dut.io.block.bits.payload(index).poke((300 + index).U)
      dut.io.block.bits.bindingStart(index).poke(0.U)
      dut.io.block.bits.bindingCount(index).poke(0.U)
      dut.io.block.bits.bindingPayload(index).foreach(_.poke(0.U))
      dut.io.block.bits.bindingPayloadValid(index).foreach(_.poke(true.B))
    }
  }

  it should "compact payloads and allocate/release one shared live workspace" in {
    test(new BankCompactor(unitParams)) { dut =>
      for (index <- 0 until unitParams.blockRecords) {
        dut.io.mask(index).poke((index == 1 || index == 3).B)
        dut.io.payload(index).poke((100 + index).U)
      }
      dut.io.selectedCount.expect(2.U)
      dut.io.compactedValid(0).expect(true.B)
      dut.io.compactedPayload(0).expect(101.U)
      dut.io.compactedPayload(1).expect(103.U)
    }
    test(new DieCompactor(unitParams)) { dut =>
      dut.io.counts(0).poke(1.U)
      dut.io.counts(1).poke(2.U)
      dut.io.offsets(0).expect(0.U)
      dut.io.offsets(1).expect(1.U)
      dut.io.total.expect(3.U)
    }
    test(new MemoryAccessArbiter(unitParams)) { dut =>
      dut.io.gpuRequest.poke(true.B)
      dut.io.gpuBank.poke(1.U)
      dut.io.compactRequest.poke(true.B)
      dut.io.compactBank.poke(1.U)
      dut.io.grantGpu.expect(true.B)
      dut.io.grantCompaction.expect(false.B)
      dut.io.compactBank.poke(0.U)
      dut.io.grantCompaction.expect(true.B)
    }
    test(new ActiveMapFifo(unitParams)) { dut =>
      dut.io.enqueue.valid.poke(true.B)
      dut.io.enqueue.bits.taskId.poke(9.U)
      dut.io.enqueue.bits.die.poke(1.U)
      dut.io.enqueue.bits.bank.poke(1.U)
      dut.io.enqueue.bits.block.poke(3.U)
      dut.io.enqueue.bits.purpose.poke(ReorgPurposeId.batch)
      dut.io.enqueue.bits.first.poke(true.B)
      dut.io.enqueue.bits.last.poke(true.B)
      dut.io.enqueue.bits.totalSelected.poke(1.U)
      dut.io.enqueue.bits.sourceLine.poke(77.U)
      dut.io.enqueue.bits.expandBindings.poke(false.B)
      for (index <- 0 until unitParams.blockRecords) {
        dut.io.enqueue.bits.mask(index).poke((index == 0).B)
        dut.io.enqueue.bits.payload(index).poke((200 + index).U)
        dut.io.enqueue.bits.bindingStart(index).poke(0.U)
        dut.io.enqueue.bits.bindingCount(index).poke(0.U)
        dut.io.enqueue.bits.bindingPayload(index).foreach(_.poke(0.U))
      }
      dut.io.dequeue.ready.poke(true.B)
      dut.clock.step()
      dut.io.enqueue.valid.poke(false.B)
      dut.io.dequeue.valid.expect(true.B)
      dut.io.dequeue.bits.taskId.expect(9.U)
      dut.io.dequeue.bits.bank.expect(1.U)
      dut.io.dequeue.bits.purpose.expect(ReorgPurposeId.batch)
      dut.io.dequeue.bits.payload(0).expect(200.U)
    }
    test(new HcfGatherWriter(unitParams)) { dut =>
      dut.io.destinationLine.poke(4096.U)
      dut.io.releaseValid.poke(false.B)
      dut.io.releaseStart.poke(0.U)
      dut.io.releaseCount.poke(0.U)
      dut.io.gpuFallback.ready.poke(true.B)
      driveBlock(dut, unitParams, 1, ReorgPurposeId.active, first = true, last = true, total = 1, Set(2))
      dut.io.destinationBase.expect(4096.U)
      dut.io.destinationStart.expect(0.U)
      dut.io.taskRangeStart.expect(0.U)
      dut.io.sourceLine.expect(101.U)
      dut.io.outputCount.expect(1.U)
      dut.io.packedPayload(0).expect(302.U)
      dut.io.blockComplete.expect(true.B)
      dut.io.finalComplete.expect(true.B)
      dut.io.finalOutputCount.expect(1.U)
      dut.io.overflow.expect(false.B)
      dut.clock.step()
      dut.io.block.valid.poke(false.B)

      driveBlock(dut, unitParams, 2, ReorgPurposeId.stable, first = true, last = true, total = 1, Set(3))
      dut.io.destinationStart.expect(1.U)
      dut.io.taskRangeStart.expect(1.U)
      dut.clock.step()
      dut.io.block.valid.poke(false.B)

      dut.io.releaseValid.poke(true.B)
      dut.io.releaseStart.poke(0.U)
      dut.io.releaseCount.poke(1.U)
      dut.clock.step()
      dut.io.releaseValid.poke(false.B)
      dut.io.releaseValid.poke(true.B)
      dut.io.releaseStart.poke(1.U)
      dut.io.releaseCount.poke(1.U)
      dut.clock.step()
      dut.io.releaseValid.poke(false.B)
      driveBlock(dut, unitParams, 3, ReorgPurposeId.batch, first = true, last = true, total = 1, Set(2))
      dut.io.destinationStart.expect(0.U)
      dut.io.taskRangeStart.expect(0.U)
    }
    test(new HcfGatherWriter(unitParams)) { dut =>
      dut.io.destinationLine.poke(0.U)
      dut.io.releaseValid.poke(false.B)
      dut.io.releaseStart.poke(0.U)
      dut.io.releaseCount.poke(0.U)
      dut.io.gpuFallback.ready.poke(false.B)
      // A five-record task meets a four-record workspace. HCF keeps the first
      // physical block in its contiguous range and falls back only the tail.
      driveBlock(dut, unitParams, 4, ReorgPurposeId.batch, first = true, last = false, total = 5, (0 until unitParams.blockRecords).toSet)
      dut.io.outputCount.expect(4.U)
      dut.io.overflow.expect(false.B)
      dut.clock.step()
      dut.io.block.valid.poke(false.B)
      driveBlock(dut, unitParams, 4, ReorgPurposeId.batch, first = false, last = true, total = 5, Set(0))
      dut.io.outputCount.expect(0.U)
      dut.io.overflow.expect(true.B)
      dut.io.finalOverflow.expect(true.B)
      dut.io.finalOutputCount.expect(4.U)
      dut.clock.step()
      dut.io.block.valid.poke(false.B)
      dut.io.gpuFallback.valid.expect(true.B)
      dut.io.gpuFallback.bits.taskId.expect(4.U)
      dut.io.gpuFallback.bits.sourceLine.expect(104.U)
      dut.io.gpuFallback.bits.mask(0).expect(true.B)
      dut.io.gpuFallback.ready.poke(true.B)
      dut.clock.step()
    }
    test(new HcfGatherWriter(unitParams)) { dut =>
      dut.io.destinationLine.poke(0.U)
      dut.io.releaseValid.poke(false.B)
      dut.io.releaseStart.poke(0.U)
      dut.io.releaseCount.poke(0.U)
      dut.io.gpuFallback.ready.poke(true.B)
      driveBlock(dut, unitParams, 7, ReorgPurposeId.active, first = true, last = true, total = 2, Set(1))
      dut.io.block.bits.expandBindings.poke(true.B)
      dut.io.block.bits.bindingStart(1).poke(900.U)
      dut.io.block.bits.bindingCount(1).poke(2.U)
      dut.io.block.bits.bindingPayload(1)(0).poke(501.U)
      dut.io.block.bits.bindingPayload(1)(1).poke(502.U)
      dut.io.outputCount.expect(2.U)
      dut.io.packedValid(0).expect(true.B)
      dut.io.packedValid(1).expect(true.B)
      dut.io.packedPayload(0).expect(501.U)
      dut.io.packedPayload(1).expect(502.U)
    }
  }
}
