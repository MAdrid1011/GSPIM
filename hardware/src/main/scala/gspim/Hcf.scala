package gspim

import chisel3._
import chisel3.util._

/** L1 filters a bank block and compacts the selected payload words in slot order. */
class BankCompactor(p: GspimParams) extends Module {
  private val countWidth = log2Ceil(p.blockRecords + 1)
  val io = IO(new Bundle {
    val mask = Input(Vec(p.blockRecords, Bool()))
    val payload = Input(Vec(p.blockRecords, UInt(32.W)))
    val selectedCount = Output(UInt(countWidth.W))
    val compactedPayload = Output(Vec(p.blockRecords, UInt(32.W)))
    val compactedValid = Output(Vec(p.blockRecords, Bool()))
  })
  io.selectedCount := PopCount(io.mask)
  for (output <- 0 until p.blockRecords) {
    val matching = (0 until p.blockRecords).map { input =>
      val preceding = if (input == 0) 0.U(countWidth.W) else PopCount(io.mask.take(input))
      io.mask(input) && preceding === output.U
    }
    io.compactedValid(output) := matching.reduce(_ || _)
    io.compactedPayload(output) := Mux1H(matching.zip(io.payload))
  }
}

/** L2 computes contiguous per-bank destinations for one die. */
class DieCompactor(p: GspimParams) extends Module {
  private val countWidth = log2Ceil(p.blockRecords + 1)
  private val totalWidth = log2Ceil(p.pimBanksPerDie * p.blockRecords + 1)
  val io = IO(new Bundle {
    val counts = Input(Vec(p.pimBanksPerDie, UInt(countWidth.W)))
    val offsets = Output(Vec(p.pimBanksPerDie, UInt(totalWidth.W)))
    val total = Output(UInt(totalWidth.W))
  })
  var prefix = 0.U(totalWidth.W)
  for (bank <- 0 until p.pimBanksPerDie) {
    io.offsets(bank) := prefix
    prefix = prefix + io.counts(bank)
  }
  io.total := prefix
}

/** GPU requests win only when they conflict with compaction in the same bank. */
class MemoryAccessArbiter(p: GspimParams) extends Module {
  val io = IO(new Bundle {
    val gpuRequest = Input(Bool())
    val gpuBank = Input(UInt(log2Ceil(p.pimBanksPerDie).W))
    val compactRequest = Input(Bool())
    val compactBank = Input(UInt(log2Ceil(p.pimBanksPerDie).W))
    val grantGpu = Output(Bool())
    val grantCompaction = Output(Bool())
  })
  io.grantGpu := io.gpuRequest
  io.grantCompaction := io.compactRequest && (!io.gpuRequest || io.gpuBank =/= io.compactBank)
}

/** One selected bank block in the HCF ActiveMap FIFO. */
class HcfBlock(p: GspimParams) extends Bundle {
  val taskId = UInt(16.W)
  val die = UInt(log2Ceil(p.diesPerPackage * p.packages).W)
  val bank = UInt(log2Ceil(p.pimBanksPerDie).W)
  val block = UInt(16.W)
  val purpose = UInt(2.W)
  val first = Bool()
  val last = Bool()
  val totalSelected = UInt(16.W)
  val sourceLine = UInt(32.W)
  val expandBindings = Bool()
  val mask = Vec(p.blockRecords, Bool())
  val payload = Vec(p.blockRecords, UInt(32.W))
  val bindingStart = Vec(p.blockRecords, UInt(32.W))
  val bindingCount = Vec(p.blockRecords, UInt(log2Ceil(p.maxBindingsPerRecord + 1).W))
  val bindingPayload = Vec(p.blockRecords, Vec(p.maxBindingsPerRecord, UInt(32.W)))
  val bindingPayloadValid = Vec(p.blockRecords, Vec(p.maxBindingsPerRecord, Bool()))
}

/** Visible queue between PPIM block-mask production and L2 gather. */
class ActiveMapFifo(p: GspimParams) extends Module {
  val io = IO(new Bundle {
    val enqueue = Flipped(Decoupled(new HcfBlock(p)))
    val dequeue = Decoupled(new HcfBlock(p))
  })
  private val queue = Module(new Queue(new HcfBlock(p), 4))
  queue.io.enq <> io.enqueue
  io.dequeue <> queue.io.deq
}

/**
  * L2 gather/writeback transaction over one shared reserved-bank workspace.
  * A task reserves one contiguous interval from its complete selected count;
  * the consumer releases that interval explicitly after use.
  */
class HcfGatherWriter(p: GspimParams) extends Module {
  private val outputRecords = p.blockRecords * p.maxBindingsPerRecord
  private val countWidth = log2Ceil(outputRecords + 1)
  private val capacity = p.reservedBanksPerDie * p.blockRecords
  val io = IO(new Bundle {
    val block = Flipped(Decoupled(new HcfBlock(p)))
    val destinationLine = Input(UInt(32.W))
    val releaseValid = Input(Bool())
    val releaseStart = Input(UInt(16.W))
    val releaseCount = Input(UInt(16.W))
    val destinationBase = Output(UInt(32.W))
    val destinationStart = Output(UInt(16.W))
    val taskRangeStart = Output(UInt(16.W))
    val sourceLine = Output(UInt(32.W))
    val outputCount = Output(UInt(countWidth.W))
    val packedPayload = Output(Vec(outputRecords, UInt(32.W)))
    val packedValid = Output(Vec(outputRecords, Bool()))
    // Anchor payloads are returned by the memory-side binding-table read at
    // these generated addresses; they are not host-injected defaults.
    val bindingSourceAddress = Output(Vec(outputRecords, UInt(32.W)))
    val bindingSourceAddressValid = Output(Vec(outputRecords, Bool()))
    val blockComplete = Output(Bool())
    val finalComplete = Output(Bool())
    val finalOutputCount = Output(UInt(16.W))
    val overflow = Output(Bool())
    val finalOverflow = Output(Bool())
    // The fallback receives the original block transaction, including mask,
    // source payloads, binding metadata, and physical source location.
    val gpuFallback = Decoupled(new HcfBlock(p))
  })
  val compactor = Module(new BankCompactor(p))
  compactor.io.mask := io.block.bits.mask
  compactor.io.payload := io.block.bits.payload

  val expandedPerRecord = (0 until p.blockRecords).map { index =>
    Mux(io.block.bits.mask(index), io.block.bits.bindingCount(index), 0.U)
  }
  val expandedCount = expandedPerRecord.reduce(_ +& _)
  val selectedCount = Mux(io.block.bits.expandBindings, expandedCount, compactor.io.selectedCount)
  val selectedPrefix = (0 until p.blockRecords).map { index =>
    expandedPerRecord.take(index).foldLeft(0.U(16.W))(_ +& _)
  }
  val compactedPayload = Wire(Vec(outputRecords, UInt(32.W)))
  val compactedValid = Wire(Vec(outputRecords, Bool()))
  for (output <- 0 until outputRecords) {
    val bindingMatches = (0 until p.blockRecords).flatMap { record =>
      (0 until p.maxBindingsPerRecord).map { binding =>
        io.block.bits.mask(record) && binding.U < io.block.bits.bindingCount(record) &&
          selectedPrefix(record) + binding.U === output.U
      }
    }
    val bindingPayloads = (0 until p.blockRecords).flatMap { record =>
      (0 until p.maxBindingsPerRecord).map(binding => io.block.bits.bindingPayload(record)(binding))
    }
    val bindingPayloadsValid = (0 until p.blockRecords).flatMap { record =>
      (0 until p.maxBindingsPerRecord).map(binding => io.block.bits.bindingPayloadValid(record)(binding))
    }
    val bindingAddresses = (0 until p.blockRecords).flatMap { record =>
      (0 until p.maxBindingsPerRecord).map(binding => io.block.bits.bindingStart(record) + binding.U)
    }
    val bindingMatch = bindingMatches.reduce(_ || _)
    val bindingAvailable = bindingMatches.zip(bindingPayloadsValid).map {
      case (matches, available) => matches && available
    }.reduce(_ || _)
    when(io.block.fire && io.block.bits.expandBindings) {
      assert(!bindingMatch || bindingAvailable,
        "selected anchor binding has no valid memory response")
    }
    compactedValid(output) := Mux(
      io.block.bits.expandBindings,
      bindingAvailable,
      if (output < p.blockRecords) compactor.io.compactedValid(output) else false.B,
    )
    compactedPayload(output) := Mux(
      io.block.bits.expandBindings,
      Mux1H(bindingMatches.zip(bindingPayloads)),
      if (output < p.blockRecords) compactor.io.compactedPayload(output) else 0.U,
    )
    io.bindingSourceAddress(output) := Mux1H(bindingMatches.zip(bindingAddresses))
    io.bindingSourceAddressValid(output) := io.block.bits.expandBindings && bindingAvailable
  }

  val live = RegInit(VecInit(Seq.fill(capacity)(false.B)))
  val taskActive = RegInit(false.B)
  val taskStart = RegInit(0.U(16.W))
  val taskGrantedCount = RegInit(0.U(16.W))
  val taskOutputCount = RegInit(0.U(16.W))
  val taskOverflow = RegInit(false.B)

  // Python HCF reserves the largest available contiguous interval and compacts
  // its prefix.  The remaining selected records take the explicit GPU fallback.
  // Matching that rule here keeps both public execution models transactionally
  // equivalent when a task is larger than the current live-range hole.
  var firstFit: UInt = 0.U(16.W)
  var largestFree: UInt = 0.U(16.W)
  for (candidateStart <- 0 until capacity) {
    var prefixFree: Bool = true.B
    var available: UInt = 0.U(16.W)
    for (index <- candidateStart until capacity) {
      val useEntry = prefixFree && !live(index)
      available = Mux(useEntry, available + 1.U, available)
      prefixFree = prefixFree && !live(index)
    }
    val better = available > largestFree
    firstFit = Mux(better, candidateStart.U, firstFit)
    largestFree = Mux(better, available, largestFree)
  }

  val firstBlock = io.block.bits.first
  val firstGrantCount = Mux(
    io.block.bits.totalSelected < largestFree,
    io.block.bits.totalSelected,
    largestFree,
  )
  val base = Mux(firstBlock, firstFit, taskStart)
  val priorCount = Mux(firstBlock, 0.U(16.W), taskOutputCount)
  val taskCapacity = Mux(firstBlock, firstGrantCount, taskGrantedCount)
  val remainingCapacity = Mux(taskCapacity > priorCount, taskCapacity - priorCount, 0.U)
  val acceptedCount = Mux(selectedCount > remainingCapacity, remainingCapacity, selectedCount)
  val nextCount = priorCount + acceptedCount
  val fallbackQueue = Module(new Queue(new HcfBlock(p), p.maxBlocksPerTask))
  val canAcceptBlock = !firstBlock || !taskActive
  val fallbackNeeded = selectedCount =/= acceptedCount
  val nextTaskOverflow = Mux(
    firstBlock,
    io.block.bits.totalSelected > firstGrantCount || fallbackNeeded,
    taskOverflow || fallbackNeeded,
  )
  fallbackQueue.io.enq.valid := io.block.valid && canAcceptBlock && fallbackNeeded
  fallbackQueue.io.enq.bits := io.block.bits
  io.gpuFallback.valid := fallbackQueue.io.deq.valid
  io.gpuFallback.bits := fallbackQueue.io.deq.bits
  fallbackQueue.io.deq.ready := io.gpuFallback.ready
  io.block.ready := canAcceptBlock && (!fallbackNeeded || fallbackQueue.io.enq.ready)
  io.destinationBase := io.destinationLine + base + priorCount
  io.destinationStart := base + priorCount
  io.taskRangeStart := base
  io.sourceLine := io.block.bits.sourceLine
  io.outputCount := acceptedCount
  io.packedPayload := compactedPayload
  for (index <- 0 until outputRecords) {
    io.packedValid(index) := compactedValid(index) && index.U < acceptedCount
  }
  io.blockComplete := io.block.fire
  io.finalComplete := io.block.fire && io.block.bits.last
  io.finalOutputCount := nextCount
  io.overflow := fallbackNeeded
  io.finalOverflow := nextTaskOverflow

  when(io.releaseValid) {
    for (index <- 0 until capacity) {
      when(index.U >= io.releaseStart && index.U < io.releaseStart + io.releaseCount) {
        live(index) := false.B
      }
    }
  }
  when(io.block.fire) {
    when(firstBlock) {
      taskStart := firstFit
      taskGrantedCount := firstGrantCount
      taskOutputCount := acceptedCount
      taskOverflow := nextTaskOverflow
      taskActive := !io.block.bits.last
      when(firstGrantCount =/= 0.U) {
        for (index <- 0 until capacity) {
          when(index.U >= firstFit && index.U < firstFit + firstGrantCount) {
            live(index) := true.B
          }
        }
      }
    }.otherwise {
      taskOutputCount := nextCount
      taskOverflow := nextTaskOverflow
      taskActive := !io.block.bits.last
    }
  }
}
