package gspim

import chisel3._
import chisel3.util._

/**
  * One die-level PPIM/HCF transaction model.
  *
  * PPIM selection and HCF reorganization have independent task controllers.
  * A completed selection map is snapshotted when its dependent reorganization
  * starts, so PPIM may select the next window while HCF drains the prior map.
  */
class GspimRank(p: GspimParams = GspimParams()) extends Module {
  private val hcfCountWidth = log2Ceil(p.pimBanksPerDie * p.maxBlocksPerTask * p.blockRecords * p.maxBindingsPerRecord + 1)
  private val programAddressWidth = log2Ceil(p.programSlots * p.microProgramLength)
  private val taskBlockWidth = log2Ceil(p.maxBlocksPerTask + 1)
  private val taskBankWidth = log2Ceil(p.pimBanksPerDie + 1)
  private val bankIndexWidth = log2Ceil(p.pimBanksPerDie)
  private val blockIndexWidth = log2Ceil(p.maxBlocksPerTask)
  private val rowFieldIndexWidth = log2Ceil(p.rowFieldSlots)
  private val layoutDescriptorWidth = log2Ceil(PpimLayoutField.count)
  val io = IO(new Bundle {
    val gpuRequest = Input(Bool())
    val gpuBank = Input(UInt(bankIndexWidth.W))
    val compactRequest = Input(Bool())
    val compactRequestBank = Input(UInt(bankIndexWidth.W))
    val grantGpu = Output(Bool())
    val grantCompaction = Output(Bool())

    // PPIM reads a fixed-size record from the local row buffer.  A bank-local
    // layout descriptor below maps representation-specific fields into this
    // common row-field array.
    val rowFields = Input(Vec(p.pimBanksPerDie, Vec(p.maxBlocksPerTask, Vec(p.blockRecords, Vec(p.rowFieldSlots, SInt(p.operandWidth.W))))))
    val windowStart = Input(SInt(p.operandWidth.W))
    val windowEnd = Input(SInt(p.operandWidth.W))
    val tau = Input(SInt(p.operandWidth.W))
    val programMask = Output(Vec(p.blockRecords, Bool()))
    val programSigmaTt = Output(Vec(p.blockRecords, SInt(p.operandWidth.W)))
    val maskInputValid = Input(Bool())
    val regularGpuAccess = Input(Bool())
    val maskOutputValid = Output(Vec(p.blockRecords, Bool()))

    val hcfPayloads = Input(Vec(p.pimBanksPerDie, Vec(p.maxBlocksPerTask, Vec(p.blockRecords, UInt(32.W)))))
    val bindingStart = Input(Vec(p.pimBanksPerDie, Vec(p.maxBlocksPerTask, Vec(p.blockRecords, UInt(32.W)))))
    val bindingCount = Input(Vec(p.pimBanksPerDie, Vec(p.maxBlocksPerTask, Vec(p.blockRecords, UInt(log2Ceil(p.maxBindingsPerRecord + 1).W)))))
    val bindingPayloads = Input(Vec(p.pimBanksPerDie, Vec(p.maxBlocksPerTask, Vec(p.blockRecords, Vec(p.maxBindingsPerRecord, UInt(32.W))))))
    val bindingPayloadsValid = Input(Vec(p.pimBanksPerDie, Vec(p.maxBlocksPerTask, Vec(p.blockRecords, Vec(p.maxBindingsPerRecord, Bool())))))
    val hcfOffsets = Output(Vec(p.pimBanksPerDie, UInt(hcfCountWidth.W)))
    val hcfRequested = Output(UInt(hcfCountWidth.W))
    val hcfGranted = Output(UInt(hcfCountWidth.W))
    val hcfOverflow = Output(Bool())
    val hcfFirstPayload = Output(Vec(p.pimBanksPerDie, UInt(32.W)))
    val hcfFifoReady = Output(Bool())
    val hcfBlockComplete = Output(Bool())
    val hcfBlockCompletion = Decoupled(new HcfBlockCompletion(p))
    val hcfFinalComplete = Output(Bool())
    val hcfCompletionTaskId = Output(UInt(16.W))
    val hcfSourceLine = Output(UInt(32.W))
    val hcfDestinationBase = Output(UInt(32.W))
    val hcfDestinationStart = Output(UInt(16.W))
    val hcfTaskRangeStart = Output(UInt(16.W))
    val hcfBlockOutputCount = Output(UInt(log2Ceil(p.blockRecords * p.maxBindingsPerRecord + 1).W))
    val hcfFinalOutputCount = Output(UInt(16.W))
    val hcfBindingSourceAddress = Output(Vec(p.blockRecords * p.maxBindingsPerRecord, UInt(32.W)))
    val hcfBindingSourceAddressValid = Output(Vec(p.blockRecords * p.maxBindingsPerRecord, Bool()))
    val hcfGpuFallback = Decoupled(new HcfBlock(p))
    val hcfReleaseValid = Input(Bool())
    val hcfReleaseStart = Input(UInt(16.W))
    val hcfReleaseCount = Input(UInt(16.W))
    // S4 supplies these physical masks for batch PIM_REORG.  S1/S3 reorgs use
    // the snapshotted PPIM selection map instead.
    val gpuReorgMasks = Input(Vec(p.pimBanksPerDie, Vec(p.maxBlocksPerTask, Vec(p.blockRecords, Bool()))))

    val programReadInstruction = Output(Vec(p.microProgramLength, UInt(p.microInstructionWidth.W)))
    val activeLayout = Output(Vec(PpimLayoutField.count, UInt(rowFieldIndexWidth.W)))

    // The paper-visible host boundary is standard LPDDR WRITE/READ traffic to
    // reserved control addresses.  No direct task or model-load bypass exists.
    val lpddrWrite = Flipped(Decoupled(new LpddrControlWrite(p)))
    val lpddrReadRequest = Input(Bool())
    val lpddrReadAddress = Input(UInt(32.W))
    val lpddrCompletionRead = Decoupled(new PimCompletion(p))
    val sourceWritebackDone = Input(Bool())
    val dependencyComplete = Input(Bool())
    val destinationWritten = Output(Bool())
  })

  val lpddrAdapter = Module(new LpddrTaskAdapter(p))
  lpddrAdapter.io.write <> io.lpddrWrite
  lpddrAdapter.io.readRequest := io.lpddrReadRequest
  lpddrAdapter.io.readAddress := io.lpddrReadAddress
  io.lpddrCompletionRead <> lpddrAdapter.io.completionRead

  val taskRequest = lpddrAdapter.io.task
  val selectController = Module(new TaskController(p))
  val reorgController = Module(new TaskController(p))
  selectController.io.request.valid := taskRequest.valid && taskRequest.bits.kind === PimTaskKind.select
  selectController.io.request.bits := taskRequest.bits
  reorgController.io.request.valid := taskRequest.valid && taskRequest.bits.kind === PimTaskKind.reorg
  reorgController.io.request.bits := taskRequest.bits
  taskRequest.ready := Mux(
    taskRequest.bits.kind === PimTaskKind.select,
    selectController.io.request.ready,
    reorgController.io.request.ready,
  )
  for (controller <- Seq(selectController, reorgController)) {
    controller.io.sourceWritebackDone := io.sourceWritebackDone
    controller.io.dependencyComplete := io.dependencyComplete
  }
  val completionArbiter = Module(new Arbiter(new PimCompletion(p), 2))
  completionArbiter.io.in(0) <> selectController.io.completion
  completionArbiter.io.in(1) <> reorgController.io.completion
  lpddrAdapter.io.completionIn <> completionArbiter.io.out
  io.destinationWritten := selectController.io.destinationWritten || reorgController.io.destinationWritten
  val completedSelectTaskId = RegInit(0.U(16.W))
  val completedSelectProgram = RegInit(PpimProgramId.tempActivity)
  when(selectController.io.completion.fire) {
    completedSelectTaskId := selectController.io.activeTaskId
  }
  when(reorgController.io.request.fire && reorgController.io.request.bits.dependency =/= 0.U) {
    assert(reorgController.io.request.bits.dependency === completedSelectTaskId,
      "PIM_REORG dependency must name the most recently completed PIM_SELECT")
  }
  when(reorgController.io.request.fire && reorgController.io.request.bits.purpose =/= ReorgPurposeId.batch) {
    assert(reorgController.io.request.bits.dependency =/= 0.U,
      "S1/S3 PIM_REORG must identify the PIM_SELECT map it consumes")
  }
  val reorgSelectionProgram = RegInit(PpimProgramId.tempActivity)
  when(reorgController.io.request.fire && reorgController.io.request.bits.purpose =/= ReorgPurposeId.batch) {
    reorgSelectionProgram := completedSelectProgram
  }

  // The paper places a program buffer and static-layout table beside every
  // PIM-bank row buffer.  A broadcast model-load write initializes all banks;
  // targeted writes make per-bank layout differences observable in RTL tests.
  val programBuffers = Seq.fill(p.pimBanksPerDie)(Module(new PPIMProgramBuffer(p)))
  val layoutBuffers = Seq.fill(p.pimBanksPerDie)(Module(new PPIMLayoutBuffer(p)))
  lpddrAdapter.io.programWrite.ready := true.B
  lpddrAdapter.io.layoutWrite.ready := true.B
  for (bank <- 0 until p.pimBanksPerDie) {
    programBuffers(bank).io.writeEnable := lpddrAdapter.io.programWrite.valid &&
      (lpddrAdapter.io.programWrite.bits.broadcast || lpddrAdapter.io.programWrite.bits.bank === bank.U)
    programBuffers(bank).io.writeAddress := lpddrAdapter.io.programWrite.bits.address
    programBuffers(bank).io.writeKind := lpddrAdapter.io.programWrite.bits.kind
    programBuffers(bank).io.writeInstruction := lpddrAdapter.io.programWrite.bits.instruction
    programBuffers(bank).io.readProgram := selectController.io.activeProgram
    layoutBuffers(bank).io.writeEnable := lpddrAdapter.io.layoutWrite.valid && lpddrAdapter.io.layoutWrite.bits.bank === bank.U
    layoutBuffers(bank).io.writeField := lpddrAdapter.io.layoutWrite.bits.field
    layoutBuffers(bank).io.writeIndex := lpddrAdapter.io.layoutWrite.bits.index
  }
  val programInstructions = VecInit(programBuffers.map(_.io.readInstruction))
  val programKinds = VecInit(programBuffers.map(_.io.readKind))
  val layouts = VecInit(layoutBuffers.map(_.io.fields))

  val selectBank = RegInit(0.U(taskBankWidth.W))
  val selectBlock = RegInit(0.U(taskBlockWidth.W))
  val selectStarted = RegInit(false.B)
  val selectionReady = RegInit(false.B)
  val selectionMaps = RegInit(VecInit(Seq.fill(p.pimBanksPerDie)(VecInit(Seq.fill(p.maxBlocksPerTask)(VecInit(Seq.fill(p.blockRecords)(false.B)))))))
  val selectedBank = (selectController.io.activeBank + selectBank)(bankIndexWidth - 1, 0)
  val selectedBlock = (selectController.io.activeBlockStart + selectBlock)(blockIndexWidth - 1, 0)
  when(selectController.io.completion.fire) {
    completedSelectProgram := programKinds(selectedBank)
  }
  io.programReadInstruction := programInstructions(selectedBank)
  io.activeLayout := layouts(selectedBank)
  val executors = Seq.fill(p.blockRecords)(Module(new PPIMProgramExecutor(p)))
  for (record <- 0 until p.blockRecords) {
    val fields = io.rowFields(selectedBank)(selectedBlock)(record)
    val layout = layouts(selectedBank)
    executors(record).io.start := selectController.io.active && !selectStarted && !selectionReady
    executors(record).io.instructions := programInstructions(selectedBank)
    executors(record).io.temporalMean := fields(layout(PpimLayoutField.temporalMean))
    for (row <- 0 until 4; column <- 0 until 4) {
      executors(record).io.temporalRotation(row)(column) := fields(layout(PpimLayoutField.rotationBase) + (row * 4 + column).U)
    }
    for (index <- 0 until 4) {
      executors(record).io.temporalScale(index) := fields(layout(PpimLayoutField.scaleBase) + index.U)
    }
    executors(record).io.windowStart := io.windowStart
    executors(record).io.windowEnd := io.windowEnd
    executors(record).io.isStatic := fields(layout(PpimLayoutField.staticFlag)) =/= 0.S
    executors(record).io.allowStatic := programKinds(selectedBank) === PpimProgramId.keyframeRange
    executors(record).io.supportStart := fields(layout(PpimLayoutField.supportStart))
    executors(record).io.supportEnd := fields(layout(PpimLayoutField.supportEnd))
    executors(record).io.score := fields(layout(PpimLayoutField.score))
    executors(record).io.tau := io.tau
    io.programMask(record) := executors(record).io.mask
    io.programSigmaTt(record) := executors(record).io.sigmaTt
  }
  val calculatedMask = VecInit(executors.map(_.io.mask))
  val executorsDone = executors.map(_.io.done).reduce(_ && _)
  when(!selectController.io.active) {
    selectBank := 0.U
    selectBlock := 0.U
    selectStarted := false.B
    selectionReady := false.B
  }.elsewhen(!selectStarted && !selectionReady) {
    selectStarted := true.B
  }.elsewhen(executorsDone) {
    selectionMaps(selectedBank)(selectedBlock) := calculatedMask
    selectStarted := false.B
    when(selectBlock +& 1.U >= selectController.io.activeBlockCount) {
      selectBlock := 0.U
      when(selectBank +& 1.U >= selectController.io.activeBankCount) {
        selectionReady := true.B
      }.otherwise {
        selectBank := selectBank +& 1.U
      }
    }.otherwise {
      selectBlock := selectBlock +& 1.U
    }
  }

  for (record <- 0 until p.blockRecords) {
    val maskPath = Module(new MaskControlledColumnPath)
    maskPath.io.inputValid := io.maskInputValid
    maskPath.io.regularGpuAccess := io.regularGpuAccess
    maskPath.io.ppimMask := Mux(selectionReady, selectionMaps(selectedBank)(selectedBlock)(record), calculatedMask(record))
    io.maskOutputValid(record) := maskPath.io.outputValid
  }

  val reorgBank = RegInit(0.U(taskBankWidth.W))
  val reorgBlock = RegInit(0.U(taskBlockWidth.W))
  val reorgEnqueued = RegInit(false.B)
  val reorgMaps = RegInit(VecInit(Seq.fill(p.pimBanksPerDie)(VecInit(Seq.fill(p.maxBlocksPerTask)(VecInit(Seq.fill(p.blockRecords)(false.B)))))))
  when(reorgController.io.request.fire) {
    reorgMaps := selectionMaps
  }
  val reorgPhysicalBank = (reorgController.io.activeBank + reorgBank)(bankIndexWidth - 1, 0)
  val reorgBlockIndex = (reorgController.io.activeBlockStart + reorgBlock)(blockIndexWidth - 1, 0)
  when(!reorgController.io.active) {
    reorgBank := 0.U
    reorgBlock := 0.U
    reorgEnqueued := false.B
  }
  val currentReorgMask = Wire(Vec(p.blockRecords, Bool()))
  for (record <- 0 until p.blockRecords) {
    currentReorgMask(record) := Mux(
      reorgController.io.activePurpose === ReorgPurposeId.batch,
      io.gpuReorgMasks(reorgPhysicalBank)(reorgBlockIndex)(record),
      reorgMaps(reorgPhysicalBank)(reorgBlockIndex)(record),
    )
  }
  val bankCompactors = Seq.fill(p.pimBanksPerDie)(Module(new BankCompactor(p)))
  val dieCompactor = Module(new DieCompactor(p))
  for (bank <- 0 until p.pimBanksPerDie) {
    val mask = Wire(Vec(p.blockRecords, Bool()))
    for (record <- 0 until p.blockRecords) {
      mask(record) := Mux(reorgPhysicalBank === bank.U, currentReorgMask(record), false.B)
    }
    bankCompactors(bank).io.mask := mask
    bankCompactors(bank).io.payload := io.hcfPayloads(bank)(reorgBlockIndex)
    dieCompactor.io.counts(bank) := bankCompactors(bank).io.selectedCount
    io.hcfFirstPayload(bank) := bankCompactors(bank).io.compactedPayload(0)
  }
  io.hcfOffsets := dieCompactor.io.offsets
  io.hcfRequested := dieCompactor.io.total

  val expandBindings = reorgController.io.activePurpose =/= ReorgPurposeId.batch &&
    reorgSelectionProgram === PpimProgramId.anchorOverlap
  val totalSelected = (0 until p.pimBanksPerDie).flatMap { bank =>
    (0 until p.maxBlocksPerTask).map { block =>
      val expanded = (0 until p.blockRecords).map { record =>
        Mux(reorgMaps(bank)(block)(record), io.bindingCount(bank)(block)(record), 0.U)
      }.reduce(_ +& _)
      val selectedMask = Wire(Vec(p.blockRecords, Bool()))
      for (record <- 0 until p.blockRecords) {
        selectedMask(record) := Mux(
          reorgController.io.activePurpose === ReorgPurposeId.batch,
          io.gpuReorgMasks(bank)(block)(record),
          reorgMaps(bank)(block)(record),
        )
      }
      val bankInScope = bank.U >= reorgController.io.activeBank && bank.U < reorgController.io.activeBank +& reorgController.io.activeBankCount
      val blockInScope = block.U >= reorgController.io.activeBlockStart &&
        block.U < reorgController.io.activeBlockStart +& reorgController.io.activeBlockCount
      Mux(bankInScope && blockInScope, Mux(expandBindings, expanded, PopCount(selectedMask)), 0.U)
    }
  }.reduce(_ +& _)
  val activeMapFifo = Module(new ActiveMapFifo(p))
  val hcfBlock = Wire(new HcfBlock(p))
  hcfBlock.taskId := reorgController.io.activeTaskId
  hcfBlock.die := reorgController.io.activeDie
  hcfBlock.bank := reorgPhysicalBank
  hcfBlock.block := reorgController.io.activeBlockStart + reorgBlock
  hcfBlock.purpose := reorgController.io.activePurpose
  hcfBlock.first := reorgBank === 0.U && reorgBlock === 0.U
  hcfBlock.last := reorgBank +& 1.U >= reorgController.io.activeBankCount && reorgBlock +& 1.U >= reorgController.io.activeBlockCount
  hcfBlock.totalSelected := totalSelected
  hcfBlock.sourceLine := reorgController.io.activeSourceLine + reorgBank * p.maxBlocksPerTask.U + reorgBlockIndex
  hcfBlock.expandBindings := expandBindings
  hcfBlock.mask := currentReorgMask
  hcfBlock.payload := io.hcfPayloads(reorgPhysicalBank)(reorgBlockIndex)
  hcfBlock.bindingStart := io.bindingStart(reorgPhysicalBank)(reorgBlockIndex)
  hcfBlock.bindingCount := io.bindingCount(reorgPhysicalBank)(reorgBlockIndex)
  hcfBlock.bindingPayload := io.bindingPayloads(reorgPhysicalBank)(reorgBlockIndex)
  hcfBlock.bindingPayloadValid := io.bindingPayloadsValid(reorgPhysicalBank)(reorgBlockIndex)
  activeMapFifo.io.enqueue.valid := reorgController.io.active &&
    !reorgEnqueued && reorgBank < reorgController.io.activeBankCount && reorgBlock < reorgController.io.activeBlockCount
  activeMapFifo.io.enqueue.bits := hcfBlock
  io.hcfFifoReady := activeMapFifo.io.enqueue.ready
  when(activeMapFifo.io.enqueue.fire) {
    when(reorgBlock +& 1.U >= reorgController.io.activeBlockCount) {
      reorgBlock := 0.U
      when(reorgBank +& 1.U >= reorgController.io.activeBankCount) {
        reorgEnqueued := true.B
      }.otherwise {
        reorgBank := reorgBank +& 1.U
      }
    }.otherwise {
      reorgBlock := reorgBlock +& 1.U
    }
  }

  val arbiter = Module(new MemoryAccessArbiter(p))
  arbiter.io.gpuRequest := io.gpuRequest
  arbiter.io.gpuBank := io.gpuBank
  arbiter.io.compactRequest := io.compactRequest || activeMapFifo.io.dequeue.valid
  arbiter.io.compactBank := Mux(activeMapFifo.io.dequeue.valid, activeMapFifo.io.dequeue.bits.bank, io.compactRequestBank)
  io.grantGpu := arbiter.io.grantGpu
  io.grantCompaction := arbiter.io.grantCompaction

  val gatherWriter = Module(new HcfGatherWriter(p))
  val blockCompletionQueue = Module(new Queue(new HcfBlockCompletion(p), p.maxBlocksPerTask))
  val gatherCanAdvance = blockCompletionQueue.io.enq.ready
  gatherWriter.io.block.valid := activeMapFifo.io.dequeue.valid && arbiter.io.grantCompaction && gatherCanAdvance
  gatherWriter.io.block.bits := activeMapFifo.io.dequeue.bits
  activeMapFifo.io.dequeue.ready := gatherWriter.io.block.ready && arbiter.io.grantCompaction && gatherCanAdvance
  gatherWriter.io.destinationLine := reorgController.io.activeDestinationLine
  gatherWriter.io.releaseValid := io.hcfReleaseValid
  gatherWriter.io.releaseStart := io.hcfReleaseStart
  gatherWriter.io.releaseCount := io.hcfReleaseCount
  io.hcfGpuFallback.valid := gatherWriter.io.gpuFallback.valid
  io.hcfGpuFallback.bits := gatherWriter.io.gpuFallback.bits
  gatherWriter.io.gpuFallback.ready := io.hcfGpuFallback.ready
  blockCompletionQueue.io.enq.valid := gatherWriter.io.block.fire
  blockCompletionQueue.io.enq.bits.taskId := activeMapFifo.io.dequeue.bits.taskId
  blockCompletionQueue.io.enq.bits.die := activeMapFifo.io.dequeue.bits.die
  blockCompletionQueue.io.enq.bits.bank := activeMapFifo.io.dequeue.bits.bank
  blockCompletionQueue.io.enq.bits.block := activeMapFifo.io.dequeue.bits.block
  blockCompletionQueue.io.enq.bits.sourceLine := gatherWriter.io.sourceLine
  blockCompletionQueue.io.enq.bits.destinationLine := reorgController.io.activeDestinationLine
  blockCompletionQueue.io.enq.bits.destinationStart := gatherWriter.io.destinationStart
  blockCompletionQueue.io.enq.bits.outputCount := gatherWriter.io.outputCount
  blockCompletionQueue.io.enq.bits.overflow := gatherWriter.io.overflow
  io.hcfBlockCompletion <> blockCompletionQueue.io.deq
  io.hcfGranted := gatherWriter.io.outputCount
  io.hcfOverflow := gatherWriter.io.overflow
  io.hcfBlockComplete := blockCompletionQueue.io.enq.fire
  io.hcfFinalComplete := gatherWriter.io.finalComplete
  io.hcfCompletionTaskId := reorgController.io.activeTaskId
  io.hcfSourceLine := gatherWriter.io.sourceLine
  io.hcfDestinationBase := gatherWriter.io.destinationBase
  io.hcfDestinationStart := gatherWriter.io.destinationStart
  io.hcfTaskRangeStart := gatherWriter.io.taskRangeStart
  io.hcfBlockOutputCount := gatherWriter.io.outputCount
  io.hcfFinalOutputCount := gatherWriter.io.finalOutputCount
  io.hcfBindingSourceAddress := gatherWriter.io.bindingSourceAddress
  io.hcfBindingSourceAddressValid := gatherWriter.io.bindingSourceAddressValid

  val selectionOutputCount = (0 until p.pimBanksPerDie).flatMap { bank =>
    (0 until p.maxBlocksPerTask).map { block =>
      val bankInScope = bank.U >= selectController.io.activeBank && bank.U < selectController.io.activeBank +& selectController.io.activeBankCount
      val blockInScope = block.U >= selectController.io.activeBlockStart &&
        block.U < selectController.io.activeBlockStart +& selectController.io.activeBlockCount
      Mux(bankInScope && blockInScope, PopCount(selectionMaps(bank)(block)), 0.U)
    }
  }.reduce(_ +& _)
  selectController.io.executionDone := selectionReady
  selectController.io.executionOutputCount := selectionOutputCount
  selectController.io.executionDestinationStart := 0.U
  selectController.io.executionOverflow := false.B
  reorgController.io.executionDone := gatherWriter.io.finalComplete
  reorgController.io.executionOutputCount := gatherWriter.io.finalOutputCount
  reorgController.io.executionDestinationStart := gatherWriter.io.taskRangeStart
  reorgController.io.executionOverflow := gatherWriter.io.finalOverflow
}
