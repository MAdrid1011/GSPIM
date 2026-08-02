package gspim

import chisel3._
import chisel3.util.log2Ceil

object PimTaskKind {
  val select: UInt = 0.U(1.W)
  val reorg: UInt = 1.U(1.W)
}

object PpimProgramId {
  val tempActivity: UInt = 0.U(2.W)
  val keyframeRange: UInt = 1.U(2.W)
  val anchorOverlap: UInt = 2.U(2.W)
  val depthStable: UInt = 3.U(2.W)
}

object PpimMicroOp {
  val nop: UInt = 0.U(4.W)
  val rotSlice: UInt = 1.U(4.W)
  val scale: UInt = 2.U(4.W)
  val dot: UInt = 3.U(4.W)
  val windowDist: UInt = 4.U(4.W)
  val compareLe: UInt = 5.U(4.W)
  val compareLt: UInt = 6.U(4.W)
  val and: UInt = 7.U(4.W)
  val compareGt: UInt = 8.U(4.W)
  val load: UInt = 9.U(4.W)
}

object ReorgPurposeId {
  val active: UInt = 0.U(2.W)
  val stable: UInt = 1.U(2.W)
  val unstable: UInt = 2.U(2.W)
  val batch: UInt = 3.U(2.W)
}

/** Reserved LPDDR control addresses are artifact parameters, never timing-model inputs. */
object LpddrControlAddress {
  val programWrite: BigInt = BigInt("ffff0000", 16)
  val layoutWrite: BigInt = BigInt("ffff0004", 16)
  val pimSelect: BigInt = BigInt("ffff0010", 16)
  val pimReorg: BigInt = BigInt("ffff0014", 16)
  val completionRead: BigInt = BigInt("ffff0020", 16)
}

class PimTaskReq(p: GspimParams) extends Bundle {
  val taskId = UInt(16.W)
  val kind = UInt(1.W)
  val die = UInt(log2Ceil(p.diesPerPackage * p.packages).W)
  // A die-level command covers a contiguous PIM-bank scope.  `bank` is its
  // first bank; `bankCount` prevents the old one-bank-per-task shortcut.
  val bank = UInt(log2Ceil(p.pimBanksPerDie).W)
  val bankCount = UInt(log2Ceil(p.pimBanksPerDie + 1).W)
  val program = UInt(log2Ceil(p.programSlots).W)
  val sourceLine = UInt(32.W)
  val destinationLine = UInt(32.W)
  val blockStart = UInt(16.W)
  val blockCount = UInt(16.W)
  val purpose = UInt(2.W)
  val dependency = UInt(16.W)
}

class PimCompletion(p: GspimParams) extends Bundle {
  val taskId = UInt(16.W)
  val die = UInt(log2Ceil(p.diesPerPackage * p.packages).W)
  val outputCount = UInt(16.W)
  val destinationLine = UInt(32.W)
  val destinationStart = UInt(16.W)
  val overflow = Bool()
}

/** One HCF-produced physical block range that may release dependent GPU work. */
class HcfBlockCompletion(p: GspimParams) extends Bundle {
  val taskId = UInt(16.W)
  val die = UInt(log2Ceil(p.diesPerPackage * p.packages).W)
  val bank = UInt(log2Ceil(p.pimBanksPerDie).W)
  val block = UInt(16.W)
  val sourceLine = UInt(32.W)
  val destinationLine = UInt(32.W)
  val destinationStart = UInt(16.W)
  val outputCount = UInt(log2Ceil(p.blockRecords * p.maxBindingsPerRecord + 1).W)
  val overflow = Bool()
}

class PpimProgramWrite(p: GspimParams) extends Bundle {
  val bank = UInt(log2Ceil(p.pimBanksPerDie).W)
  val broadcast = Bool()
  val address = UInt(log2Ceil(p.programSlots * p.microProgramLength).W)
  // A task selects a program-buffer slot. The load-time kind records the
  // representation semantics needed by shared paths such as binding expansion.
  val kind = UInt(2.W)
  val instruction = UInt(p.microInstructionWidth.W)
}

class PpimLayoutWrite(p: GspimParams) extends Bundle {
  val bank = UInt(log2Ceil(p.pimBanksPerDie).W)
  val field = UInt(log2Ceil(PpimLayoutField.count).W)
  val index = UInt(log2Ceil(p.rowFieldSlots).W)
}

/** Logical payload of one standard WRITE to a reserved LPDDR control address. */
class LpddrControlWrite(p: GspimParams) extends Bundle {
  val address = UInt(32.W)
  val task = new PimTaskReq(p)
  val program = new PpimProgramWrite(p)
  val layout = new PpimLayoutWrite(p)
}
