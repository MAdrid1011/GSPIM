package gspim

case class GspimParams(
  packages: Int = 8,
  diesPerPackage: Int = 4,
  banksPerDie: Int = 16,
  pimBanksPerDie: Int = 12,
  reservedBanksPerDie: Int = 4,
  operandWidth: Int = 26,
  fractionalBits: Int = 16,
  blockRecords: Int = 16,
  programSlots: Int = 4,
  microProgramLength: Int = 5,
  maxBlocksPerTask: Int = 4,
  maxBindingsPerRecord: Int = 4,
  rowFieldSlots: Int = 32,
  microInstructionWidth: Int = 32
) {
  require(pimBanksPerDie + reservedBanksPerDie == banksPerDie)
  require(operandWidth == 26)
  require(fractionalBits > 0 && fractionalBits < operandWidth)
  require(programSlots > 0 && (programSlots & (programSlots - 1)) == 0)
  require(microProgramLength > 0)
  require(maxBlocksPerTask > 0)
  require(maxBindingsPerRecord > 0)
  require(rowFieldSlots >= 25)
  require(microInstructionWidth > 0)
}
