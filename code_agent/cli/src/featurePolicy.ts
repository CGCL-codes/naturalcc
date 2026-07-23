interface FeaturePolicy {
  allowEmptyInstruction?: boolean
  emptyInstructionMessage?: string
}

const FEATURE_POLICIES: Record<string, FeaturePolicy> = {
  knowledge_graph: {
    allowEmptyInstruction: true,
    emptyInstructionMessage: 'Knowledge graph generation',
  },
}

export function canRunWithoutInstruction(feature: string): boolean {
  return FEATURE_POLICIES[feature]?.allowEmptyInstruction === true
}

export function displayInstruction(feature: string, instruction: string): string {
  if (instruction.trim()) return instruction
  return FEATURE_POLICIES[feature]?.emptyInstructionMessage ?? instruction
}
