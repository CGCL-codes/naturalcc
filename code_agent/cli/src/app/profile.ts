import type { ModelSystemMessage } from '../model/types/types.js'

export interface CodingAgentProfile {
  readonly name: 'coding-agent'
  readonly systemMessage: ModelSystemMessage
}

export function createCodingAgentProfile(systemPrompt: string): CodingAgentProfile {
  return Object.freeze({
    name: 'coding-agent' as const,
    systemMessage: {
      role: 'system' as const,
      content: systemPrompt
        ? [{ type: 'text' as const, text: systemPrompt }]
        : [],
    },
  })
}
