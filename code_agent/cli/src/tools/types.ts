export interface ToolSchema {
  name: string
  description?: string
  parameters: Record<string, unknown>
  readOnly?: boolean
  requiresApproval?: boolean
}

export class ToolInputError extends Error {
  readonly code = 'tool_validation_error' as const

  constructor(message: string) {
    super(message)
    this.name = 'ToolInputError'
  }
}

export interface ToolCall {
  id: string
  name: string
  arguments: Record<string, unknown>
  rawArguments?: string
  argumentsError?: string
}

export interface ToolExecutionContext {
  projectDir: string
  signal?: AbortSignal
  timeoutMs?: number
  maxOutputChars?: number
}

export interface ToolExecutionResult {
  content: string
}

export interface Tool extends ToolSchema {
  parse?(input: Record<string, unknown>): unknown
  execute(
    input: Record<string, unknown>,
    context: ToolExecutionContext,
  ): Promise<ToolExecutionResult>
}

export interface ToolRegistry {
  schemas(): ToolSchema[]
  validate?(call: ToolCall): void
  execute(
    call: ToolCall,
    context: ToolExecutionContext,
  ): Promise<ToolExecutionResult>
}
