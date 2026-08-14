export interface ContextBuildInput {
  projectDir: string
  now: Date
  signal?: AbortSignal
}

export interface ContextSection {
  readonly tag: string
  readonly content: string
  readonly maxChars?: number
}

export interface ContextContribution {
  projectMemory?: string
  gitStatus?: string
  projectName?: string
  currentDate?: string
  warnings?: readonly string[]
  sections?: readonly ContextSection[]
}

/** Application-selected context source; future providers can add sections without changing AgentRunner. */
export interface ContextProvider {
  readonly name: string
  build(input: ContextBuildInput): Promise<ContextContribution>
}

export interface DynamicContext {
  gitStatus: string
  projectName: string
  currentDate: string
}

export interface AgentPromptContext {
  projectMemory: string
  gitStatus: string
  projectName: string
  currentDate: string
  warnings: readonly string[]
  sections: readonly ContextSection[]
}
