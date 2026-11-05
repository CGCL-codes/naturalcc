export interface ContextBuildInput {
  projectDir: string
  now: Date
  /** IANA time zone used to derive currentDate from now. */
  timeZone?: string
  signal?: AbortSignal
}

export type GitUnavailableReason =
  | 'command_missing'
  | 'command_failed'
  | 'timeout'
  | 'output_limit'
  | 'invalid_utf8'

export type GitStatusResult =
  | { readonly kind: 'clean' }
  | { readonly kind: 'dirty'; readonly status: string }
  | { readonly kind: 'non_git' }
  | { readonly kind: 'unavailable'; readonly reason: GitUnavailableReason }

export interface GitSessionSnapshot {
  readonly workspaceIdentity: string
  readonly sampledAt: string
  readonly result: GitStatusResult
}

export interface SessionEnvironmentFacts {
  readonly workspaceIdentity: string
  readonly projectName: string
  readonly localDate: string
  readonly timeZone: string
  readonly git: GitSessionSnapshot
}

export type GitStatusSampler = (
  projectDir: string,
  sampledAt: string,
  signal?: AbortSignal,
) => Promise<GitSessionSnapshot>

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
  /** Trusted built-in Session facts; extension strings cannot override this. */
  environment?: SessionEnvironmentFacts
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
  environment: SessionEnvironmentFacts
}

export interface AgentPromptContext {
  projectMemory: string
  gitStatus: string
  projectName: string
  currentDate: string
  environment?: SessionEnvironmentFacts
  warnings: readonly string[]
  sections: readonly ContextSection[]
}
