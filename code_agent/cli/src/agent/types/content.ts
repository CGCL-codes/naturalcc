// [maint-note:12571c77e021]
// 数据契约导航：ContextBuildInput；字段含义以消费端校验为准。
// 成员 projectDir：string。
// 成员 now：Date。
// 成员 timeZone（可省略）：string。
// 成员 signal（可省略）：AbortSignal。
// [/maint-note:12571c77e021]
export interface ContextBuildInput {
  projectDir: string
  now: Date
  /** IANA time zone used to derive currentDate from now. */
  timeZone?: string
  signal?: AbortSignal
}

// [maint-note:b2af698f5d3f]
// 数据契约导航：GitUnavailableReason；字段含义以消费端校验为准。
// 类型组合：| 'command_missing' | 'command_failed' | 'timeout' | 'output_limit' | 'invalid_utf8'。
// [/maint-note:b2af698f5d3f]
export type GitUnavailableReason =
  | 'command_missing'
  | 'command_failed'
  | 'timeout'
  | 'output_limit'
  | 'invalid_utf8'

// [maint-note:30d0a9fa0444]
// 数据契约导航：GitStatusResult；字段含义以消费端校验为准。
// 类型组合：| { readonly kind: 'clean' } | { readonly kind: 'dirty'; readonly status: string } | { readonly kind: 'non_git' } | { readonly kind: 'unavailable'; readonly reason: GitUnavailableReason }。
// [/maint-note:30d0a9fa0444]
export type GitStatusResult =
  | { readonly kind: 'clean' }
  | { readonly kind: 'dirty'; readonly status: string }
  | { readonly kind: 'non_git' }
  | { readonly kind: 'unavailable'; readonly reason: GitUnavailableReason }

// [maint-note:eabf2f8ad8c6]
// 数据契约导航：GitSessionSnapshot；字段含义以消费端校验为准。
// 成员 workspaceIdentity：string。
// 成员 sampledAt：string。
// 成员 result：GitStatusResult。
// [/maint-note:eabf2f8ad8c6]
export interface GitSessionSnapshot {
  readonly workspaceIdentity: string
  readonly sampledAt: string
  readonly result: GitStatusResult
}

// [maint-note:8ddffb674f4c]
// 数据契约导航：SessionEnvironmentFacts；字段含义以消费端校验为准。
// 成员 workspaceIdentity：string。
// 成员 projectName：string。
// 成员 localDate：string。
// 成员 timeZone：string。
// 成员 git：GitSessionSnapshot。
// [/maint-note:8ddffb674f4c]
export interface SessionEnvironmentFacts {
  readonly workspaceIdentity: string
  readonly projectName: string
  readonly localDate: string
  readonly timeZone: string
  readonly git: GitSessionSnapshot
}

// [maint-note:152b0b869b79]
// 数据契约导航：GitStatusSampler；字段含义以消费端校验为准。
// 类型组合：( projectDir: string, sampledAt: string, signal?: AbortSignal, ) => Promise<GitSessionSnapshot>。
// [/maint-note:152b0b869b79]
export type GitStatusSampler = (
  projectDir: string,
  sampledAt: string,
  signal?: AbortSignal,
) => Promise<GitSessionSnapshot>

// [maint-note:c8c607fdbbd8]
// 数据契约导航：ContextSection；字段含义以消费端校验为准。
// 成员 tag：string。
// 成员 content：string。
// 成员 maxChars（可省略）：number。
// [/maint-note:c8c607fdbbd8]
export interface ContextSection {
  readonly tag: string
  readonly content: string
  readonly maxChars?: number
}

// [maint-note:cc2143f1998a]
// 数据契约导航：ContextContribution；字段含义以消费端校验为准。
// 成员 projectMemory（可省略）：string。
// 成员 gitStatus（可省略）：string。
// 成员 projectName（可省略）：string。
// 成员 currentDate（可省略）：string。
// 成员 environment（可省略）：SessionEnvironmentFacts。
// 成员 warnings（可省略）：readonly string[]。
// 成员 sections（可省略）：readonly ContextSection[]。
// [/maint-note:cc2143f1998a]
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
// [maint-note:87a42adcfd67]
// 数据契约导航：ContextProvider；字段含义以消费端校验为准。
// 成员 name：string。
// 成员 build：Promise<ContextContribution>。
// [/maint-note:87a42adcfd67]
export interface ContextProvider {
  readonly name: string
  build(input: ContextBuildInput): Promise<ContextContribution>
}

// [maint-note:1250978adecf]
// 数据契约导航：DynamicContext；字段含义以消费端校验为准。
// 成员 gitStatus：string。
// 成员 projectName：string。
// 成员 currentDate：string。
// 成员 environment：SessionEnvironmentFacts。
// [/maint-note:1250978adecf]
export interface DynamicContext {
  gitStatus: string
  projectName: string
  currentDate: string
  environment: SessionEnvironmentFacts
}

// [maint-note:b448ea249bf2]
// 数据契约导航：AgentPromptContext；字段含义以消费端校验为准。
// 成员 projectMemory：string。
// 成员 gitStatus：string。
// 成员 projectName：string。
// 成员 currentDate：string。
// 成员 environment（可省略）：SessionEnvironmentFacts。
// 成员 warnings：readonly string[]。
// 成员 sections：readonly ContextSection[]。
// [/maint-note:b448ea249bf2]
export interface AgentPromptContext {
  projectMemory: string
  gitStatus: string
  projectName: string
  currentDate: string
  environment?: SessionEnvironmentFacts
  warnings: readonly string[]
  sections: readonly ContextSection[]
}
