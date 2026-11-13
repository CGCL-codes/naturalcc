// [maint-note:7a0b5fb593a6]
// 数据契约导航：Role；字段含义以消费端校验为准。
// 类型组合：'user' | 'assistant' | 'system' | 'tool'。
// [/maint-note:7a0b5fb593a6]
export type Role = 'user' | 'assistant' | 'system' | 'tool'
// [maint-note:27b87b906a0f]
// 数据契约导航：MessageType；字段含义以消费端校验为准。
// 类型组合：'normal' | 'error'。
// [/maint-note:27b87b906a0f]
export type MessageType = 'normal' | 'error'
// [maint-note:67eb8eb68933]
// 数据契约导航：MessageTone；字段含义以消费端校验为准。
// 类型组合：'normal' | 'error'。
// [/maint-note:67eb8eb68933]
export type MessageTone = 'normal' | 'error'
// [maint-note:d5cbec66c6e5]
// 数据契约导航：MessageTerminalKind；字段含义以消费端校验为准。
// 类型组合：'error' | 'interrupted' | 'paused'。
// [/maint-note:d5cbec66c6e5]
export type MessageTerminalKind = 'error' | 'interrupted' | 'paused'
/** Domain reason enum for a paused terminal; never inferred from message text. */
// [maint-note:aa811cc6c5c7]
// 数据契约导航：MessageTerminalReason；字段含义以消费端校验为准。
// 类型组合：'budget_exhausted' | 'no_progress'。
// [/maint-note:aa811cc6c5c7]
export type MessageTerminalReason = 'budget_exhausted' | 'no_progress'
// [maint-note:3f258f1ebacd]
// 数据契约导航：ToolMessagePhase；字段含义以消费端校验为准。
// 类型组合：'running' | 'awaiting_approval' | 'succeeded' | 'failed' | 'denied' | 'interrupted'。
// [/maint-note:3f258f1ebacd]
export type ToolMessagePhase = 'running' | 'awaiting_approval' | 'succeeded' | 'failed' | 'denied' | 'interrupted'

import type { ToolResultPresentation } from '../toolPresenters/index.js'

// [maint-note:b3831b668c2e]
// 数据契约导航：Message；字段含义以消费端校验为准。
// 成员 id：string。
// 成员 role：Role。
// 成员 status：MessageType。
// 成员 content：string。
// 成员 time：string。
// 成员 createdAtMs：number。
// 成员 tone：MessageTone。
// 成员 terminalKind（可省略）：MessageTerminalKind。
// 成员 terminalReason（可省略）：MessageTerminalReason。
// 成员 workspaceMayHaveChanged（可省略）：boolean。
// 成员 isDraft（可省略）：boolean。
// 成员 correlationKey（可省略）：string。
// 成员 toolName（可省略）：string。
// 成员 toolInput（可省略）：unknown。
// 成员 toolPhase（可省略）：ToolMessagePhase。
// 成员 output（可省略）：string。
// 成员 approvalRequestId（可省略）：string。
// 成员 approvalDecision（可省略）：'approved' | 'denied'。
// 成员 approvalReason（可省略）：string。
// 成员 toolPresentation（可省略）：ToolResultPresentation。
// 成员 viewCollapsed（可省略）：boolean。
// [/maint-note:b3831b668c2e]
export interface Message {
  id: string
  role: Role
  status: MessageType
  content: string
  time: string
  /** Wall-clock instant captured once when this visible record/draft is created. */
  createdAtMs: number
  /** Visual style is orthogonal to terminal/tool domain state. */
  tone: MessageTone
  terminalKind?: MessageTerminalKind
  /** Set only for a paused terminal; the header renders it, never the body. */
  terminalReason?: MessageTerminalReason
  workspaceMayHaveChanged?: boolean
  isDraft?: boolean
  /** Internal UI correlation; never rendered as part of the message text. */
  correlationKey?: string
  toolName?: string
  toolInput?: unknown
  toolPhase?: ToolMessagePhase
  output?: string
  approvalRequestId?: string
  /** First approval decision for this tool call; late decisions are ignored. */
  approvalDecision?: 'approved' | 'denied'
  /** Model/provider reason retained for the denied or interrupted preview. */
  approvalReason?: string
  toolPresentation?: ToolResultPresentation
  /** this tool message has a retained detail record (see the detail store). */
  viewCollapsed?: boolean
}
