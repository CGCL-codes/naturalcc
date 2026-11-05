export type Role = 'user' | 'assistant' | 'system' | 'tool'
export type MessageType = 'normal' | 'error'
export type MessageTone = 'normal' | 'error'
export type MessageTerminalKind = 'error' | 'interrupted' | 'paused'
/** Domain reason enum for a paused terminal; never inferred from message text. */
export type MessageTerminalReason = 'budget_exhausted' | 'no_progress'
export type ToolMessagePhase = 'running' | 'awaiting_approval' | 'succeeded' | 'failed' | 'denied' | 'interrupted'

import type { ToolResultPresentation } from '../toolPresenters/index.js'

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
