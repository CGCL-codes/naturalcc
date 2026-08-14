export type Role = 'user' | 'assistant' | 'system' | 'tool'
export type MessageType = 'normal' | 'error'
export type MessageTone = 'normal' | 'error'
export type MessageTerminalKind = 'error' | 'interrupted'
export type ToolMessagePhase = 'running' | 'awaiting_approval' | 'succeeded' | 'failed' | 'interrupted'

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
  workspaceMayHaveChanged?: boolean
  isDraft?: boolean
  /** Internal UI correlation; never rendered as part of the message text. */
  correlationKey?: string
  toolName?: string
  toolInput?: unknown
  toolPhase?: ToolMessagePhase
  output?: string
  approvalRequestId?: string
  toolPresentation?: ToolResultPresentation
}
