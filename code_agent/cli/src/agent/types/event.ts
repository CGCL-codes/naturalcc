export type AgentErrorCode =
  | 'config_error'
  | 'session_busy'
  | 'model_protocol_error'
  | 'model_request_error'
  | 'model_incomplete'
  | 'tool_not_found'
  | 'tool_validation_error'
  | 'tool_denied'
  | 'approval_provider_error'
  | 'tool_timeout'
  | 'tool_execution_error'
  | 'max_turns_reached'
  | 'request_budget_exhausted'
  | 'interrupted'

export type AgentEvent =
  | { type: 'user_message'; content: string }
  | { type: 'assistant_delta'; content: string }
  /** Completed model text that precedes a tool-call batch. Not terminal. */
  | { type: 'assistant_segment'; content: string }
  | { type: 'assistant_message'; content: string }
  | { type: 'tool_start'; callId: string; name: string; input: unknown }
  | {
      type: 'tool_result'
      callId: string
      name: string
      output: string
      isError: boolean
      /** Canonical tool outcome code; UI state must not infer it from prose. */
      code?: string
      /**
       * UI-local display facts from the tool's canonical executor. They
       * ride the UI-facing event stream only — never AgentThread, model
       * requests, or one-shot canonical stdout/stderr.
       */
      display?: import('../../tools/types.js').ToolDisplayFacts
    }
  | {
      type: 'approval_required'
      requestId: string
      callId: string
      name: string
      input: unknown
      reason: string
    }
  | { type: 'approval_resolved'; requestId: string; approved: boolean }
  | {
      type: 'error'
      code: AgentErrorCode
      message: string
      recoverable: boolean
      /** True when a non-read-only tool may have changed the real workspace. */
      workspaceMayHaveChanged: boolean
    }
  | {
      type: 'interrupted'
      message: string
      /** Conversation rollback does not roll back an already executed tool. */
      workspaceMayHaveChanged: boolean
    }
  | {
      type: 'paused'
      reason: 'budget_exhausted' | 'no_progress'
      content: string
      /** The checkpoint is not a completion; the workspace can still be dirty. */
      workspaceMayHaveChanged: boolean
    }
  | { type: 'done'; content: string }
