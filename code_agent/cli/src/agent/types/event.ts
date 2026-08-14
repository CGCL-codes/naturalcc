export type AgentErrorCode =
  | 'config_error'
  | 'session_busy'
  | 'model_protocol_error'
  | 'model_request_error'
  | 'model_incomplete'
  | 'tool_not_found'
  | 'tool_validation_error'
  | 'tool_denied'
  | 'tool_timeout'
  | 'tool_execution_error'
  | 'max_turns_reached'
  | 'interrupted'

export type AgentEvent =
  | { type: 'user_message'; content: string }
  | { type: 'assistant_delta'; content: string }
  | { type: 'assistant_message'; content: string }
  | { type: 'tool_start'; callId: string; name: string; input: unknown }
  | {
      type: 'tool_result'
      callId: string
      name: string
      output: string
      isError: boolean
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
  | { type: 'done'; content: string }
