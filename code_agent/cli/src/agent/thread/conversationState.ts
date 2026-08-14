import type { ModelMessage } from '../../model/types/types.js'

export interface ConversationState {
  readonly id: string
  messages: ModelMessage[]
  status: 'idle' | 'running'
}
