import type { ModelMessage } from '../../model/types/types.js'

// [maint-note:bf15159bbce7]
// 数据契约导航：ConversationState；字段含义以消费端校验为准。
// 成员 id：string。
// 成员 messages：ModelMessage[]。
// 成员 status：'idle' | 'running'。
// [/maint-note:bf15159bbce7]
export interface ConversationState {
  readonly id: string
  messages: ModelMessage[]
  status: 'idle' | 'running'
}
