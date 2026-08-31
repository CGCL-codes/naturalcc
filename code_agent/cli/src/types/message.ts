// 定义消息角色
export type Role = 'user' | 'assistant' | 'system' | 'error'

// 定义消息接口
export interface Message {
  role: Role
  content: string
  time : string
}