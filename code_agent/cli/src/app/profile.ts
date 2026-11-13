import type { ModelSystemMessage } from '../model/types/types.js'

// [maint-note:7043a95486ff]
// 数据契约导航：CodingAgentProfile；字段含义以消费端校验为准。
// 成员 name：'coding-agent'。
// 成员 systemMessage：ModelSystemMessage。
// [/maint-note:7043a95486ff]
export interface CodingAgentProfile {
  readonly name: 'coding-agent'
  readonly systemMessage: ModelSystemMessage
}

// [maint-note:039f911ddc05]
// 维护导航：createCodingAgentProfile；下列为本函数源码中的接口与路径索引。
// 输入 systemPrompt：string。
// 返回契约：CodingAgentProfile。
// 返回路径：Object.freeze({ name: 'coding-agent' as const, systemMessage: { role: 'system' as const, content: systemPrompt ? [{ type: 'text' as const, text: systemPrompt }] : [], }, })。
// 直接调用：Object.freeze。
// [/maint-note:039f911ddc05]
export function createCodingAgentProfile(systemPrompt: string): CodingAgentProfile {
  return Object.freeze({
    name: 'coding-agent' as const,
    systemMessage: {
      role: 'system' as const,
      content: systemPrompt
        ? [{ type: 'text' as const, text: systemPrompt }]
        : [],
    },
  })
}
