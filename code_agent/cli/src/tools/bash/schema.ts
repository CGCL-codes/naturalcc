import type { ToolSchema } from '../types.js'
import { MAX_APPROVAL_REASON_CHARS } from '../approval.js'

export const bashSchema: ToolSchema = {
  name: 'bash',
  description:
    '在项目当前工作目录执行一条 shell 命令，返回 stdout、stderr、exitCode 的 JSON。适合运行测试、构建、项目脚本等结构化工具无法表达的 shell 工作。查找、列举、搜索或统计项目文件时优先使用 list_files/search_text，不要仅因熟悉 find/rg/grep 绕过等价的结构化工具；执行 bash 会进入本地审批策略。',
  readOnly: false,
  requiresApproval: true,
  repeatKind: 'opaque',
  parameters: {
    type: 'object',
    properties: {
      command: {
        type: 'string',
        description: '要执行的 shell 命令，一条字符串',
      },
      reason: {
        type: 'string',
        minLength: 1,
        maxLength: MAX_APPROVAL_REASON_CHARS,
        description: '用一句简洁、具体、面向用户的话说明为什么必须执行这条命令，不得编造结果',
      },
    },
    required: ['command', 'reason'],
    additionalProperties: false,
  },
}
