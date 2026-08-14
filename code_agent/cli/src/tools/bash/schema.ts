import type { ToolSchema } from '../types.js'
import { MAX_APPROVAL_REASON_CHARS } from '../approval.js'

export const bashSchema: ToolSchema = {
  name: 'bash',
  description:
    '在项目当前工作目录执行一条 shell 命令，返回 stdout、stderr、exitCode 的 JSON。适合统计文件、运行测试、查看 git 状态等。',
  readOnly: false,
  requiresApproval: true,
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
